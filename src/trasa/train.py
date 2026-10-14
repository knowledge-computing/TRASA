"""Train the TRASA graph model and controlled ablation variants."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import PatchDataset, collate_patches
from .geometry import SAMRoadGeometry
from .losses import GraphCriterion
from .model import TRASA
from .runtime import load_model, restore_training_state, save_checkpoint, seed_everything, sha256
from .provenance import partitions, record_run, source_fingerprint, write_json
from .validate_data import validate_dataset


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    totals, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            rgb = batch['rgb'].to(device)
            masks = None if batch['guidance'] is None else batch['guidance'].to(device)
            outputs = model(rgb, masks)
            losses = criterion(model, outputs, batch['targets'])
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError('Non-finite training objective')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0) + float(value.detach()) * len(rgb)
            count += len(rgb)
    return {name: value / count for name, value in totals.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume')
    parser.add_argument('--geometry-checkpoint')
    parser.add_argument('--sam-variant', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--sam-decoder', action='store_true')
    parser.add_argument('--ablation', choices=['ldtr', 'ldtr-sam', 'ldtr-ntc', 'trasa'], default='trasa')
    parser.add_argument('--max-epochs-this-run', type=int, help='Stop cleanly after this many epochs; retain the full configuration for resume')
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    config = replace(config, guidance=args.ablation in ('trasa', 'ldtr-sam'),
                     ntc=args.ablation in ('trasa', 'ldtr-ntc'))
    seed_everything(config.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / 'last.pt').exists() or (output / 'history.jsonl').exists()):
        raise ValueError('Output contains a run; use --resume or a new output directory')
    online = bool(args.geometry_checkpoint)
    train = PatchDataset(args.manifest, 'train', config, not online)
    val = PatchDataset(args.manifest, 'val', config, not online)
    report = validate_dataset(args.manifest, config, splits=('train', 'val'), guided=config.guidance and not online)
    if not report['valid']:
        reason = 'Resume data provenance changed' if args.resume else 'Training data validation failed'
        raise ValueError(f'{reason}: {json.dumps(report)}')
    provenance = {'manifest_sha256': sha256(args.manifest),
                  'train_content_sha256': train.fingerprint(), 'val_content_sha256': val.fingerprint(),
                  'guidance_provenance': train.guidance_provenance,
                  'torch': str(torch.__version__), 'device': args.device,
                  'map_partitions': partitions(train.document), 'dataset_kind': train.document.get('dataset_kind', 'unverified'),
                  'variant': args.ablation, 'implementation_sha256': source_fingerprint()}
    if online:
        expected = (train.guidance_provenance or {}).get('checkpoint_sha256')
        if expected != sha256(args.geometry_checkpoint):
            raise ValueError('Online geometry checkpoint hash differs from manifest')
        provenance['geometry_checkpoint_sha256'] = expected
    generator = torch.Generator().manual_seed(config.seed)
    # num_workers=0 intentionally keeps reproducible loader/RNG resume simple.
    loader = DataLoader(train, config.batch_size, shuffle=True, collate_fn=collate_patches, generator=generator)
    validation = DataLoader(val, config.batch_size, shuffle=False, collate_fn=collate_patches)
    start, best = 0, float('inf')
    if args.resume:
        model, state = load_model(args.resume, args.device)
        if model.config != config or state['provenance'] != provenance:
            raise ValueError('Resume configuration or data provenance changed')
        if online != (model.geometry is not None):
            raise ValueError('Resume must retain cached/online guidance mode')
        start, best = state['epoch'] + 1, state['best_val']
    else:
        geometry = None
        if online:
            if not config.guidance:
                raise ValueError('Geometry checkpoint supplied to RGB-only ablation')
            geometry = SAMRoadGeometry(config.patch_size, args.sam_variant, args.sam_decoder).load_checkpoint(args.geometry_checkpoint)
            geometry.validate_provenance(train.guidance_provenance)
        model = TRASA(config, geometry).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    if args.resume:
        restore_training_state(state, optimizer, generator)
    criterion = GraphCriterion(config)
    record_run(output, config.asdict(), provenance, args.device)
    if args.max_epochs_this_run is not None and args.max_epochs_this_run < 1:
        raise ValueError('max-epochs-this-run must be positive')
    stop = min(config.epochs, start + args.max_epochs_this_run) if args.max_epochs_this_run else config.epochs
    for epoch in range(start, stop):
        started = time.perf_counter()
        train_losses = run_epoch(model, loader, criterion, args.device, optimizer)
        val_losses = run_epoch(model, validation, criterion, args.device)
        from .predict_dataset import predict_dataset
        validation_results = predict_dataset(model, val, config.batch_size)
        row = {'epoch': epoch, 'train': train_losses, 'val': val_losses,
               'val_metrics': {key: item['metrics'] for key, item in validation_results.items()},
               'elapsed_seconds': time.perf_counter() - started}
        improved = val_losses['total'] < best
        best = min(best, val_losses['total'])
        save_checkpoint(output / 'last.pt', model, optimizer, epoch, generator, provenance, best)
        if improved:
            save_checkpoint(output / 'best.pt', model, optimizer, epoch, generator, provenance, best)
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        write_json(output / 'run_status.json', {'completed_epochs': epoch + 1, 'planned_epochs': config.epochs,
                                               'complete': epoch + 1 == config.epochs})
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
