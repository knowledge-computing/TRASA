"""Execute a labeled synthetic pipeline; optional real SAM initialization enables guidance."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .benchmark import VARIANTS
from .config import Config
from .synthetic import generate
from .prepare_data import main as prepare
from .folds import main as folds
from .train_geometry import main as train_geometry
from .cache_guidance import main as cache
from .train import main as train
from .predict_dataset import main as predict
from .infer import main as infer
from .evaluate import main as evaluate
from .tables import main as tables
from .validate_data import validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sam-checkpoint', help='Real SAM ViT-B initialization for guided variants')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    variants = args.variants or (['trasa'] if args.sam_checkpoint else ['ldtr-ntc'])
    guided = any(v in ('trasa', 'ldtr-sam') for v in variants)
    if guided and not args.sam_checkpoint:
        raise ValueError('Guided demonstrations require real SAM initialization; no substitute guidance is fabricated')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Demo output exists; choose a new directory')
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    raw = output / 'raw'
    index = generate(raw, args.patch_size)
    prepared = output / 'prepared'
    prepare(['--index', str(index), '--output', str(prepared), '--coordinates', 'pixels', '--copy-images', '--patch-size', str(args.patch_size)])
    folds(['--manifest', str(prepared / 'manifest.json'), '--output', str(output / 'folds')])
    manifest = output / 'folds/fold_00.json'
    cfg = Config(patch_size=args.patch_size, hidden_dim=64, encoder_layers=1, decoder_layers=1,
                 num_queries=8, feedforward_dim=128, pretrained_backbone=False, epochs=args.epochs, batch_size=2)
    config = output / 'smoke.yaml'
    config.write_text(yaml.safe_dump(cfg.asdict()))
    report = validate_dataset(manifest, cfg, guided=False)
    if not report['valid']:
        raise ValueError(json.dumps(report))
    geometry = output / 'geometry'
    if guided:
        train_geometry(['--manifest', str(manifest), '--sam-checkpoint', args.sam_checkpoint, '--output', str(geometry),
                        '--patch-size', str(args.patch_size), '--epochs', '1', '--batch-size', '1', '--device', args.device])
        cache(['--manifest', str(manifest), '--checkpoint', str(geometry / 'best.ckpt'),
               '--output', str(output / 'guidance'), '--patch-size', str(args.patch_size), '--device', args.device])
        manifest = output / 'guidance/manifest.json'
    test_id = json.loads(manifest.read_text())['schedule']['test']
    reports = []
    for variant in variants:
        run = output / variant
        train(['--config', str(config), '--manifest', str(manifest), '--output', str(run), '--device', args.device, '--ablation', variant])
        train(['--config', str(config), '--manifest', str(manifest), '--output', str(run), '--device', args.device, '--ablation', variant,
               '--resume', str(run / 'last.pt')])
        predict(['--manifest', str(manifest), '--checkpoint', str(run / 'best.pt'), '--output', str(run / 'test'), '--device', args.device])
        command = ['--checkpoint', str(run / 'best.pt'), '--image', str(raw / f'maps/{test_id}.png'),
                   '--output', str(run / 'unseen_map.json'), '--device', args.device]
        if variant in ('trasa', 'ldtr-sam'):
            command.extend(['--geometry-checkpoint', str(geometry / 'best.ckpt')])
        infer(command)
        evaluate(['--reference', str(run / 'test/map_00_reference.json'), '--prediction', str(run / 'unseen_map.json'),
                  '--output', str(run / 'unseen_map_metrics.json')])
        reports.append(str(run / 'test/metrics.json'))
    tables(['--reports', *reports, '--output', str(output / 'tables'), '--allow-incomplete'])
    (output / 'DEMONSTRATION.json').write_text(json.dumps({'dataset_kind': 'synthetic', 'benchmark_reproduction': False,
        'patch_size': args.patch_size, 'variants': variants, 'report_artifacts': [str(Path(p).relative_to(output)) for p in reports]}, indent=2) + '\n')
    print('Synthetic execution completed. These artifacts are not paper benchmark results.')


if __name__ == '__main__':
    main()
