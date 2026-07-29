import os
import sys
import json
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch


# export SAM3_REPO=/users/4/kirsa002/IMOLA/sam3
SAM3_REPO = os.environ.get("SAM3_REPO", "")
if SAM3_REPO and SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)


try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    raise ImportError(
        "Could not import SAM3. Make sure you cloned facebookresearch/sam3 "
        "and installed it with `pip install -e .`, or set SAM3_REPO to the cloned repo path."
    ) from e


ROOT = "/users/4/kirsa002/IMOLA/CA_area"

PATCH_DIR = os.path.join(ROOT, "outputs/patches_128/test")
OUT_DIR = os.path.join(ROOT, "outputs/sam3_baseline/type_specific_roads")



EXAMPLE_PATCH = os.path.join(
    ROOT,
    "outputs/patches_128/examples/example_road_patch_001.png"
)

EXAMPLE_COORDS_JSON = os.path.join(
    ROOT,
    "outputs/patches_128/examples/example_road_patch_001_lines.json"
)

PATCH_SIZE = 128
ROAD_TYPE = "road_line"
MAX_PATCHES = 200


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_pil_image(path):
    return Image.open(path).convert("RGB")


def load_line_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def draw_lines_from_json(img, line_data, color=(255, 0, 0), thickness=2):
    vis = img.copy()

    for ent in line_data.get("line_entities", []):
        if ent.get("type") not in ["road", "road_line", "road_major", "road_minor"]:
            continue

        coords = ent.get("coords", [])
        if len(coords) < 2:
            continue

        pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=thickness)

    return vis


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    vis = img.copy()

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=color, thickness=thickness)

    return vis


def make_prompt_text(road_type="road_line"):
    return (
        f"Extract {road_type} features from this topographic map patch. "
        "Focus only on road-line symbols. Ignore contour lines, rivers, text, "
        "grid lines, neatlines, and administrative boundaries."
    )


def line_entity_to_box(coords, image_size=(128, 128), padding=6, min_size=4):
    if len(coords) < 2:
        return None

    width, height = image_size

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    if x2 - x1 < min_size:
        cx = (x1 + x2) / 2
        x1 = cx - min_size / 2
        x2 = cx + min_size / 2

    if y2 - y1 < min_size:
        cy = (y1 + y2) / 2
        y1 = cy - min_size / 2
        y2 = cy + min_size / 2

    x1 = max(0, int(round(x1 - padding)))
    y1 = max(0, int(round(y1 - padding)))
    x2 = min(width - 1, int(round(x2 + padding)))
    y2 = min(height - 1, int(round(y2 + padding)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def format_example_annotation(line_data, box_padding=6):
    image_size = line_data.get("image_size", [PATCH_SIZE, PATCH_SIZE])
    width, height = image_size[0], image_size[1]

    entities = []
    boxes = []

    for k, ent in enumerate(line_data.get("line_entities", [])):
        coords = ent.get("coords", [])
        line_type = ent.get("type", "road")

        if len(coords) < 2:
            continue

        box = line_entity_to_box(
            coords=coords,
            image_size=(width, height),
            padding=box_padding
        )

        if box is None:
            continue

        entities.append({
            "id": f"line_{k}",
            "label": line_type,
            "coords": coords,
            "box": box
        })

        boxes.append(box)

    example_obj = {
        "task": "road_line_extraction",
        "patch_size": [width, height],
        "entities": entities,
        "boxes": boxes
    }

    return example_obj


def build_sam3():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading SAM3 image model on {device}...")
    model = build_sam3_image_model()
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(model)

    return model, processor, device


def tensor_to_numpy(x):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def output_to_union_mask(output, image_shape):
    h, w = image_shape[:2]

    masks = output.get("masks", None)
    if masks is None:
        return np.zeros((h, w), dtype=np.uint8), []

    masks_np = tensor_to_numpy(masks)

    if masks_np is None:
        return np.zeros((h, w), dtype=np.uint8), []

    if masks_np.ndim == 4:
        # [N, 1, H, W] -> [N, H, W]
        masks_np = masks_np.squeeze(1)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    if masks_np.ndim != 3:
        print(f"WARNING: unexpected masks shape: {masks_np.shape}")
        return np.zeros((h, w), dtype=np.uint8), []

    binary_masks = []
    union = np.zeros((masks_np.shape[-2], masks_np.shape[-1]), dtype=np.uint8)

    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        m_bin = (m > 0).astype(np.uint8)
        binary_masks.append(m_bin)
        union = np.maximum(union, m_bin)

    union = (union * 255).astype(np.uint8)

    # resize if SAM3 returns different resolution
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

    return union, binary_masks


def extract_output_boxes_scores(output):
    boxes = tensor_to_numpy(output.get("boxes", None))
    scores = tensor_to_numpy(output.get("scores", None))

    if boxes is not None:
        boxes = boxes.tolist()

    if scores is not None:
        scores = scores.tolist()

    return boxes, scores


def run_sam3_text_prompt(processor, pil_img, prompt_text):
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt_text)
    return output


def run_sam3_box_prompts(processor, pil_img, boxes):
    state = processor.set_image(pil_img)

    merged_outputs = {
        "masks": [],
        "boxes": [],
        "scores": []
    }

    for box in boxes:
        box = [float(v) for v in box]

        try:
            output = processor.set_box_prompt(state=state, box=box)
        except TypeError:
            try:
                output = processor.set_box_prompt(state=state, boxes=[box])
            except TypeError:
                output = processor.set_box_prompt(state, box)

        for key in ["masks", "boxes", "scores"]:
            if key in output and output[key] is not None:
                merged_outputs[key].append(output[key])

    final_output = {}

    for key, values in merged_outputs.items():
        if len(values) == 0:
            final_output[key] = None
            continue

        if isinstance(values[0], torch.Tensor):
            try:
                final_output[key] = torch.cat(values, dim=0)
            except Exception:
                final_output[key] = values[0]
        else:
            final_output[key] = values

    return final_output


def save_prediction_outputs(patch_path, pred, out_dir):
    stem = Path(patch_path).stem

    mask = pred["mask"]
    boxes = pred.get("boxes", None)
    scores = pred.get("scores", None)
    prompt_mode = pred.get("prompt_mode", None)
    prompt_text = pred.get("prompt_text", None)
    prompt_boxes = pred.get("prompt_boxes", None)

    mask_path = os.path.join(out_dir, f"{stem}_sam3_mask.png")
    json_path = os.path.join(out_dir, f"{stem}_sam3_output.json")

    cv2.imwrite(mask_path, mask)

    with open(json_path, "w") as f:
        json.dump({
            "patch_id": stem,
            "method": "SAM3",
            "prompt_mode": prompt_mode,
            "prompt_text": prompt_text,
            "prompt_boxes": prompt_boxes,
            "output_boxes": boxes,
            "scores": scores,
            "mask_path": mask_path
        }, f, indent=2)

    return mask_path, json_path


# def maybe_load_patch_boxes(patch_path, json_dir, box_padding):
#     if json_dir is None:
#         return None

#     stem = Path(patch_path).stem
#     json_path = os.path.join(json_dir, f"{stem}_lines.json")

#     if not os.path.exists(json_path):
#         return None

#     line_data = load_line_json(json_path)
#     ann = format_example_annotation(line_data, box_padding=box_padding)
#     return ann["boxes"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--patch_dir", default=PATCH_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)

    parser.add_argument("--example_patch", default=EXAMPLE_PATCH)
    parser.add_argument("--example_json", default=EXAMPLE_COORDS_JSON)

    parser.add_argument("--road_type", default=ROAD_TYPE)
    parser.add_argument("--max_patches", type=int, default=MAX_PATCHES)

    parser.add_argument(
        "--prompt_mode",
        default="text",
        choices=["text", "box_if_json"],
    )

    parser.add_argument(
        "--inference_json_dir",
        default=None,
    )

    parser.add_argument(
        "--box_padding",
        type=int,
        default=6,
        help="Padding around line-derived boxes in pixels."
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    example_img_np = load_image(args.example_patch)
    example_img_pil = load_pil_image(args.example_patch)
    example_line_data = load_line_json(args.example_json)

    example_annotation = format_example_annotation(
        example_line_data,
        box_padding=args.box_padding
    )

    prompt_text = make_prompt_text(args.road_type)

    print("prompt:")
    print(prompt_text)

    print("example annotation:")
    print(json.dumps(example_annotation, indent=2)[:2000])

    example_vis_lines = draw_lines_from_json(example_img_np, example_line_data)
    example_vis_lines_path = os.path.join(args.out_dir, "debug_example_patch_with_lines.png")
    cv2.imwrite(example_vis_lines_path, cv2.cvtColor(example_vis_lines, cv2.COLOR_RGB2BGR))

    example_vis_boxes = draw_boxes(example_img_np, example_annotation["boxes"])
    example_vis_boxes_path = os.path.join(args.out_dir, "debug_example_patch_with_boxes.png")
    cv2.imwrite(example_vis_boxes_path, cv2.cvtColor(example_vis_boxes, cv2.COLOR_RGB2BGR))

    example_boxes_json_path = os.path.join(args.out_dir, "debug_example_boxes_from_lines.json")
    with open(example_boxes_json_path, "w") as f:
        json.dump(example_annotation, f, indent=2)

    model, processor, device = build_sam3()

    print("running SAM3 box prompts on reference patch...")
    with torch.no_grad():
        ref_output = run_sam3_box_prompts(
            processor=processor,
            pil_img=example_img_pil,
            boxes=example_annotation["boxes"]
        )

    ref_mask, _ = output_to_union_mask(ref_output, example_img_np.shape)
    ref_boxes_out, ref_scores = extract_output_boxes_scores(ref_output)

    ref_mask_path = os.path.join(args.out_dir, "debug_reference_sam3_box_prompt_mask.png")
    cv2.imwrite(ref_mask_path, ref_mask)

    ref_output_json_path = os.path.join(args.out_dir, "debug_reference_sam3_box_prompt_output.json")
    with open(ref_output_json_path, "w") as f:
        json.dump({
            "patch_id": Path(args.example_patch).stem,
            "method": "SAM3",
            "prompt_mode": "box",
            "prompt_boxes": example_annotation["boxes"],
            "output_boxes": ref_boxes_out,
            "scores": ref_scores,
            "mask_path": ref_mask_path
        }, f, indent=2)

    # Inference patches
    patch_paths = sorted(glob.glob(os.path.join(args.patch_dir, "*.png")))
    patch_paths = patch_paths[:args.max_patches]

    print(f"found {len(patch_paths)} inference patches")

    for i, patch_path in enumerate(patch_paths):
        print(f"[{i + 1}/{len(patch_paths)}] running SAM3 on {os.path.basename(patch_path)}")

        patch_img_np = load_image(patch_path)
        patch_img_pil = load_pil_image(patch_path)

        prompt_boxes = None
        prompt_mode_used = "text"

        with torch.no_grad():
            if args.prompt_mode == "box_if_json":
                # prompt_boxes = maybe_load_patch_boxes(
                #     patch_path=patch_path,
                #     json_dir=args.inference_json_dir,
                #     box_padding=args.box_padding
                # )

                if prompt_boxes:
                    # Valid only if boxes are for this exact inference patch
                    output = run_sam3_box_prompts(
                        processor=processor,
                        pil_img=patch_img_pil,
                        boxes=prompt_boxes
                    )
                    prompt_mode_used = "box"
                else:
                    output = run_sam3_text_prompt(
                        processor=processor,
                        pil_img=patch_img_pil,
                        prompt_text=prompt_text
                    )
                    prompt_mode_used = "text"
            else:
                output = run_sam3_text_prompt(
                    processor=processor,
                    pil_img=patch_img_pil,
                    prompt_text=prompt_text
                )
                prompt_mode_used = "text"

        pred_mask, _ = output_to_union_mask(output, patch_img_np.shape)
        out_boxes, out_scores = extract_output_boxes_scores(output)

        pred = {
            "mask": pred_mask,
            "boxes": out_boxes,
            "scores": out_scores,
            "prompt_mode": prompt_mode_used,
            "prompt_text": prompt_text if prompt_mode_used == "text" else None,
            "prompt_boxes": prompt_boxes if prompt_mode_used == "box" else None
        }

        mask_path, json_path = save_prediction_outputs(
            patch_path=patch_path,
            pred=pred,
            out_dir=args.out_dir
        )

        if i < 5:
            print(" saved:", mask_path)
            print(" saved:", json_path)

    print("done")


if __name__ == "__main__":
    main()