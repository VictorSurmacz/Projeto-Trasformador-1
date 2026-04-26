import argparse
import json
import math
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia deteccao e contagem para YOLO11n")
    parser.add_argument("--weights", type=Path, required=True, help="Peso treinado (.pt)")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"), help="Arquivo data.yaml")
    parser.add_argument("--split", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-json", type=Path, default=Path("runs/yolo11n/evaluation_metrics.json"))
    return parser.parse_args()


def xywhn_to_xyxy(box, width, height):
    x_center, y_center, w_norm, h_norm = box
    w = w_norm * width
    h = h_norm * height
    x1 = (x_center * width) - (w / 2.0)
    y1 = (y_center * height) - (h / 2.0)
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(pred_boxes, gt_boxes, iou_threshold):
    pairs = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            pairs.append((iou_xyxy(pred, gt), pi, gi))

    pairs.sort(key=lambda item: item[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0

    for iou_value, pi, gi in pairs:
        if iou_value < iou_threshold:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def load_data_config(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    raw_path = Path(data.get("path", "."))
    if raw_path.is_absolute():
        dataset_root = raw_path
    else:
        candidate_yaml_relative = (data_yaml.parent / raw_path).resolve()
        candidate_cwd_relative = (Path.cwd() / raw_path).resolve()
        if candidate_yaml_relative.exists():
            dataset_root = candidate_yaml_relative
        else:
            dataset_root = candidate_cwd_relative

    return {
        "path": dataset_root,
        "train": data.get("train"),
        "val": data.get("val"),
        "test": data.get("test"),
    }


def read_gt_boxes(label_path, image_path):
    with Image.open(image_path) as image:
        width, height = image.size

    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = parts
        box = xywhn_to_xyxy([float(x), float(y), float(w), float(h)], width, height)
        boxes.append(box)
    return boxes


def extract_pred_boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()

    boxes = []
    for box, cls_id in zip(xyxy, classes):
        if int(cls_id) != 0:
            continue
        boxes.append(box)
    return boxes


def evaluate_split(model, split_name, images_dir, labels_dir, iou_threshold, conf, imgsz, device):
    image_paths = sorted([p for p in images_dir.glob("*") if p.is_file()])

    tp_total = 0
    fp_total = 0
    fn_total = 0
    abs_errors = []
    sq_errors = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = read_gt_boxes(label_path, image_path)

        result = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_boxes = extract_pred_boxes(result)

        tp, fp, fn = greedy_match(pred_boxes, gt_boxes, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        count_error = abs(len(pred_boxes) - len(gt_boxes))
        abs_errors.append(count_error)
        sq_errors.append(count_error ** 2)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0

    return {
        "split": split_name,
        "images": len(image_paths),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "rmse": rmse,
    }


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Peso nao encontrado: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml nao encontrado: {args.data}")

    data_cfg = load_data_config(args.data)
    model = YOLO(str(args.weights))

    metrics = {
        "weights": str(args.weights),
        "data": str(args.data),
        "iou_threshold": args.iou_threshold,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "results": [],
    }

    for split in args.split:
        split_rel = data_cfg.get(split)
        if not split_rel:
            continue

        images_dir = data_cfg["path"] / split_rel
        labels_dir = data_cfg["path"] / "labels" / split

        split_metrics = evaluate_split(
            model=model,
            split_name=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            iou_threshold=args.iou_threshold,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        metrics["results"].append(split_metrics)

        print(
            f"[{split}] images={split_metrics['images']} "
            f"P={split_metrics['precision']:.4f} "
            f"R={split_metrics['recall']:.4f} "
            f"F1={split_metrics['f1']:.4f} "
            f"MAE={split_metrics['mae']:.4f} "
            f"RMSE={split_metrics['rmse']:.4f}"
        )

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Metricas salvas em: {args.save_json}")


if __name__ == "__main__":
    main()
