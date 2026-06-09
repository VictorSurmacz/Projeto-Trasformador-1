import argparse
import json
import math
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia deteccao e contagem para YOLO11n com NMS entre fatias")
    parser.add_argument("--weights", type=Path, required=True, help="Peso treinado (.pt)")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"), help="Arquivo data.yaml")
    parser.add_argument("--split", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-json", type=Path, default=Path("runs/yolo11n/evaluation_metrics.json"))
    parser.add_argument(
        "--sliced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ativa modo slicing: divide cada imagem em 4 fatias, roda inferencia por fatia e aplica NMS global",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Fracao de sobreposicao usada no slicing (deve ser igual ao valor usado no slice_dataset.py)",
    )
    parser.add_argument(
        "--slice-nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold do NMS aplicado entre fatias para remover deteccoes duplicadas",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NMS entre fatias
# ---------------------------------------------------------------------------

def nms_boxes(boxes_with_scores, iou_threshold):
    """
    Aplica NMS em uma lista de (score, [x1, y1, x2, y2]).
    Retorna lista de boxes sobreviventes em coordenadas absolutas.
    """
    if not boxes_with_scores:
        return []

    # Ordena por score decrescente
    sorted_boxes = sorted(boxes_with_scores, key=lambda x: x[0], reverse=True)

    kept = []
    suppressed = set()

    for i, (score_i, box_i) in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        kept.append(box_i)
        for j, (score_j, box_j) in enumerate(sorted_boxes):
            if j <= i or j in suppressed:
                continue
            if iou_xyxy(box_i, box_j) >= iou_threshold:
                suppressed.add(j)

    return kept


def get_slice_coords(img_w, img_h, overlap):
    """
    Retorna as coordenadas (x0, y0, x1, y1) de cada uma das 4 fatias,
    usando a mesma logica do slice_dataset.py.
    """
    slice_w = int(img_w * (0.5 + overlap / 2))
    slice_h = int(img_h * (0.5 + overlap / 2))

    starts = [
        (0, 0),
        (img_w - slice_w, 0),
        (0, img_h - slice_h),
        (img_w - slice_w, img_h - slice_h),
    ]

    coords = []
    for x0, y0 in starts:
        x1 = x0 + slice_w
        y1 = y0 + slice_h
        coords.append((x0, y0, x1, y1))
    return coords


# ---------------------------------------------------------------------------
# Leitura de dados
# ---------------------------------------------------------------------------

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


def extract_pred_boxes_with_scores(result):
    """Retorna lista de (score, [x1, y1, x2, y2]) para classe 0."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    scores = result.boxes.conf.tolist()

    boxes = []
    for box, cls_id, score in zip(xyxy, classes, scores):
        if int(cls_id) != 0:
            continue
        boxes.append((score, box))
    return boxes


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def predict_full_image_with_scores(model, image_path, conf, imgsz, device):
    """Inferencia normal na imagem inteira. Retorna lista de (score, box)."""
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    return extract_pred_boxes_with_scores(result)


def predict_full_image(model, image_path, conf, imgsz, device):
    boxes_with_scores = predict_full_image_with_scores(model, image_path, conf, imgsz, device)
    return [box for _, box in boxes_with_scores]


def predict_sliced_with_scores(model, image_path, conf, imgsz, device, overlap, slice_nms_iou):
    """
    Divide a imagem em 4 fatias com sobreposicao, roda inferencia em cada uma,
    converte as coordenadas de volta para a imagem original e aplica NMS global.
    Retorna lista de (score, box).
    """
    with Image.open(image_path) as img:
        img_w, img_h = img.size
        slice_coords = get_slice_coords(img_w, img_h, overlap)

        all_boxes_with_scores = []

        for x0, y0, x1, y1 in slice_coords:
            fatia = img.crop((x0, y0, x1, y1))

            result = model.predict(
                source=fatia,
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]

            boxes_with_scores = extract_pred_boxes_with_scores(result)

            for score, box in boxes_with_scores:
                bx1, by1, bx2, by2 = box
                abs_box = [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
                all_boxes_with_scores.append((score, abs_box))

    # NMS global para remover duplicatas nas regioes de sobreposicao
    kept_boxes = nms_boxes(all_boxes_with_scores, iou_threshold=slice_nms_iou)
    # Recupera scores dos boxes mantidos
    result_with_scores = []
    for kept_box in kept_boxes:
        for score, box in all_boxes_with_scores:
            if box == kept_box:
                result_with_scores.append((score, kept_box))
                break
    return result_with_scores


def predict_sliced(model, image_path, conf, imgsz, device, overlap, slice_nms_iou):
    boxes_with_scores = predict_sliced_with_scores(model, image_path, conf, imgsz, device, overlap, slice_nms_iou)
    return [box for _, box in boxes_with_scores]


# ---------------------------------------------------------------------------
# Avaliacao por split
# ---------------------------------------------------------------------------

def compute_ap(precisions, recalls):
    """
    Calcula a Average Precision (AP) usando interpolacao de 11 pontos.
    precisions e recalls sao listas ordenadas por threshold decrescente.
    """
    ap = 0.0
    for thr in [i / 10.0 for i in range(11)]:
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr) if prec_at_thr else 0.0
    return ap / 11.0


def compute_map50(pred_boxes_per_image, gt_boxes_per_image, iou_threshold):
    """
    Calcula mAP50 acumulando TP/FP por score e construindo a curva P-R.
    pred_boxes_per_image: lista de listas de (score, box)
    gt_boxes_per_image:   lista de listas de boxes gt
    """
    # Coleta todas as deteccoes com score
    all_detections = []  # (score, image_idx, box)
    for img_idx, boxes in enumerate(pred_boxes_per_image):
        for score, box in boxes:
            all_detections.append((score, img_idx, box))

    # Ordena por score decrescente
    all_detections.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(gt) for gt in gt_boxes_per_image)
    if total_gt == 0:
        return 0.0

    matched_gt = [set() for _ in gt_boxes_per_image]
    tp_list = []
    fp_list = []

    for score, img_idx, pred_box in all_detections:
        gt_boxes = gt_boxes_per_image[img_idx]
        best_iou = 0.0
        best_gi = -1

        for gi, gt_box in enumerate(gt_boxes):
            if gi in matched_gt[img_idx]:
                continue
            iou = iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img_idx].add(best_gi)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Curva P-R acumulada
    tp_cum = []
    fp_cum = []
    running_tp = 0
    running_fp = 0
    for tp, fp in zip(tp_list, fp_list):
        running_tp += tp
        running_fp += fp
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    precisions = [tp / (tp + fp) if (tp + fp) > 0 else 0.0 for tp, fp in zip(tp_cum, fp_cum)]
    recalls = [tp / total_gt for tp in tp_cum]

    return compute_ap(precisions, recalls)


def evaluate_split(
    model, split_name, images_dir, labels_dir,
    iou_threshold, conf, imgsz, device,
    sliced, overlap, slice_nms_iou,
):
    image_paths = sorted([p for p in images_dir.glob("*") if p.is_file()])

    tp_total = 0
    fp_total = 0
    fn_total = 0
    abs_errors = []
    sq_errors = []

    # Para mAP50
    pred_boxes_per_image = []  # lista de (score, box)
    gt_boxes_per_image = []

    skipped = 0
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        try:
            gt_boxes = read_gt_boxes(label_path, image_path)
        except Exception as e:
            print(f"  [aviso] Pulando imagem corrompida: {image_path.name} ({e})")
            skipped += 1
            continue

        gt_boxes_per_image.append(gt_boxes)

        try:
            if sliced:
                pred_boxes_with_scores = predict_sliced_with_scores(
                    model, image_path, conf, imgsz, device, overlap, slice_nms_iou
                )
            else:
                pred_boxes_with_scores = predict_full_image_with_scores(
                    model, image_path, conf, imgsz, device
                )
        except Exception as e:
            print(f"  [aviso] Erro na inferencia: {image_path.name} ({e})")
            skipped += 1
            gt_boxes_per_image.pop()
            continue

        pred_boxes_per_image.append(pred_boxes_with_scores)
        pred_boxes = [box for _, box in pred_boxes_with_scores]

        tp, fp, fn = greedy_match(pred_boxes, gt_boxes, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        count_error = abs(len(pred_boxes) - len(gt_boxes))
        abs_errors.append(count_error)
        sq_errors.append(count_error ** 2)

    if skipped:
        print(f"  [aviso] {skipped} imagem(ns) ignorada(s) por erro")

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0
    map50 = compute_map50(pred_boxes_per_image, gt_boxes_per_image, iou_threshold)

    return {
        "split": split_name,
        "images": len(image_paths),
        "sliced": sliced,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "mae": mae,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Peso nao encontrado: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml nao encontrado: {args.data}")

    data_cfg = load_data_config(args.data)
    model = YOLO(str(args.weights))

    modo = "sliced" if args.sliced else "full"
    print(f"Modo de inferencia: {modo}")

    metrics = {
        "weights": str(args.weights),
        "data": str(args.data),
        "iou_threshold": args.iou_threshold,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "sliced": args.sliced,
        "overlap": args.overlap if args.sliced else None,
        "slice_nms_iou": args.slice_nms_iou if args.sliced else None,
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
            sliced=args.sliced,
            overlap=args.overlap,
            slice_nms_iou=args.slice_nms_iou,
        )
        metrics["results"].append(split_metrics)

        print(
            f"[{split}] images={split_metrics['images']} "
            f"P={split_metrics['precision']:.4f} "
            f"R={split_metrics['recall']:.4f} "
            f"F1={split_metrics['f1']:.4f} "
            f"mAP50={split_metrics['map50']:.4f} "
            f"MAE={split_metrics['mae']:.4f} "
            f"RMSE={split_metrics['rmse']:.4f}"
        )

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Metricas salvas em: {args.save_json}")


if __name__ == "__main__":
    main()