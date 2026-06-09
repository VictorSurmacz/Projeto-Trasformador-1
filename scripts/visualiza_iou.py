import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Visualiza TP, FP e FN por imagem com IoU >= threshold")
    parser.add_argument("--weights", type=Path, required=True, help="Peso treinado (.pt)")
    parser.add_argument("--images", type=Path, required=True, help="Pasta com imagens")
    parser.add_argument("--labels", type=Path, required=True, help="Pasta com labels .txt")
    parser.add_argument("--output", type=Path, default=Path("runs/visualize_iou"), help="Pasta de saida")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def xywhn_to_xyxy(box, width, height):
    xc, yc, w, h = box
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes, gt_boxes, iou_threshold):
    """
    Retorna:
    - matched_pred: set de indices de pred que sao TP
    - matched_gt:   set de indices de gt que foram detectados
    """
    pairs = []
    for pi, pred in enumerate(pred_boxes):
        for gi, gt in enumerate(gt_boxes):
            iou = iou_xyxy(pred, gt)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()

    for iou_val, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    return matched_pred, matched_gt


def read_gt_boxes(label_path, img_w, img_h):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = parts
        boxes.append(xywhn_to_xyxy([float(xc), float(yc), float(w), float(h)], img_w, img_h))
    return boxes


def draw_box(img, box, color, label, thickness=3):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = max(y1 - 6, text_size[1] + 4)

    cv2.rectangle(img,
        (text_x, text_y - text_size[1] - 4),
        (text_x + text_size[0] + 4, text_y + 2),
        color, -1)
    cv2.putText(img, label, (text_x + 2, text_y), font, font_scale, (255, 255, 255), font_thickness)


def process_image(model, image_path, label_path, output_dir, iou_threshold, conf, imgsz, device):
    # Lê imagem
    img_pil = Image.open(image_path)
    img_w, img_h = img_pil.size
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Ground truth
    gt_boxes = read_gt_boxes(label_path, img_w, img_h)

    # Predições
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    pred_boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, cls_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            if int(cls_id) == 0:
                pred_boxes.append(box)

    # Matching
    matched_pred, matched_gt = greedy_match(pred_boxes, gt_boxes, iou_threshold)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    # Desenha GT boxes
    for gi, gt_box in enumerate(gt_boxes):
        if gi in matched_gt:
            # GT detectado — borda azul
            draw_box(img_cv, gt_box, (255, 100, 0), "GT", thickness=2)
        else:
            # FN — GT não detectado — borda vermelha tracejada (usamos magenta)
            draw_box(img_cv, gt_box, (255, 0, 255), "FN", thickness=2)

    # Desenha pred boxes
    for pi, pred_box in enumerate(pred_boxes):
        if pi in matched_pred:
            # TP — verde
            draw_box(img_cv, pred_box, (0, 200, 0), "TP", thickness=3)
        else:
            # FP — vermelho
            draw_box(img_cv, pred_box, (0, 0, 220), "FP", thickness=3)

    # Legenda no canto
    legend_lines = [
        f"TP={tp}  FP={fp}  FN={fn}",
        f"GT={len(gt_boxes)}  Pred={len(pred_boxes)}",
        f"IoU>={iou_threshold}  conf>={conf}",
    ]
    for i, line in enumerate(legend_lines):
        cv2.putText(img_cv, line, (10, 35 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
        cv2.putText(img_cv, line, (10, 35 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Salva
    out_path = output_dir / f"{image_path.stem}_iou.jpg"
    cv2.imwrite(str(out_path), img_cv)

    return tp, fp, fn, len(gt_boxes), len(pred_boxes)


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Peso nao encontrado: {args.weights}")
    if not args.images.exists():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    image_paths = sorted([p for p in args.images.glob("*") if p.is_file()])
    print(f"Processando {len(image_paths)} imagens...")
    print(f"Legenda: TP=verde  FP=vermelho  FN=magenta  GT=azul")
    print()

    total_tp = total_fp = total_fn = 0

    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        try:
            tp, fp, fn, gt_count, pred_count = process_image(
                model, image_path, label_path, args.output,
                args.iou_threshold, args.conf, args.imgsz, args.device
            )
            total_tp += tp; total_fp += fp; total_fn += fn
            print(f"{image_path.name}: GT={gt_count} Pred={pred_count} TP={tp} FP={fp} FN={fn}")
        except Exception as e:
            print(f"[aviso] {image_path.name}: {e}")

    print()
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Total: TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"Imagens salvas em: {args.output}")


if __name__ == "__main__":
    main()