import argparse
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Divide imagens do dataset em 4 fatias com sobreposicao")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Pasta raiz do dataset")
    parser.add_argument("--overlap", type=float, default=0.2, help="Fracao de sobreposicao entre fatias (0.0 a 0.5)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Quais splits processar",
    )
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Substitui as imagens originais pelas fatias (recomendado)",
    )
    return parser.parse_args()


def slice_image_and_labels(
    image_path: Path,
    label_path: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    overlap: float,
):
    with Image.open(image_path) as img:
        img_w, img_h = img.size

        # Calcula tamanho de cada fatia com sobreposicao
        # Com 2x2 grid e overlap, cada fatia tem tamanho ligeiramente maior que metade
        slice_w = int(img_w * (0.5 + overlap / 2))
        slice_h = int(img_h * (0.5 + overlap / 2))

        # Ponto de inicio de cada fatia (x, y)
        # 4 fatias: topo-esq, topo-dir, baixo-esq, baixo-dir
        starts = [
            (0, 0),
            (img_w - slice_w, 0),
            (0, img_h - slice_h),
            (img_w - slice_w, img_h - slice_h),
        ]

        slices = []
        for i, (x0, y0) in enumerate(starts):
            x1 = x0 + slice_w
            y1 = y0 + slice_h
            cropped = img.crop((x0, y0, x1, y1))
            slices.append((i, x0, y0, x1, y1, cropped))

    # Le anotacoes originais
    gt_boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = parts
            gt_boxes.append((int(cls_id), float(xc), float(yc), float(w), float(h)))

    stem = image_path.stem
    suffix = image_path.suffix
    saved = 0

    for i, x0, y0, x1, y1, cropped in slices:
        slice_name = f"{stem}_s{i}{suffix}"
        label_name = f"{stem}_s{i}.txt"

        # Ajusta bounding boxes para o sistema de coordenadas da fatia
        new_lines = []
        for cls_id, xc_n, yc_n, w_n, h_n in gt_boxes:
            # Converte para pixels absolutos na imagem original
            xc_px = xc_n * img_w
            yc_px = yc_n * img_h
            w_px = w_n * img_w
            h_px = h_n * img_h

            bx1 = xc_px - w_px / 2
            by1 = yc_px - h_px / 2
            bx2 = xc_px + w_px / 2
            by2 = yc_px + h_px / 2

            # Intersecta com a fatia
            inter_x1 = max(bx1, x0)
            inter_y1 = max(by1, y0)
            inter_x2 = min(bx2, x1)
            inter_y2 = min(by2, y1)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue  # Box fora da fatia

            # Verifica se pelo menos 50% da box original esta dentro da fatia
            # (evita incluir boxes cortadas demais)
            orig_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            if inter_area / orig_area < 0.5:
                continue

            # Converte para coordenadas relativas a fatia
            new_xc = ((inter_x1 + inter_x2) / 2 - x0) / slice_w
            new_yc = ((inter_y1 + inter_y2) / 2 - y0) / slice_h
            new_w = (inter_x2 - inter_x1) / slice_w
            new_h = (inter_y2 - inter_y1) / slice_h

            # Garante que esta dentro dos limites [0, 1]
            new_xc = max(0.0, min(1.0, new_xc))
            new_yc = max(0.0, min(1.0, new_yc))
            new_w = max(0.0, min(1.0, new_w))
            new_h = max(0.0, min(1.0, new_h))

            new_lines.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

        # Salva fatia e label
        cropped.save(out_images_dir / slice_name)
        (out_labels_dir / label_name).write_text("\n".join(new_lines), encoding="utf-8")
        saved += 1

    return saved


def process_split(dataset: Path, split: str, overlap: float, replace: bool):
    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split

    if not images_dir.exists():
        print(f"[{split}] Pasta nao encontrada: {images_dir}")
        return

    image_files = sorted([p for p in images_dir.glob("*") if p.is_file()])
    if not image_files:
        print(f"[{split}] Nenhuma imagem encontrada.")
        return

    if replace:
        # Processa em pasta temporaria e depois substitui
        tmp_images = dataset / f"_tmp_images_{split}"
        tmp_labels = dataset / f"_tmp_labels_{split}"
        tmp_images.mkdir(parents=True, exist_ok=True)
        tmp_labels.mkdir(parents=True, exist_ok=True)

        total_slices = 0
        skipped = 0
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            try:
                saved = slice_image_and_labels(image_path, label_path, tmp_images, tmp_labels, overlap)
                total_slices += saved
            except Exception as e:
                print(f"  [aviso] Pulando imagem corrompida: {image_path.name} ({e})")
                skipped += 1

        # Remove originais e move fatias para o lugar
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)
        tmp_images.rename(images_dir)
        tmp_labels.rename(labels_dir)

        if skipped:
            print(f"[{split}] {skipped} imagem(ns) corrompida(s) ignorada(s)")
    else:
        total_slices = 0
        skipped = 0
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            try:
                saved = slice_image_and_labels(image_path, label_path, images_dir, labels_dir, overlap)
                total_slices += saved
            except Exception as e:
                print(f"  [aviso] Pulando imagem corrompida: {image_path.name} ({e})")
                skipped += 1

        if skipped:
            print(f"[{split}] {skipped} imagem(ns) corrompida(s) ignorada(s)")

    print(f"[{split}] {len(image_files)} imagens → {total_slices} fatias (overlap={overlap})")


def main():
    args = parse_args()

    if not 0.0 <= args.overlap < 0.5:
        raise ValueError("--overlap deve estar entre 0.0 e 0.49")

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {args.dataset}")

    print(f"Iniciando slicing 2x2 com overlap={args.overlap} nos splits: {args.splits}")

    for split in args.splits:
        process_split(args.dataset, split, args.overlap, args.replace)

    print("Slicing finalizado.")


if __name__ == "__main__":
    main()