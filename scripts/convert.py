import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

RAW_BASE = Path("cattle-detection-and-counting-in-uav-images-DatasetNinja")
OUT_BASE = Path("dataset")
CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Converte DatasetNinja para formato YOLO")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE)
    parser.add_argument("--out-base", type=Path, default=OUT_BASE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-dataset2-as-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa Dataset2 como conjunto de teste externo",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpa dataset/images e dataset/labels antes de converter",
    )
    return parser.parse_args()


def reset_output_dirs(out_base: Path, clean_output: bool):
    images_dir = out_base / "images"
    labels_dir = out_base / "labels"
    if clean_output:
        for directory in [images_dir, labels_dir]:
            if directory.exists():
                shutil.rmtree(directory)

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)


def safe_yolo_box(img_w: int, img_h: int, exterior):
    x1, y1 = exterior[0]
    x2, y2 = exterior[1]

    xmin = max(0.0, min(float(x1), float(x2)))
    xmax = min(float(img_w), max(float(x1), float(x2)))
    ymin = max(0.0, min(float(y1), float(y2)))
    ymax = min(float(img_h), max(float(y1), float(y2)))

    width_px = xmax - xmin
    height_px = ymax - ymin

    if width_px <= 0 or height_px <= 0:
        return None

    x_center = ((xmin + xmax) / 2.0) / float(img_w)
    y_center = ((ymin + ymax) / 2.0) / float(img_h)
    width = width_px / float(img_w)
    height = height_px / float(img_h)
    return x_center, y_center, width, height


def convert_annotation_file(ann_file: Path, image_path: Path, output_label: Path):
    with Image.open(image_path) as image:
        img_w, img_h = image.size

    with open(ann_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    lines = []
    invalid_boxes = 0

    for obj in data.get("objects", []):
        exterior = obj.get("points", {}).get("exterior")
        if not exterior or len(exterior) < 2:
            invalid_boxes += 1
            continue

        box = safe_yolo_box(img_w, img_h, exterior)
        if box is None:
            invalid_boxes += 1
            continue

        x_center, y_center, width, height = box
        lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_label, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return len(lines), invalid_boxes


def convert_split(ann_files, img_dir: Path, split: str, out_base: Path):
    converted = 0
    missing_images = 0
    invalid_boxes = 0

    for ann_file in ann_files:
        image_name = ann_file.name.replace(".json", "")
        image_path = img_dir / image_name
        if not image_path.exists():
            missing_images += 1
            continue

        output_image = out_base / "images" / split / image_name
        output_label = out_base / "labels" / split / f"{Path(image_name).stem}.txt"
        shutil.copy2(image_path, output_image)

        _, invalid_count = convert_annotation_file(ann_file, image_path, output_label)
        invalid_boxes += invalid_count
        converted += 1

    return {
        "split": split,
        "converted_images": converted,
        "missing_images": missing_images,
        "invalid_boxes": invalid_boxes,
    }


def split_dataset1(raw_base: Path, train_ratio: float, seed: int):
    ann_files = sorted((raw_base / "Dataset1" / "ann").glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(ann_files)

    split_idx = int(len(ann_files) * train_ratio)
    train_ann = ann_files[:split_idx]
    val_ann = ann_files[split_idx:]
    return train_ann, val_ann


def get_dataset2_annotations(raw_base: Path):
    return sorted((raw_base / "Dataset2" / "ann").glob("*.json"))


def verify_no_orphans(out_base: Path):
    checks = []
    for split in ["train", "val", "test"]:
        image_dir = out_base / "images" / split
        label_dir = out_base / "labels" / split
        image_stems = {p.stem for p in image_dir.glob("*") if p.is_file()}
        label_stems = {p.stem for p in label_dir.glob("*.txt") if p.is_file()}

        images_without_labels = sorted(image_stems - label_stems)
        labels_without_images = sorted(label_stems - image_stems)

        checks.append(
            {
                "split": split,
                "images": len(image_stems),
                "labels": len(label_stems),
                "images_without_labels": len(images_without_labels),
                "labels_without_images": len(labels_without_images),
            }
        )
    return checks


def write_yaml(out_base: Path):
    yaml_content = """path: .

train: images/train
val: images/val
test: images/test

names:
  0: cattle
"""
    with open(out_base / "data.yaml", "w", encoding="utf-8") as file:
        file.write(yaml_content)


def write_manifest(out_base: Path, config, split_stats, orphan_checks):
    manifest = {
        "config": config,
        "split_stats": split_stats,
        "consistency": orphan_checks,
    }
    with open(out_base / "split_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio deve estar entre 0 e 1")

    reset_output_dirs(args.out_base, clean_output=args.clean_output)

    train_ann, val_ann = split_dataset1(args.raw_base, args.train_ratio, args.seed)
    split_stats = []

    split_stats.append(
        convert_split(train_ann, args.raw_base / "Dataset1" / "img", "train", args.out_base)
    )
    split_stats.append(
        convert_split(val_ann, args.raw_base / "Dataset1" / "img", "val", args.out_base)
    )

    if args.use_dataset2_as_test:
        test_ann = get_dataset2_annotations(args.raw_base)
        split_stats.append(
            convert_split(test_ann, args.raw_base / "Dataset2" / "img", "test", args.out_base)
        )

    write_yaml(args.out_base)
    orphan_checks = verify_no_orphans(args.out_base)
    write_manifest(
        args.out_base,
        config={
            "raw_base": str(args.raw_base),
            "out_base": str(args.out_base),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "use_dataset2_as_test": args.use_dataset2_as_test,
            "clean_output": args.clean_output,
        },
        split_stats=split_stats,
        orphan_checks=orphan_checks,
    )

    print("Conversao finalizada")
    for stat in split_stats:
        print(
            f"[{stat['split']}] imagens={stat['converted_images']} "
            f"missing={stat['missing_images']} invalid_boxes={stat['invalid_boxes']}"
        )
    for check in orphan_checks:
        print(
            f"[consistency:{check['split']}] images={check['images']} labels={check['labels']} "
            f"images_without_labels={check['images_without_labels']} "
            f"labels_without_images={check['labels_without_images']}"
        )


if __name__ == "__main__":
    main()