import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Treina YOLO11s com fine-tuning")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="Peso pre-treinado")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"), help="Arquivo data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="Ex.: 0, cpu, 0,1")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo11s"))
    parser.add_argument("--name", type=str, default="finetune")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Arquivo de dataset nao encontrado: {args.data}")

    project_dir = args.project.resolve()

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        pretrained=True,
    )

    print(f"Treino finalizado. Artefatos em: {results.save_dir}")


if __name__ == "__main__":
    main()