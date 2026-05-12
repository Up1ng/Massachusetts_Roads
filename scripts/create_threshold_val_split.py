import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset2/tiff")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    train_labels_dir = dataset_root / "train_labels"
    threshold_val_dir = dataset_root / "threshold_val"
    threshold_val_labels_dir = dataset_root / "threshold_val_labels"

    if threshold_val_dir.exists() or threshold_val_labels_dir.exists():
        raise ValueError("threshold_val already exists")

    image_files = sorted(train_dir.glob("*.tiff"))

    if len(image_files) < args.count:
        raise ValueError("Not enough images in train")

    selected_images = random.sample(image_files, args.count)

    threshold_val_dir.mkdir(parents=True, exist_ok=True)
    threshold_val_labels_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for image_path in selected_images:
        mask_path = train_labels_dir / (image_path.stem + ".tif")

        if not mask_path.exists():
            continue

        shutil.move(str(image_path), str(threshold_val_dir / image_path.name))
        shutil.move(str(mask_path), str(threshold_val_labels_dir / mask_path.name))
        moved_count += 1

    print("threshold_val created")
    print("moved:", moved_count)
    print("train left:", len(list(train_dir.glob("*.tiff"))))
    print("threshold_val path:", threshold_val_dir)
    print("threshold_val_labels path:", threshold_val_labels_dir)


if __name__ == "__main__":
    main()
