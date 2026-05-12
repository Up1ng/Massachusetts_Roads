import random
import shutil
from pathlib import Path


def main():
    root = Path("dataset2/tiff")
    train_dir = root / "train"
    train_labels_dir = root / "train_labels"
    val_dir = root / "val"
    val_labels_dir = root / "val_labels"
    output_root = Path("dataset2_balanced")

    if output_root.exists():
        raise ValueError("dataset2_balanced already exists")

    if not val_dir.exists() and not val_labels_dir.exists():
        pairs = []
        for image_path in sorted(train_dir.glob("*.tiff")):
            mask_path = train_labels_dir / f"{image_path.stem}.tif"
            if mask_path.exists():
                pairs.append((image_path, mask_path))

        if not pairs:
            raise ValueError("No labeled train pairs found")

        random.seed(42)
        val_count = max(1, int(len(pairs) * 0.15))
        selected = random.sample(pairs, val_count)

        val_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in selected:
            shutil.move(str(image_path), str(val_dir / image_path.name))
            shutil.move(str(mask_path), str(val_labels_dir / mask_path.name))

        print(f"created val with {len(selected)} pairs")
    else:
        print("val already exists, skipping val creation")


if __name__ == "__main__":
    main()
