import argparse
import random
import shutil
from pathlib import Path


def collect_pairs(split_dir, labels_dir):
    pairs = []

    for image_path in sorted(split_dir.glob("*.tiff")):
        mask_path = labels_dir / (image_path.stem + ".tif")
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def copy_pairs(pairs, image_dir, label_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(mask_path, label_dir / mask_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="dataset2")
    parser.add_argument("--output-root", default="dataset2_balanced")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-threshold-val", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_tiff = source_root / "tiff"
    output_tiff = output_root / "tiff"

    if output_root.exists():
        raise ValueError("Output folder already exists")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must be 1.0")

    all_pairs = []
    all_pairs.extend(collect_pairs(source_tiff / "train", source_tiff / "train_labels"))
    all_pairs.extend(collect_pairs(source_tiff / "val", source_tiff / "val_labels"))
    all_pairs.extend(collect_pairs(source_tiff / "test", source_tiff / "test_labels"))

    if not all_pairs:
        raise ValueError("No image-mask pairs found")

    random.seed(args.seed)
    random.shuffle(all_pairs)

    total_count = len(all_pairs)
    train_count = int(total_count * args.train_ratio)
    val_count = int(total_count * args.val_ratio)
    test_count = total_count - train_count - val_count

    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:train_count + val_count]
    test_pairs = all_pairs[train_count + val_count:]

    copy_pairs(train_pairs, output_tiff / "train", output_tiff / "train_labels")
    copy_pairs(val_pairs, output_tiff / "val", output_tiff / "val_labels")
    copy_pairs(test_pairs, output_tiff / "test", output_tiff / "test_labels")

    if args.keep_threshold_val:
        threshold_val_dir = source_tiff / "threshold_val"
        threshold_val_labels_dir = source_tiff / "threshold_val_labels"
        if threshold_val_dir.exists() and threshold_val_labels_dir.exists():
            threshold_pairs = collect_pairs(threshold_val_dir, threshold_val_labels_dir)
            copy_pairs(
                threshold_pairs,
                output_tiff / "threshold_val",
                output_tiff / "threshold_val_labels",
            )

    for file_name in ["metadata.csv", "label_class_dict.csv"]:
        source_file = source_root / file_name
        if source_file.exists():
            shutil.copy2(source_file, output_root / file_name)

    print("New dataset created:", output_root)
    print("train:", len(train_pairs))
    print("val:", len(val_pairs))
    print("test:", len(test_pairs))

    if args.keep_threshold_val:
        threshold_dir = output_tiff / "threshold_val"
        if threshold_dir.exists():
            print("threshold_val:", len(list(threshold_dir.glob("*.tiff"))))


if __name__ == "__main__":
    main()
