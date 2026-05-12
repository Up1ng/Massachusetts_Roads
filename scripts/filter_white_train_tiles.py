import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import tifffile


def get_image_info(image_path):
    image = tifffile.imread(image_path)
    image = np.asarray(image)

    if image.ndim == 2:
        gray = image.astype(np.float32)
        white_pixels = image >= 250
    else:
        image_float = image.astype(np.float32)
        gray = 0.299 * image_float[..., 0] + 0.587 * image_float[..., 1] + 0.114 * image_float[..., 2]
        white_pixels = np.all(image >= 250, axis=-1)

    info = {
        "image_name": image_path.name,
        "label_name": image_path.stem + ".tif",
        "mean_brightness": float(gray.mean()),
        "white_part": float(white_pixels.mean()),
    }
    return info


def get_reference_threshold(folder_path, multiplier):
    values = []

    for image_path in sorted(folder_path.glob("*.tiff")):
        info = get_image_info(image_path)
        values.append(info["white_part"])

    if not values:
        raise ValueError("No TIFF files found in " + str(folder_path))

    return max(values) * multiplier


def find_bad_images(train_folder, threshold):
    bad_images = []

    for image_path in sorted(train_folder.glob("*.tiff")):
        info = get_image_info(image_path)
        if info["white_part"] > threshold:
            bad_images.append(info)

    bad_images.sort(key=lambda x: x["white_part"], reverse=True)
    return bad_images


def save_report(report_path, bad_images):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label_name", "mean_brightness", "white_part"])

        for info in bad_images:
            writer.writerow(
                [
                    info["image_name"],
                    info["label_name"],
                    f"{info['mean_brightness']:.6f}",
                    f"{info['white_part']:.6f}",
                ]
            )


def move_bad_images(dataset_root, bad_images, quarantine_name):
    train_folder = dataset_root / "train"
    labels_folder = dataset_root / "train_labels"
    quarantine_train = dataset_root / quarantine_name / "train"
    quarantine_labels = dataset_root / quarantine_name / "train_labels"

    quarantine_train.mkdir(parents=True, exist_ok=True)
    quarantine_labels.mkdir(parents=True, exist_ok=True)

    for info in bad_images:
        image_src = train_folder / info["image_name"]
        label_src = labels_folder / info["label_name"]
        image_dst = quarantine_train / info["image_name"]
        label_dst = quarantine_labels / info["label_name"]

        if image_src.exists():
            shutil.move(str(image_src), str(image_dst))

        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))


def delete_bad_images(dataset_root, bad_images):
    train_folder = dataset_root / "train"
    labels_folder = dataset_root / "train_labels"

    for info in bad_images:
        image_path = train_folder / info["image_name"]
        label_path = labels_folder / info["label_name"]

        if image_path.exists():
            image_path.unlink()

        if label_path.exists():
            label_path.unlink()


def print_examples(bad_images):
    print("First suspicious files:")

    for info in bad_images[:20]:
        print(
            info["image_name"],
            "mean=" + f"{info['mean_brightness']:.3f}",
            "white_part=" + f"{info['white_part']:.5f}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/tiff")
    parser.add_argument("--white-threshold", type=float, default=None)
    parser.add_argument("--ref-multiplier", type=float, default=3.0)
    parser.add_argument("--quarantine-name", default="train_quarantine_white")
    parser.add_argument("--report-name", default="white_tiles_report.csv")
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    if args.move and args.delete:
        raise ValueError("Choose only one action: --move or --delete")

    dataset_root = Path(args.dataset_root)
    train_folder = dataset_root / "train"
    val_folder = dataset_root / "val"
    test_folder = dataset_root / "test"

    if args.white_threshold is None:
        val_threshold = get_reference_threshold(val_folder, args.ref_multiplier)
        test_threshold = get_reference_threshold(test_folder, args.ref_multiplier)
        threshold = max(val_threshold, test_threshold)
    else:
        threshold = args.white_threshold

    bad_images = find_bad_images(train_folder, threshold)

    quarantine_folder = dataset_root / args.quarantine_name
    report_path = quarantine_folder / args.report_name
    save_report(report_path, bad_images)

    print("White threshold:", round(threshold, 6))
    print("Found suspicious images:", len(bad_images))
    print("Report saved to:", report_path)
    print_examples(bad_images)

    if args.move:
        move_bad_images(dataset_root, bad_images, args.quarantine_name)
        print("Images and masks were moved to quarantine.")

    if args.delete:
        delete_bad_images(dataset_root, bad_images)
        print("Images and masks were deleted.")


if __name__ == "__main__":
    main()
