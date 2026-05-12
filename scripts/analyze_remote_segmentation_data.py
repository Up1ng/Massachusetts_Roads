from pathlib import Path

import numpy as np
from PIL import Image


def main():
    root = Path("dataset2_balanced/tiff")
    for split in ["train", "val", "test", "threshold_val"]:
        mask_dir = root / f"{split}_labels"
        values = set()
        positive_fractions = []
        for mask_path in sorted(mask_dir.glob("*"))[:20]:
            array = np.array(Image.open(mask_path))
            values.update(np.unique(array).tolist())
            positive_fractions.append(float((array > 0).mean()))

        print(split)
        print("  unique_values_sample:", sorted(values)[:20])
        print("  unique_count_sample:", len(values))
        print("  mean_positive_fraction_sample:", sum(positive_fractions) / len(positive_fractions))


if __name__ == "__main__":
    main()
