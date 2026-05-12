import importlib.util
from pathlib import Path

import torch


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("remote_fpn", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module(Path("fpn.py"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = module.ResNet18FPN().to(device)
    checkpoint = torch.load(
        "resnet18_segmentation_run/best_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_samples = module.collect_samples("val")
    loader = torch.utils.data.DataLoader(
        module.RoadDataset(val_samples, module.PATCH_SIZE, train=False, stride=module.VAL_STRIDE),
        batch_size=module.BATCH_SIZE,
        shuffle=False,
        num_workers=module.NUM_WORKERS,
    )

    mins = []
    maxs = []
    means = []
    positive_fracs = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            mins.append(float(probs.min()))
            maxs.append(float(probs.max()))
            means.append(float(probs.mean()))
            positive_fracs.append(float((probs >= 0.1).float().mean()))

    print("prob_min:", min(mins))
    print("prob_max:", max(maxs))
    print("prob_mean:", sum(means) / len(means))
    print("frac_ge_0.1:", sum(positive_fracs) / len(positive_fracs))

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        metrics = module.evaluate(model, loader, torch.nn.BCEWithLogitsLoss(), device, threshold=threshold)
        print(f"threshold={threshold:.2f} dice={metrics['dice']:.6f} loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
