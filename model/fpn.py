import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as F


DATA_ROOT = Path("dataset2_balanced")
OUTPUT_DIR = Path("resnet18_segmentation_run")
PATCH_SIZE = 256
TRAIN_PATCHES_PER_IMAGE = 48
VAL_STRIDE = 256
BATCH_SIZE = 64
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
STAGE1_LEARNING_RATE = 1e-2
STAGE2_LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_SIZE = 4
GAMMA = 0.5
NUM_WORKERS = 32
SEED = 42
THRESHOLD = 0.5
THRESHOLD_CANDIDATES = [round(value, 2) for value in np.arange(0.05, 0.51, 0.01)]

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def collect_samples(split):
    base_dir = DATA_ROOT / "tiff"
    image_dir = base_dir / split
    mask_dir = base_dir / f"{split}_labels"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Could not find split folders inside {base_dir.resolve()}.")

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        matches = [path for path in mask_dir.glob(f"{image_path.stem}.*") if path.suffix.lower() in VALID_EXTENSIONS]
        mask_path = matches[0] if matches else None
        if mask_path is not None:
            samples.append((image_path, mask_path))

    if not samples:
        raise RuntimeError(f"No matched image/mask pairs found in {image_dir} and {mask_dir}.")

    return samples


def load_pair(sample):
    image_path, mask_path = sample
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
    return image, mask


def pad_pair(image, mask, patch_size):
    width, height = image.size
    padding = (0, 0, max(0, patch_size - width), max(0, patch_size - height))
    if padding[2] or padding[3]:
        image = F.pad(image, padding, fill=0)
        mask = F.pad(mask, padding, fill=0)
    return image, mask


def image_to_tensor(image):
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def mask_to_tensor(mask):
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy((array > 127).astype(np.float32)).unsqueeze(0)


class RoadDataset(Dataset):
    def __init__(self, samples, patch_size, train=True, patches_per_image=1, stride=None):
        self.samples = samples
        self.patch_size = patch_size
        self.train = train
        self.patches_per_image = patches_per_image
        self.index = []

        if not train:
            for sample in samples:
                with Image.open(sample[0]) as image:
                    width, height = image.size
                lefts = self.make_positions(max(width, patch_size), patch_size, stride)
                tops = self.make_positions(max(height, patch_size), patch_size, stride)
                self.index.extend((sample, left, top) for top in tops for left in lefts)

    @staticmethod
    def make_positions(size, patch_size, stride):
        if size <= patch_size:
            return [0]
        positions = list(range(0, size - patch_size + 1, stride))
        if positions[-1] != size - patch_size:
            positions.append(size - patch_size)
        return positions

    def __len__(self):
        return len(self.samples) * self.patches_per_image if self.train else len(self.index)

    def __getitem__(self, index):
        if self.train:
            sample = self.samples[index % len(self.samples)]
            image, mask = pad_pair(*load_pair(sample), self.patch_size)
            width, height = image.size
            left = random.randint(0, width - self.patch_size)
            top = random.randint(0, height - self.patch_size)
        else:
            sample, left, top = self.index[index]
            image, mask = pad_pair(*load_pair(sample), self.patch_size)

        image = F.crop(image, top, left, self.patch_size, self.patch_size)
        mask = F.crop(mask, top, left, self.patch_size, self.patch_size)
        if self.train and random.random() < 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        return image_to_tensor(image), mask_to_tensor(mask)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet18FPN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4
        self.fpn_dim = 128

        self.lateral4 = nn.Conv2d(512, self.fpn_dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, self.fpn_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, self.fpn_dim, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, self.fpn_dim, kernel_size=1)
        self.lateral0 = nn.Conv2d(64, self.fpn_dim, kernel_size=1)

        self.up4_to3 = self.make_upsample_block(self.fpn_dim)
        self.up3_to2 = self.make_upsample_block(self.fpn_dim)
        self.up2_to1 = self.make_upsample_block(self.fpn_dim)
        self.up1_to0 = self.make_upsample_block(self.fpn_dim)

        self.smooth4 = ConvBlock(self.fpn_dim, self.fpn_dim)
        self.smooth3 = ConvBlock(self.fpn_dim, self.fpn_dim)
        self.smooth2 = ConvBlock(self.fpn_dim, self.fpn_dim)
        self.smooth1 = ConvBlock(self.fpn_dim, self.fpn_dim)
        self.smooth0 = ConvBlock(self.fpn_dim, self.fpn_dim)

        self.head = nn.Sequential(
            ConvBlock(self.fpn_dim * 5, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    @staticmethod
    def make_upsample_block(channels):
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def upsample_add(cls, upsample, x, skip):
        x = upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return x + skip

    def forward(self, x):
        skip0 = self.stem(x)
        skip1 = self.encoder1(self.pool(skip0))
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        p4 = self.smooth4(self.lateral4(skip4))
        p3 = self.smooth3(self.upsample_add(self.up4_to3, p4, self.lateral3(skip3)))
        p2 = self.smooth2(self.upsample_add(self.up3_to2, p3, self.lateral2(skip2)))
        p1 = self.smooth1(self.upsample_add(self.up2_to1, p2, self.lateral1(skip1)))
        p0 = self.smooth0(self.upsample_add(self.up1_to0, p1, self.lateral0(skip0)))

        pyramid = [p0]
        for feature in [p1, p2, p3, p4]:
            pyramid.append(
                nn.functional.interpolate(
                    feature,
                    size=p0.shape[-2:],
                    mode="nearest"
                )
            )

        fused = torch.cat(pyramid, dim=1)
        logits = self.head(fused)
        return nn.functional.interpolate(logits, size=x.shape[-2:], mode="nearest")

    def freeze_early_encoder(self):
        for block in [self.stem, self.encoder1, self.encoder2]:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def build_metrics(tp, fp, fn, loss):
    eps = 1e-7
    return {
        "loss": loss,
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
    }


def collect_confusion(logits, masks, threshold):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * masks).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    return tp, fp, fn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, threshold=THRESHOLD):
    model.eval()
    total_loss = tp = fp = fn = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            batch_tp, batch_fp, batch_fn = collect_confusion(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    return build_metrics(tp, fp, fn, total_loss / len(loader.dataset))


def find_best_threshold(model, loader, device):
    best_threshold = THRESHOLD
    best_dice = -1.0
    model.eval()

    with torch.no_grad():
        cached_logits = []
        cached_masks = []
        for images, masks in loader:
            images = images.to(device)
            logits = model(images).cpu()
            cached_logits.append(logits)
            cached_masks.append(masks)

    for threshold in THRESHOLD_CANDIDATES:
        tp = fp = fn = 0.0
        for logits, masks in zip(cached_logits, cached_masks):
            batch_tp, batch_fp, batch_fn = collect_confusion(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
        dice = build_metrics(tp, fp, fn, 0.0)["dice"]
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    return best_threshold, best_dice


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    for key in ("train_loss", "val_loss", "val_dice"):
        plt.plot(epochs, history[key], label=key)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics.png", dpi=150)
    plt.close()


def save_predictions(model, samples, device, threshold, limit=4):
    model.eval()
    mean = MEAN.to(device)
    std = STD.to(device)
    with torch.no_grad():
        for index, sample in enumerate(samples[:limit], start=1):
            image, mask = pad_pair(*load_pair(sample), PATCH_SIZE)
            image = F.center_crop(image, [PATCH_SIZE, PATCH_SIZE])
            mask = F.center_crop(mask, [PATCH_SIZE, PATCH_SIZE])
            image_tensor = image_to_tensor(image).unsqueeze(0).to(device)
            pred = (torch.sigmoid(model(image_tensor)).cpu() >= threshold).float()[0, 0]

            _, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(torch.clamp(image_tensor[0].cpu() * std.cpu() + mean.cpu(), 0, 1).permute(1, 2, 0).numpy())
            axes[1].imshow(mask_to_tensor(mask)[0].numpy(), cmap="gray")
            axes[2].imshow(pred.numpy(), cmap="gray")
            for axis, title in zip(axes, ("image", "mask", "prediction")):
                axis.set_title(title)
                axis.axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"prediction_{index}.png", dpi=150)
            plt.close()


def make_optimizer(model, learning_rate):
    return SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    optimizer,
    scheduler,
    start_epoch,
    num_epochs,
    history,
    best_val_loss,
    best_model,
):
    for local_epoch in range(num_epochs):
        epoch = start_epoch + local_epoch
        total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=THRESHOLD)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"epoch {epoch}/{total_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice@{THRESHOLD:.2f}={val_metrics['dice']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model = copy.deepcopy(model.state_dict())

    return best_val_loss, best_model


def main():
    set_seed()
    make_output_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples = collect_samples("train")
    val_samples = collect_samples("val")
    threshold_val_samples = collect_samples("threshold_val")
    train_loader = DataLoader(
        RoadDataset(train_samples, PATCH_SIZE, train=True, patches_per_image=TRAIN_PATCHES_PER_IMAGE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        RoadDataset(val_samples, PATCH_SIZE, train=False, stride=VAL_STRIDE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    threshold_val_loader = DataLoader(
        RoadDataset(threshold_val_samples, PATCH_SIZE, train=False, stride=VAL_STRIDE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = ResNet18FPN().to(device)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model = None
    best_threshold = THRESHOLD
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    print(f"train_images: {len(train_samples)}")
    print(f"val_images: {len(val_samples)}")
    print(f"threshold_val_images: {len(threshold_val_samples)}")
    print(f"train_patches: {len(train_loader.dataset)}")
    print(f"val_patches: {len(val_loader.dataset)}")
    print(f"threshold_val_patches: {len(threshold_val_loader.dataset)}")
    print(f"device: {device}")
    print("loss: BCEWithLogitsLoss")
    print(f"stage1: freeze stem, encoder1, encoder2 | lr={STAGE1_LEARNING_RATE}")
    print(f"stage2: unfreeze all | lr={STAGE2_LEARNING_RATE}")

    model.freeze_early_encoder()
    optimizer = make_optimizer(model, STAGE1_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    best_val_loss, best_model = run_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        optimizer,
        scheduler,
        start_epoch=1,
        num_epochs=STAGE1_EPOCHS,
        history=history,
        best_val_loss=best_val_loss,
        best_model=best_model,
    )

    model.unfreeze_all()
    optimizer = make_optimizer(model, STAGE2_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    best_val_loss, best_model = run_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        optimizer,
        scheduler,
        start_epoch=STAGE1_EPOCHS + 1,
        num_epochs=STAGE2_EPOCHS,
        history=history,
        best_val_loss=best_val_loss,
        best_model=best_model,
    )

    model.load_state_dict(best_model)
    best_threshold, threshold_val_dice = find_best_threshold(model, threshold_val_loader, device)
    val_metrics = evaluate(model, val_loader, criterion, device, threshold=best_threshold)
    print(
        f"best_val_loss={best_val_loss:.4f} | "
        f"best_threshold={best_threshold:.2f} | "
        f"val_dice_at_best_threshold={val_metrics['dice']:.4f} | "
        f"threshold_val_dice={threshold_val_dice:.4f}"
    )

    torch.save(
        {
            "model_state_dict": best_model,
            "best_val_loss": best_val_loss,
            "patch_size": PATCH_SIZE,
            "threshold": best_threshold,
        },
        OUTPUT_DIR / "best_model.pth",
    )
    plot_history(history)
    save_predictions(model, val_samples, device, best_threshold)
    print("done")


if __name__ == "__main__":
    main()
