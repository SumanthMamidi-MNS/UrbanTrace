"""Train the CRNN OCR recogniser (engine.perception.crnn) on plate crops.

Runs under the SEPARATE OCR venv (torch + CUDA). Reads a labels CSV in
exactly the schema engine.perception.synth_plates.py writes (image_path,
canonical, text_line1, text_line2, layout, style, condition) -- a real-data
CSV for fine-tuning must use the same columns (condition may be a free-form
tag or omitted).

Fine-tuning on real data (once the user's real crops + a labelled CSV exist):

    python -m engine.perception.train_ocr \\
        --train-csv path/to/real_train_labels.csv \\
        --val-csv   path/to/real_val_labels.csv \\
        --init-checkpoint <data_dir>/ocr/runs/synth/best.pt \\
        --run-dir <data_dir>/ocr/runs/real_finetune \\
        --epochs 15 --lr 1e-4

(`<data_dir>` is `URBANTRACE_DATA_DIR`, see `engine.paths`.)

`--init-checkpoint` warm-starts the model weights only (fresh optimizer/
epoch count) -- the fine-tuning entry point. `--resume` instead restores
optimizer + epoch state from `<run-dir>/last.pt`, for continuing an
interrupted run of the SAME dataset. `--extra-csv` appends a second labelled
set (e.g. a small real set) to the training data of an otherwise-synthetic
run, for joint training rather than sequential fine-tuning.

Two-line plates are split into independent top/bottom single-line crops at
dataset-loading time (engine.perception.crnn.split_two_line_crop) -- the
model itself only ever sees one-line crops; see crnn.py's module docstring.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from engine.paths import get_data_dir
from engine.perception.crnn import (
    CRNN,
    IMG_HEIGHT,
    IMG_WIDTH,
    build_ctc_loss,
    ctc_targets_from_labels,
    fit_temperature,
    split_two_line_crop,
)
from engine.perception.ctc_to_slots import char_to_label, ctc_greedy_decode


class PlateOCRDataset(Dataset):
    """Flattens a labels CSV into single-line (image, label) training
    examples: a one-line row yields one example (the full plate text); a
    two-line row yields two (top-half -> text_line1, bottom-half ->
    text_line2). The CRNN never sees a two-line image directly."""

    def __init__(self, csv_path: Path, root: Path | None = None):
        self.root = root or csv_path.parent
        self.items: list[tuple[Path, str, str | None]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img_path = self.root / row["image_path"]
                if row["layout"] == "one_line":
                    self.items.append((img_path, row["text_line1"], None))
                else:
                    self.items.append((img_path, row["text_line1"], "top"))
                    self.items.append((img_path, row["text_line2"], "bottom"))
        if not self.items:
            raise ValueError(f"no rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path, label, half = self.items[idx]
        img = Image.open(path).convert("L")
        if half is not None:
            arr = np.array(img)
            top, bottom = split_two_line_crop(arr)
            img = Image.fromarray(top if half == "top" else bottom)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, label


def collate_fn(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    ctc_loss_fn: torch.nn.CTCLoss,
    seq_len: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        targets, target_lengths = ctc_targets_from_labels(labels, char_to_label)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        autocast_device = "cuda" if device == "cuda" else "cpu"
        with autocast(device_type=autocast_device, enabled=(device == "cuda")):
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_crop_accuracy(model: CRNN, loader: DataLoader, device: str) -> float:
    """Per-crop (not whole-plate) exact-match accuracy -- a cheap training-
    time monitoring signal. eval_ocr.py computes the real whole-plate /
    per-condition / calibration metrics post-hoc via ctc_to_slots."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i, label in enumerate(labels):
            decoded, _ = ctc_greedy_decode(probs[i])
            correct += int(decoded == label)
            total += 1
    return correct / max(total, 1)


def save_checkpoint(
    path: Path,
    model: CRNN,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_val_acc: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CRNN,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    device: str = "cpu",
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CRNN plate OCR recogniser.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument(
        "--extra-csv",
        type=Path,
        default=None,
        help=(
            "Additional labels CSV (same schema) appended to the training "
            "set, e.g. a small real-data set trained jointly with synthetic data."
        ),
    )
    parser.add_argument("--extra-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=get_data_dir() / "ocr/runs/default")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer+epoch state from <run-dir>/last.pt.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights only (fresh optimizer/epoch) -- the fine-tuning entry.",
    )
    parser.add_argument(
        "--fit-temperature",
        action="store_true",
        help=(
            "After training (or immediately, with --epochs 0 and --init-checkpoint/--resume "
            "to just (re)calibrate an existing checkpoint), fit a softmax temperature on the "
            "BEST checkpoint's weights using the validation set (FINDING 3) and store it in "
            "<run-dir>/best.pt as 'temperature'. eval_ocr.py applies it automatically."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)

    train_ds: Dataset = PlateOCRDataset(args.train_csv)
    if args.extra_csv:
        extra_ds = PlateOCRDataset(args.extra_csv, root=args.extra_root)
        train_ds = ConcatDataset([train_ds, extra_ds])
        print(f"train set: {len(train_ds)} crops (base + extra)", flush=True)
    else:
        print(f"train set: {len(train_ds)} crops", flush=True)
    val_ds = PlateOCRDataset(args.val_csv)
    print(f"val set: {len(val_ds)} crops", flush=True)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda"))
    ctc_loss_fn = build_ctc_loss()
    seq_len = IMG_WIDTH // 4  # matches crnn.CRNN's width pooling (/4) -- see its module docstring

    start_epoch = 0
    best_val_acc = 0.0

    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model, device=device)
        print(f"warm-started weights from {args.init_checkpoint}", flush=True)

    last_ckpt = args.run_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        start_epoch, best_val_acc = load_checkpoint(
            last_ckpt, model, optimizer, scaler, device=device
        )
        print(
            f"resumed from {last_ckpt} at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, ctc_loss_fn, seq_len
        )
        val_acc = evaluate_crop_accuracy(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_crop_acc={val_acc:.4f} ({dt:.1f}s)",
            flush=True,
        )

        save_checkpoint(last_ckpt, model, optimizer, scaler, epoch + 1, best_val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.run_dir / "best.pt", model, optimizer, scaler, epoch + 1, best_val_acc
            )

    print(f"training complete. best_val_crop_acc={best_val_acc:.4f}", flush=True)

    if args.fit_temperature:
        best_path = args.run_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, model, device=device)
            print(
                f"loaded {best_path} to calibrate the BEST (not necessarily final) weights",
                flush=True,
            )
        else:
            print("no best.pt found; fitting temperature on the current model weights", flush=True)
        temperature = fit_temperature(model, val_loader, device, seq_len, char_to_label)
        print(
            f"fitted temperature T={temperature:.4f} (minimises CTC NLL on the validation set)",
            flush=True,
        )
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            ckpt["temperature"] = temperature
            torch.save(ckpt, best_path)
            print(f"stored temperature={temperature:.4f} in {best_path}", flush=True)
        else:
            print("no best.pt to store temperature in -- run with --epochs > 0 first", flush=True)


if __name__ == "__main__":
    main()
