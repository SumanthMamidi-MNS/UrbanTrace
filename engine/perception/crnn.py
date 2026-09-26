"""Compact CRNN (CNN + BiLSTM + CTC) for Indian plate OCR.

Requires torch -- this module is only ever imported from the SEPARATE OCR
venv, never from the main project venv or from
engine/perception/ctc_to_slots.py (which stays numpy-only so it can be
tested without torch; see that module's docstring).

Architecture (input 1x32x128 grayscale -> (T=32, NUM_CLASSES) logits):

    Conv(1,32,3) + ReLU -> MaxPool(2,2)            32x128 -> 16x64
    Conv(32,64,3) + ReLU -> MaxPool(2,2)            16x64  -> 8x32
    Conv(64,128,3) + BN + ReLU
    Conv(128,128,3) + ReLU -> MaxPool((2,1))        8x32   -> 4x32   (height only)
    Conv(128,256,3) + BN + ReLU -> MaxPool((2,1))   4x32   -> 2x32   (height only)
    Conv(256,256,(2,1)) + BN + ReLU                 2x32   -> 1x32
    -> squeeze height -> sequence of T=32 steps, 256 features each
    BiLSTM(256 -> 128, 2 layers, bidirectional) -> 256 features/step
    Linear(256 -> NUM_CLASSES)

Height is pooled to 1 (a standard CRNN move: after the height dimension
collapses, what remains is a genuine 1-D sequence along width, which is what
CTC needs). Width is only pooled twice (/4, not /32), so 128px of width
survives as 32 timesteps -- comfortably more than twice the longest plate
string (max 11 real characters: 2+2+2+4 plus at least one CTC-blank gap
between any two equal adjacent characters), which is what CTC alignment
needs to be feasible at all.

Two-line plates (see engine/perception/ctc_to_slots.py's decode_two_line_plate
for how the two decoded streams are combined into slots): a two-line plate
crop is split into its top and bottom halves at the vertical midpoint
BEFORE reaching this model -- `split_two_line_crop` below -- and each half is
run through this SAME model as an independent one-line image. Real Indian
two-line HSRP plates are laid out as two roughly equal-height rows, so a
fixed 50/50 split (no learned or heuristic text-row detection) is an
intentional simplification: it can clip a couple of pixels off characters
that straddle the midline on a badly-cropped detection, which is accepted as
a known limitation rather than building a second row-segmentation model for
this scope.
"""

import numpy as np
import torch
from torch import nn

from engine.perception.ctc_to_slots import CTC_BLANK_IDX, NUM_CLASSES

IMG_HEIGHT = 32
IMG_WIDTH = 128


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32 (height only)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32 (height only)
            nn.Conv2d(256, 256, (2, 1)),  # 2x32 -> 1x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, 128) -> logits (B, T=32, num_classes)."""
        features = self.cnn(x)  # (B, 256, 1, 32)
        b, c, h, w = features.shape
        if h != 1:
            raise RuntimeError(f"expected CNN to collapse height to 1, got {h}")
        seq = features.squeeze(2).permute(0, 2, 1)  # (B, W=32, C=256)
        seq, _ = self.lstm(seq)  # (B, 32, lstm_hidden*2)
        logits = self.fc(seq)  # (B, 32, num_classes)
        return logits


@torch.no_grad()
def infer_softmax(
    model: CRNN,
    images: torch.Tensor,
    device: str | torch.device = "cpu",
    temperature: float = 1.0,
) -> np.ndarray:
    """images: (B, 1, 32, 128) float tensor, already normalised to [0, 1].
    Returns (B, T, num_classes) softmax probabilities as a numpy array --
    the exact input shape engine.perception.ctc_to_slots.ctc_greedy_decode
    (and friends) expect per-sample.

    `temperature` implements FINDING 3's calibration fix: divide the logits
    by a fitted scalar T > 1 before softmax to de-peak an overconfident
    model (`fit_temperature` below fits T; callers read it back from
    `checkpoint["temperature"]`, defaulting to 1.0 -- i.e. uncalibrated --
    for older checkpoints that predate this feature). Dividing logits by a
    POSITIVE scalar before softmax preserves the argmax exactly, so this
    never changes which character is decoded, only how peaked its posterior
    is -- decode accuracy is temperature-invariant by construction.
    """
    model.eval()
    logits = model(images.to(device))
    probs = torch.softmax(logits / temperature, dim=-1)
    return probs.cpu().numpy()


def split_two_line_crop(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a two-line plate crop into (top_half, bottom_half) at the
    vertical midpoint. `image` is (H, W) or (H, W, C). See module docstring
    for why a fixed 50/50 split is used rather than detecting the text rows.
    """
    h = image.shape[0]
    mid = h // 2
    return image[:mid], image[mid:]


def ctc_targets_from_labels(labels: list[str], char_to_label) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch of label strings into the concatenated-targets format
    torch.nn.CTCLoss expects: one flat LongTensor of all target labels back
    to back, plus a LongTensor of each sample's length. `char_to_label` is
    engine.perception.ctc_to_slots.char_to_label, passed in as a parameter
    rather than imported at call sites to keep this a pure, testable
    function of its inputs."""
    flat: list[int] = []
    lengths: list[int] = []
    for label_str in labels:
        ids = [char_to_label(c) for c in label_str]
        flat.extend(ids)
        lengths.append(len(ids))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def build_ctc_loss() -> nn.CTCLoss:
    """CTCLoss with blank=CTC_BLANK_IDX (0) and zero_infinity=True -- a
    pathological batch item (e.g. a target longer than the model's T=32
    timesteps can align, which should not happen given max plate length but
    is cheap to guard) must not NaN out the whole batch's gradient."""
    return nn.CTCLoss(blank=CTC_BLANK_IDX, zero_infinity=True)


def fit_temperature(
    model: CRNN,
    val_loader,
    device: str,
    seq_len: int,
    char_to_label,
    max_iter: int = 50,
) -> float:
    """FINDING 3: fit a single scalar softmax temperature T on the
    VALIDATION set by minimising CTC NLL (i.e. the training loss itself,
    evaluated at logits/T) -- NOT whole-plate ECE directly. CTC NLL is
    differentiable in T, so a standard 1-parameter LBFGS optimisation
    (the textbook temperature-scaling recipe, Guo et al. 2017) applies
    directly; whole-plate ECE requires ctc_to_slots' grammar-alignment
    decode, which uses argmax and is not differentiable, so it could only be
    optimised by a non-gradient scan over candidate T values. NLL was
    chosen for the cleaner, more standard optimisation; `eval_ocr.py`
    reports the actual whole-plate ECE this choice produces, before and
    after, so the metric that matters is still checked, just not directly
    optimised.

    Logits are parameterised as T = exp(log_T) so T > 0 always, and
    computed once for the whole validation set up front (it is small enough
    to fit in memory) rather than recomputed on every LBFGS step, since the
    model's weights are frozen throughout."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_target_lengths: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            targets, target_lengths = ctc_targets_from_labels(labels, char_to_label)
            all_targets.append(targets)
            all_target_lengths.append(target_lengths)

    logits_cat = torch.cat(all_logits, dim=0).to(device)
    targets_cat = torch.cat(all_targets, dim=0).to(device)
    target_lengths_cat = torch.cat(all_target_lengths, dim=0).to(device)
    input_lengths_cat = torch.full(
        (logits_cat.size(0),), seq_len, dtype=torch.long, device=device
    )
    ctc_loss_fn = build_ctc_loss()

    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        log_probs = torch.log_softmax(logits_cat / temperature, dim=-1).permute(1, 0, 2)
        loss = ctc_loss_fn(log_probs, targets_cat, input_lengths_cat, target_lengths_cat)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())
