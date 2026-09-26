"""Reproducible entry point for fine-tuning fast-plate-ocr's pretrained
`cct-s-v2-global-model` on our real Indian plate TRAIN/VAL split.

Runs under the SEPARATE OCR venv (needs `fast_plate_ocr`, `keras`,
`albumentations`, `onnxruntime` for the export step -- see
docs/decisions.md "OCR module"). This script never imports torch and is
never run from the main venv.

Two things this script does, in order:

1. **CSV conversion.** Our real-plate CSVs (`engine/perception/build_real_set.py`
   format: `image_path,canonical,text_line1,text_line2,layout,style,condition`)
   are converted to fast-plate-ocr's own training format
   (`image_path,plate_text` -- see `fast_plate_ocr.train.data.annotations`).
   `plate_text` is `text_line1 + text_line2` (the RAW printed characters, no
   canonical blank-padding: `fast_plate_ocr.train.utilities.utils.target_transform`
   right-pads with the model's OWN pad character itself). The converted CSV
   is written into the SAME DIRECTORY as the source CSV (as
   `<source>_fpo.csv`), never a separate scratch directory: fast-plate-ocr's
   `PlateRecognitionPyDataset` resolves `image_path` as
   `dirname(realpath(annotations_file)) + os.sep + image_path` (plain string
   concatenation, not `os.path.join`), so the converted CSV must sit next to
   the same `images/` folder the original CSV does for the existing
   relative `image_path` values (e.g. `images/train_000000.jpg`) to still
   resolve.

2. **Fine-tuning.** Shells out to the installed `fast-plate-ocr train` CLI
   (not re-implemented here -- see docs/decisions.md "no usable Indian
   plate-text dataset..." entries on why we reuse third-party tooling
   rather than rebuild it) with `--weights-path` pointing at the pretrained
   `cct_s_v2_global.keras` (the TRAINABLE weights published in the same
   GitHub release the hub's ONNX download already uses -- same model per
   the user's approval), `--model-config-file`/`--plate-config-file`
   pointing at that release's own configs, TRAIN on real TRAIN,
   VALIDATE on real VAL, batch size modest by default (16 -- our real
   TRAIN split is only 929 images), and the library's own standard
   augmentation pipeline (no `--augmentation-path` override).

NEVER pass the real TEST split to this script -- TEST is exclusively for
`eval_fpo.py`. `--epochs 1` or `--epochs 2` is the "prove it trains" smoke
run; the lead runs the real training with a realistic epoch count (see this
module's docstring / the FINAL REPORT for the exact command).

After training, `export_to_onnx()` (or the printed `fast-plate-ocr export`
command) turns `<output_dir>/best.keras` into an ONNX + plate_config.yaml
pair directly evaluable by `eval_fpo.py`, exactly like the pretrained
zero-shot checkpoint.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNS_DIR = Path("C:/sutra-data/ocr/fpo/runs")
DEFAULT_PRETRAINED_DIR = Path("C:/sutra-data/ocr/fpo/pretrained/cct-s-v2-global-model")


def _fast_plate_ocr_cli() -> str:
    """Path to the `fast-plate-ocr` console script installed alongside the
    CURRENT interpreter (`sys.executable`'s own `Scripts/`/`bin/` dir),
    rather than relying on it being on `PATH` -- this script is invoked as
    `<venv-ocr>/Scripts/python.exe train_fpo.py ...`, which does not itself
    activate the venv or extend PATH."""
    scripts_dir = Path(sys.executable).resolve().parent
    exe_name = "fast-plate-ocr.exe" if sys.platform == "win32" else "fast-plate-ocr"
    candidate = scripts_dir / exe_name
    return str(candidate) if candidate.is_file() else "fast-plate-ocr"


def convert_csv_to_fpo_format(src_csv: Path, max_plate_slots: int = 10) -> Path:
    """Convert one `build_real_set.py`-format CSV to fast-plate-ocr's
    `image_path,plate_text` format, written as `<src_csv.stem>_fpo.csv` in
    the SAME directory as `src_csv` (see module docstring for why the
    directory must match). Returns the path to the converted CSV.

    Rows whose `text_line1 + text_line2` is empty (a label that was excluded
    upstream, or a genuinely unreadable crop) are skipped -- an empty
    `plate_text` would train the model to predict "all pad", which is not a
    real plate-reading target.
    """
    dst_csv = src_csv.with_name(f"{src_csv.stem}_fpo.csv")
    with src_csv.open("r", encoding="utf-8", newline="") as f_in:
        rows = list(csv.DictReader(f_in))

    out_rows = []
    for row in rows:
        plate_text = (row.get("text_line1", "") + row.get("text_line2", "")).strip().upper()
        if not plate_text:
            continue
        if len(plate_text) > max_plate_slots:
            raise ValueError(
                f"{src_csv}: plate_text {plate_text!r} (from {row['image_path']}) exceeds "
                f"max_plate_slots={max_plate_slots}"
            )
        out_rows.append({"image_path": row["image_path"], "plate_text": plate_text})

    if not out_rows:
        raise ValueError(f"{src_csv}: no rows with non-empty plate_text after conversion")

    with dst_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        writer.writerows(out_rows)

    return dst_csv


def build_train_command(
    model_config: Path,
    plate_config: Path,
    train_csv: Path,
    val_csv: Path,
    weights_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int | None,
) -> list[str]:
    cmd = [
        _fast_plate_ocr_cli(),
        "train",
        "--model-config-file",
        str(model_config),
        "--plate-config-file",
        str(plate_config),
        "--annotations",
        str(train_csv),
        "--val-annotations",
        str(val_csv),
        "--weights-path",
        str(weights_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def build_export_command(
    keras_model: Path, plate_config: Path, save_dir: Path
) -> list[str]:
    return [
        _fast_plate_ocr_cli(),
        "export",
        "--model",
        str(keras_model),
        "--format",
        "onnx",
        "--plate-config-file",
        str(plate_config),
        "--save-dir",
        str(save_dir),
    ]


def export_to_onnx_direct(keras_model_path: Path, plate_config_path: Path, out_file: Path) -> Path:
    """Export a fine-tuned `.keras` checkpoint to ONNX, reusing the
    library's own `fast_plate_ocr.cli.export.export_onnx` logic -- EXCEPT
    for the one step that doesn't work on Windows: that function writes the
    freshly-exported graph into a `tempfile.NamedTemporaryFile()` and then
    has `keras`/`tf2onnx` re-open that SAME path by name while the original
    handle is still held open inside the `with` block. POSIX allows
    reopening a file that's still open elsewhere; Windows does not
    (`PermissionError: [Errno 13] Permission denied`), so
    `fast-plate-ocr export --format onnx` fails on Windows regardless of
    input (confirmed by reproducing on the 2-epoch fine-tune's own
    checkpoint -- training itself succeeded, only this export step failed).

    This function performs the exact same sequence -- `model.export(...,
    format="onnx")`, then optional `onnxslim` simplification, then an
    onnxruntime validation pass comparing Keras vs ONNX outputs -- but
    writes directly to a real, plainly-opened-and-closed file at each step
    instead of a `NamedTemporaryFile`, which sidesteps the platform bug
    without touching the installed library.
    """
    import numpy as np
    import onnx
    import onnxruntime as rt
    import onnxslim
    from fast_plate_ocr.cli.export import _prepare_model_for_onnx_export
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    plate_config = load_plate_config_from_yaml(plate_config_path)
    model = load_keras_model(keras_model_path, plate_config)

    export_model, spec_shape, dummy_input = _prepare_model_for_onnx_export(
        model,
        plate_config,
        dynamic_batch=True,
        input_dtype="uint8",
        data_format="channels_last",
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_onnx = out_file.with_name(out_file.stem + "_raw.onnx")
    if raw_onnx.exists():
        raw_onnx.unlink()

    import keras

    spec = [keras.InputSpec(name="input", shape=spec_shape, dtype="uint8")]
    export_model.export(str(raw_onnx), format="onnx", verbose=False, input_signature=spec)

    if out_file.exists():
        out_file.unlink()
    model_simp = onnxslim.slim(onnx.load(str(raw_onnx)))
    onnx.save(model_simp, str(out_file))
    raw_onnx.unlink()

    # Validate: exported ONNX output must match the Keras model on the same
    # random input (same tolerance the library's own export_onnx uses).
    sess = rt.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [o.name for o in sess.get_outputs()]
    keras_keys = list(export_model.output.keys())
    keras_out = export_model.predict(dummy_input, verbose=0)
    if isinstance(keras_out, (list, tuple)):
        keras_out = dict(zip(keras_keys, keras_out, strict=False))
    onnx_raw = sess.run(onnx_output_names, {input_name: dummy_input})
    onnx_out = dict(zip(onnx_output_names, onnx_raw, strict=False))
    for name in keras_keys:
        if not np.allclose(keras_out[name], onnx_out[name], rtol=1e-4, atol=1e-4):
            print(f"WARNING: ONNX output {name!r} deviates from Keras beyond tolerance", flush=True)
        else:
            print(f"ONNX output {name!r} matches Keras", flush=True)

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune fast-plate-ocr's pretrained cct-s-v2-global-model on the real "
            "Indian plate TRAIN/VAL split. NEVER pass the real TEST split here."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("C:/sutra-data/ocr/real/train/labels.csv"),
        help="Real TRAIN labels.csv (build_real_set.py format).",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("C:/sutra-data/ocr/real/val/labels.csv"),
        help="Real VAL labels.csv (build_real_set.py format). NEVER the test split.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_PRETRAINED_DIR / "cct_s_v2_global_model_config.yaml",
    )
    parser.add_argument(
        "--plate-config",
        type=Path,
        default=DEFAULT_PRETRAINED_DIR / "cct_s_v2_global_plate_config.yaml",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=DEFAULT_PRETRAINED_DIR / "cct_s_v2_global.keras",
        help="Trainable Keras weights (github.com/ankandrew/fast-plate-ocr releases/arg-plates).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="After training, export <output_dir>/<run>/best.keras to ONNX via "
        "'fast-plate-ocr export' (prints the exact command either way).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, don't run them.")
    args = parser.parse_args()

    for path in (args.model_config, args.plate_config, args.pretrained_weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"converting {args.train_csv} ...", flush=True)
    train_fpo_csv = convert_csv_to_fpo_format(args.train_csv)
    print(f"  -> {train_fpo_csv}", flush=True)
    print(f"converting {args.val_csv} ...", flush=True)
    val_fpo_csv = convert_csv_to_fpo_format(args.val_csv)
    print(f"  -> {val_fpo_csv}", flush=True)

    train_cmd = build_train_command(
        model_config=args.model_config,
        plate_config=args.plate_config,
        train_csv=train_fpo_csv,
        val_csv=val_fpo_csv,
        weights_path=args.pretrained_weights,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("training command:", " ".join(train_cmd), flush=True)

    if args.dry_run:
        return

    subprocess.run(train_cmd, check=True)

    if args.export_onnx:
        # The library timestamps each run into its own subdirectory
        # (output_dir/YYYY-MM-DD_HH-MM-SS/best.keras) -- pick the most
        # recently modified one rather than guessing the timestamp.
        run_dirs = sorted(
            (p for p in args.output_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if not run_dirs:
            raise RuntimeError(f"no run directory found under {args.output_dir} after training")
        latest_run = run_dirs[-1]
        best_keras = latest_run / "best.keras"
        if not best_keras.is_file():
            raise FileNotFoundError(best_keras)

        # NOTE: NOT `fast-plate-ocr export` (build_export_command) -- that
        # CLI command's ONNX export path is broken on Windows (see
        # export_to_onnx_direct's docstring: it reopens a still-open
        # NamedTemporaryFile by name, which Windows refuses). Confirmed by
        # reproducing the failure on this exact checkpoint before switching
        # to the direct in-process export below, which performs the same
        # steps without the buggy temp-file pattern.
        out_file = latest_run / f"{best_keras.stem}.onnx"
        export_to_onnx_direct(best_keras, args.plate_config, out_file)
        print(f"exported ONNX to {out_file}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
