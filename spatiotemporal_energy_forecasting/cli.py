from __future__ import annotations

import argparse
import os

from .config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the spatiotemporal energy forecasting model."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIPE_CLEANED_DIR"),
        help="Directory containing P_kW.csv, S_kVA.csv, std_P_kW.csv, std_S_kVA.csv, and total_power.csv.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--readout", choices=["mean", "meanmax", "sum"], default="meanmax")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("--data-dir is required unless HIPE_CLEANED_DIR is set")

    try:
        from .pipeline import run_training
    except ImportError as exc:
        parser.error(
            "Training dependencies are not installed. "
            "Install them with `pip install -r requirements.txt`. "
            f"Original error: {exc}"
        )

    config = TrainingConfig(
        data_dir=args.data_dir,
        horizon=args.horizon,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stop=args.early_stop,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        readout=args.readout,
        seed=args.seed,
    )
    results = run_training(config, plot=args.plot)
    print(
        "Done. "
        f"MSE: {results['mse']:.4f}, "
        f"RMSE: {results['rmse']:.4f}, "
        f"MAE: {results['mae']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
