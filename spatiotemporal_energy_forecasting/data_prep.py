from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_EXCLUDED_FILES = (
    "MainTerminal_PhaseCount_3_geq_2017-10-01_lt_2018-01-01.csv",
)

FEATURE_COLUMNS = {
    "P_kW": "P_kW",
    "S_kVA": "S_kVA",
    "std_P_kW": "std_P_kW",
    "std_S_kVA": "std_S_kVA",
}


def _parse_sensor_datetime(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.split("+", n=1).str[0]
    try:
        return pd.to_datetime(cleaned, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(cleaned)


def _machine_name(csv_path: Path) -> str:
    return csv_path.name.split("_")[0]


def aggregate_machine_file(
    csv_path: str | Path,
    *,
    start_hour: int = 6,
    end_hour: int = 19,
) -> pd.DataFrame:
    """Load one HIPE machine file and aggregate it to 10-minute features."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"SensorDateTime", "P_kW", "S_kVA"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path} is missing required columns: {missing_text}")

    df = df[["SensorDateTime", "P_kW", "S_kVA"]].copy()
    df["SensorDateTime"] = _parse_sensor_datetime(df["SensorDateTime"])
    df = df[df["SensorDateTime"].dt.weekday < 5]
    df = df[
        (df["SensorDateTime"].dt.hour >= start_hour)
        & (df["SensorDateTime"].dt.hour < end_hour)
    ]
    df["SensorDateTime"] = df["SensorDateTime"].dt.round("10min")

    aggregated = (
        df.groupby("SensorDateTime")
        .agg(
            P_kW_sum=("P_kW", "sum"),
            S_kVA_sum=("S_kVA", "sum"),
            std_P_kW=("P_kW", "std"),
            std_S_kVA=("S_kVA", "std"),
        )
        .reset_index()
    )
    aggregated["std_P_kW"] = aggregated["std_P_kW"].fillna(0)
    aggregated["std_S_kVA"] = aggregated["std_S_kVA"].fillna(0)

    prefix = _machine_name(csv_path)
    return aggregated.rename(
        columns={
            "P_kW_sum": f"{prefix}_P_kW",
            "S_kVA_sum": f"{prefix}_S_kVA",
            "std_P_kW": f"{prefix}_std_P_kW",
            "std_S_kVA": f"{prefix}_std_S_kVA",
        }
    )


def _merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No data frames were produced from the input CSV files")

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="SensorDateTime", how="outer"),
        frames,
    )
    return merged.sort_values("SensorDateTime").reset_index(drop=True)


def _clip_negative_readings(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    value_columns = [column for column in output.columns if column != "SensorDateTime"]
    output[value_columns] = output[value_columns].clip(lower=0)
    return output


def prepare_cleaned_data(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    start_hour: int = 6,
    end_hour: int = 19,
    excluded_files: Iterable[str] = DEFAULT_EXCLUDED_FILES,
) -> dict[str, Path]:
    """Create model-ready feature CSV files from raw HIPE machine CSV files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    excluded = set(excluded_files)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    csv_files = sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.name not in excluded
    )
    if not csv_files:
        raise ValueError(f"No usable CSV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in FEATURE_COLUMNS}
    for csv_file in csv_files:
        aggregated = aggregate_machine_file(
            csv_file,
            start_hour=start_hour,
            end_hour=end_hour,
        )
        prefix = _machine_name(csv_file)
        frames["P_kW"].append(aggregated[["SensorDateTime", f"{prefix}_P_kW"]])
        frames["S_kVA"].append(aggregated[["SensorDateTime", f"{prefix}_S_kVA"]])
        frames["std_P_kW"].append(
            aggregated[["SensorDateTime", f"{prefix}_std_P_kW"]]
        )
        frames["std_S_kVA"].append(
            aggregated[["SensorDateTime", f"{prefix}_std_S_kVA"]]
        )

    merged = {
        name: _clip_negative_readings(_merge_feature_frames(feature_frames))
        for name, feature_frames in frames.items()
    }

    total_power = merged["P_kW"].copy()
    value_columns = [
        column for column in total_power.columns if column != "SensorDateTime"
    ]
    total_power["sum_P_kW"] = total_power[value_columns].sum(axis=1)

    outputs = {
        "P_kW": output_dir / "P_kW.csv",
        "S_kVA": output_dir / "S_kVA.csv",
        "std_P_kW": output_dir / "std_P_kW.csv",
        "std_S_kVA": output_dir / "std_S_kVA.csv",
        "target": output_dir / "total_power.csv",
    }

    merged["P_kW"].to_csv(outputs["P_kW"], index=False)
    merged["S_kVA"].to_csv(outputs["S_kVA"], index=False)
    merged["std_P_kW"].to_csv(outputs["std_P_kW"], index=False)
    merged["std_S_kVA"].to_csv(outputs["std_S_kVA"], index=False)
    total_power[["SensorDateTime", "sum_P_kW"]].to_csv(
        outputs["target"],
        index=False,
    )

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare HIPE CSV files for spatiotemporal energy forecasting."
    )
    parser.add_argument("--input-dir", required=True, help="Raw HIPE CSV directory.")
    parser.add_argument("--output-dir", required=True, help="Cleaned CSV output directory.")
    parser.add_argument("--start-hour", type=int, default=6)
    parser.add_argument("--end-hour", type=int, default=19)
    parser.add_argument(
        "--exclude",
        action="append",
        default=list(DEFAULT_EXCLUDED_FILES),
        help="CSV filename to exclude. Can be supplied multiple times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = prepare_cleaned_data(
        args.input_dir,
        args.output_dir,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        excluded_files=args.exclude,
    )

    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

