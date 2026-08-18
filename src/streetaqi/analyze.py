"""Air-quality summaries and figures for canonical streetaqi readings."""

from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from streetaqi.data import load_readings

if TYPE_CHECKING:
    from pathlib import Path

PM25_CATEGORY_UPPER_BOUNDS = (
    (9.0, "Good"),
    (35.4, "Moderate"),
    (55.4, "Unhealthy for Sensitive Groups"),
    (125.4, "Unhealthy"),
    (225.4, "Very Unhealthy"),
    (math.inf, "Hazardous"),
)

PM25_EXCEEDANCE_THRESHOLDS = (
    (9.0, "Moderate or worse"),
    (35.4, "Unhealthy for Sensitive Groups or worse"),
    (55.4, "Unhealthy or worse"),
    (125.4, "Very Unhealthy or worse"),
    (225.4, "Hazardous"),
)

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


def get_pm25_category(pm25: float) -> str:
    """Return the 2024 EPA PM2.5 AQI concentration category.

    This classification compares a concentration with AQI breakpoints. It does
    not turn a mobile instantaneous reading into a regulatory 24-hour AQI.

    Args:
        pm25: Non-negative finite PM2.5 concentration in micrograms per cubic
            meter.

    Returns:
        The category containing the concentration.

    Raises:
        ValueError: If ``pm25`` is negative or not finite.
    """
    if not math.isfinite(pm25) or pm25 < 0:
        raise ValueError("PM2.5 concentration must be finite and non-negative")
    truncated = math.floor(pm25 * 10) / 10
    return next(
        label for upper, label in PM25_CATEGORY_UPPER_BOUNDS if truncated <= upper
    )


def _require_nonempty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("At least one reading is required")


def compute_summary_stats(df: pd.DataFrame) -> dict[str, int | float]:
    """Compute observation-level summaries from validated readings.

    CO2 summaries use only rows whose persisted ``co2_qc_pass`` flag is true;
    PM2.5 summaries retain all validated PM2.5 observations.

    Args:
        df: Canonical readings.

    Returns:
        Counts and distribution summaries.

    Raises:
        ValueError: If the frame is empty or no CO2 reading passes QC.
    """
    _require_nonempty(df)
    co2 = df.loc[df["co2_qc_pass"], "co2"]
    if co2.empty:
        raise ValueError("At least one CO2 reading must pass quality control")

    return {
        "n_readings": len(df),
        "n_co2_qc_pass": len(co2),
        "n_days": int(df["day"].nunique()),
        "n_itineraries": int(df["itinerary_id"].nunique()),
        "pm25_mean": float(df["pm25"].mean()),
        "pm25_median": float(df["pm25"].median()),
        "pm25_std": float(df["pm25"].std()),
        "pm25_min": float(df["pm25"].min()),
        "pm25_max": float(df["pm25"].max()),
        "pm25_q25": float(df["pm25"].quantile(0.25)),
        "pm25_q75": float(df["pm25"].quantile(0.75)),
        "co2_mean": float(co2.mean()),
        "co2_median": float(co2.median()),
        "co2_std": float(co2.std()),
        "co2_min": float(co2.min()),
        "co2_max": float(co2.max()),
        "co2_q25": float(co2.quantile(0.25)),
        "co2_q75": float(co2.quantile(0.75)),
    }


def compute_threshold_exceedance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observation shares above each PM2.5 AQI breakpoint.

    Args:
        df: Canonical readings.

    Returns:
        A table with a clearly labeled observation-level estimand.
    """
    _require_nonempty(df)
    truncated = np.floor(df["pm25"] * 10) / 10
    rows = []
    for threshold, category in PM25_EXCEEDANCE_THRESHOLDS:
        count = int((truncated > threshold).sum())
        rows.append(
            {
                "threshold_ug_m3": threshold,
                "category_or_worse": category,
                "n_readings": count,
                "share_of_readings": count / len(df),
            }
        )
    return pd.DataFrame(rows)


def compute_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the distribution of readings across PM2.5 breakpoints.

    Args:
        df: Canonical readings.

    Returns:
        Counts and observation shares in canonical category order.
    """
    _require_nonempty(df)
    categories = df["pm25"].map(get_pm25_category)
    counts = categories.value_counts()
    return pd.DataFrame(
        {
            "category": [label for _, label in PM25_CATEGORY_UPPER_BOUNDS],
            "n_readings": [
                int(counts.get(label, 0)) for _, label in PM25_CATEGORY_UPPER_BOUNDS
            ],
            "share_of_readings": [
                float(counts.get(label, 0) / len(df))
                for _, label in PM25_CATEGORY_UPPER_BOUNDS
            ],
        }
    )


def compute_per_stop_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observation-level summaries by day and itinerary.

    Args:
        df: Canonical readings.

    Returns:
        One row per day-itinerary group.
    """
    _require_nonempty(df)
    working = df.assign(co2_valid=df["co2"].where(df["co2_qc_pass"]))
    stats = working.groupby(["day", "itinerary_id"], as_index=False, sort=True).agg(
        n_readings=("id", "size"),
        n_co2_qc_pass=("co2_valid", "count"),
        pm25_mean=("pm25", "mean"),
        pm25_median=("pm25", "median"),
        pm25_std=("pm25", "std"),
        pm25_min=("pm25", "min"),
        pm25_max=("pm25", "max"),
        co2_mean=("co2_valid", "mean"),
        co2_median=("co2_valid", "median"),
        co2_std=("co2_valid", "std"),
        co2_min=("co2_valid", "min"),
        co2_max=("co2_valid", "max"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    for threshold, category in PM25_EXCEEDANCE_THRESHOLDS:
        column = "share_" + category.lower().replace(" ", "_")
        truncated = np.floor(working["pm25"] * 10) / 10
        shares = (
            working.assign(_above=truncated > threshold)
            .groupby(["day", "itinerary_id"], as_index=False, sort=True)
            .agg(**{column: ("_above", "mean")})
        )
        stats = stats.merge(
            shares,
            on=["day", "itinerary_id"],
            how="left",
            validate="one_to_one",
        )
    return stats


def _setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
    return plt


def make_map(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Write an interactive map when the optional map dependency is present."""
    try:
        import folium
    except ImportError:
        return None

    mapped = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    if mapped.empty:
        return None

    center_latitude = float(mapped["latitude"].median())
    center_longitude = float(mapped["longitude"].median())
    map_object = folium.Map(
        location=[center_latitude, center_longitude],
        zoom_start=11,
        tiles="CartoDB Positron",
    )
    for row in mapped.to_dict(orient="records"):
        pm25 = float(row["pm25"])
        co2 = float(row["co2"])
        category = get_pm25_category(pm25)
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        popup = (
            f"<b>PM2.5:</b> {pm25:.0f} μg/m³<br>"
            f"<b>CO2:</b> {co2:.0f} ppm<br>"
            f"<b>Category:</b> {escape(category)}<br>"
            f"<b>Day:</b> {row['day']}<br>"
            f"<b>Time:</b> {escape(str(row['captured_at']))}"
        )
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=6,
            color=AQI_COLORS[category],
            weight=2,
            fill=True,
            fill_color=AQI_COLORS[category],
            fill_opacity=0.7,
            popup=popup,
        ).add_to(map_object)

    output_path = output_dir / "readings_map.html"
    map_object.save(str(output_path))
    return output_path


def make_histograms(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write PM2.5 and quality-controlled CO2 histograms."""
    plt = _setup_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    pm25_max = max(float(df["pm25"].max()), 20.0)
    axes[0].hist(
        df["pm25"],
        bins=np.arange(0, pm25_max + 20, 20),
        color="#3182bd",
        alpha=0.7,
        edgecolor="white",
    )
    for threshold, category in PM25_EXCEEDANCE_THRESHOLDS[:4]:
        if threshold < pm25_max:
            axes[0].axvline(
                threshold,
                color=AQI_COLORS[get_pm25_category(threshold)],
                linestyle="--",
                linewidth=1.5,
                label=category,
            )
    axes[0].set_xlabel("PM2.5 (μg/m³)")
    axes[0].set_ylabel("Number of readings")
    axes[0].set_title(f"PM2.5 readings (N={len(df):,})")
    axes[0].legend(loc="upper right", fontsize=7)

    co2 = df.loc[df["co2_qc_pass"], "co2"]
    co2_max = max(float(co2.max()), 100.0)
    axes[1].hist(
        co2,
        bins=np.arange(0, co2_max + 100, 100),
        color="#e6550d",
        alpha=0.7,
        edgecolor="white",
    )
    axes[1].set_xlabel("CO2 (ppm)")
    axes[1].set_ylabel("Number of readings")
    axes[1].set_title(f"QC-passing CO2 readings (N={len(co2):,})")

    figure.tight_layout()
    output_path = output_dir / "reading_histograms.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def make_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a PM2.5-versus-CO2 scatter plot using QC-passing pairs."""
    plt = _setup_matplotlib()
    paired = df[df["co2_qc_pass"]]
    figure, axis = plt.subplots(figsize=(8, 6))
    colors = [AQI_COLORS[get_pm25_category(value)] for value in paired["pm25"]]
    axis.scatter(
        paired["co2"],
        paired["pm25"],
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="white",
        linewidth=0.5,
    )
    if len(paired) >= 2 and paired["co2"].nunique() >= 2:
        coefficients = np.polyfit(paired["co2"], paired["pm25"], 1)
        line = np.poly1d(coefficients)
        x_values = np.linspace(paired["co2"].min(), paired["co2"].max(), 100)
        correlation = paired["pm25"].corr(paired["co2"])
        axis.plot(
            x_values,
            line(x_values),
            "k--",
            linewidth=2,
            alpha=0.7,
            label=f"Linear trend (r={correlation:.2f})",
        )
        axis.legend(loc="upper right")
    axis.set_xlabel("CO2 (ppm)")
    axis.set_ylabel("PM2.5 (μg/m³)")
    axis.set_title(f"QC-passing PM2.5 and CO2 pairs (N={len(paired):,})")
    axis.grid(True, alpha=0.3)

    output_path = output_dir / "pm25_co2_scatter.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def process(readings_path: Path, output_dir: Path) -> dict[str, Path]:
    """Run the analysis pipeline and write generated artifacts.

    Args:
        readings_path: Canonical Parquet file or import-boundary CSV.
        output_dir: Directory for generated tables and figures.

    Returns:
        Mapping from artifact name to path.
    """
    frame = load_readings(readings_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "summary.parquet"
    threshold_path = output_dir / "threshold_exceedance.parquet"
    category_path = output_dir / "category_distribution.parquet"
    stop_path = output_dir / "per_stop.parquet"

    pd.DataFrame([compute_summary_stats(frame)]).to_parquet(summary_path, index=False)
    compute_threshold_exceedance(frame).to_parquet(threshold_path, index=False)
    compute_category_distribution(frame).to_parquet(category_path, index=False)
    compute_per_stop_stats(frame).to_parquet(stop_path, index=False)

    artifacts = {
        "summary": summary_path,
        "threshold_exceedance": threshold_path,
        "category_distribution": category_path,
        "per_stop": stop_path,
        "histograms": make_histograms(frame, figures_dir),
        "scatter": make_scatter(frame, figures_dir),
    }
    map_path = make_map(frame, figures_dir)
    if map_path is not None:
        artifacts["map"] = map_path
    return artifacts
