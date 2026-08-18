"""CLI for streetaqi air quality analysis tools."""

import logging
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="streetaqi")
def main() -> None:
    """Street-level air quality analysis tools."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.option(
    "--images",
    type=str,
    required=True,
    help="Glob pattern for images (e.g., 'images/pollution/**/*.jpg')",
)
@click.option(
    "--model",
    type=str,
    default="gemini-3.5-flash",
    show_default=True,
    help="Gemini or Claude model ID.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/annotations"),
    show_default=True,
    help="Output directory for results.",
)
@click.option(
    "--reference",
    type=click.Path(exists=True, path_type=Path),
    help="Canonical readings Parquet for reference-value comparison.",
)
@click.option(
    "--batch-id",
    type=str,
    help="Resume from existing Claude batch ID",
)
@click.option(
    "--batch",
    is_flag=True,
    help="Use batch API for Gemini (50% cost savings, async processing)",
)
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    help="Poll interval in seconds for batch processing (default: 30)",
)
def annotate(
    images: str,
    model: str,
    output: Path,
    reference: Path | None,
    batch_id: str | None,
    batch: bool,
    poll_interval: int,
) -> None:
    """OCR air quality sensor readings from images using Claude or Gemini APIs."""
    from streetaqi.annotate import find_images, process

    image_paths = find_images(images)
    if not image_paths:
        raise click.ClickException(f"No images found matching pattern: {images}")

    click.echo(f"Found {len(image_paths)} images")

    result = process(
        images=image_paths,
        output_dir=output,
        model=model,
        reference_path=reference,
        batch_id=batch_id,
        use_batch=batch,
        poll_interval=poll_interval,
    )
    click.echo(f"OCR results: {result}")


@main.command()
@click.option(
    "--readings",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="OCR-result Parquet file from the annotate command.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output HTML file path (default: same as input with .html extension)",
)
@click.option(
    "--image-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Trusted directory from which images may be embedded.",
)
def viewer(readings: Path, output: Path | None, image_root: Path | None) -> None:
    """Generate HTML viewer for OCR results with QC capabilities."""
    from streetaqi.viewer import process

    generated = process(readings, output, image_root)
    click.echo(f"Generated viewer: {generated}")


@main.command()
@click.option(
    "--readings",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Canonical readings Parquet or import-boundary CSV.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/analysis"),
    show_default=True,
    help="Output directory for analysis results.",
)
def analyze(readings: Path, output: Path) -> None:
    """Run statistical analysis on air quality data."""
    from streetaqi.analyze import process

    artifacts = process(readings, output)
    for name, path in artifacts.items():
        click.echo(f"{name}: {path}")


@main.command("sample-data")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("delhi_readings.parquet"),
    show_default=True,
    help="Destination for the bundled canonical readings.",
)
def sample_data(output: Path) -> None:
    """Export the bundled Delhi readings as canonical Parquet."""
    from streetaqi.data import load_bundled_readings, write_readings

    written = write_readings(load_bundled_readings(), output)
    click.echo(f"Sample data: {written}")


if __name__ == "__main__":
    main()
