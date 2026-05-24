"""CLI for streetaqi air quality analysis tools."""

from pathlib import Path

import click


@click.group()
@click.version_option()
def main():
    """Street-level air quality analysis tools."""
    pass


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
    default="gemini-2.0-flash",
    help="Model to use (default: gemini-2.0-flash). Options: gemini-2.0-flash, claude-haiku-4-5",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/annotations"),
    help="Output directory for results (default: output/annotations)",
)
@click.option(
    "--manifest",
    type=click.Path(exists=True, path_type=Path),
    help="Path to manifest.json to merge logged values with OCR readings",
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
    manifest: Path | None,
    batch_id: str | None,
    batch: bool,
    poll_interval: int,
):
    """OCR air quality sensor readings from images using Claude or Gemini APIs."""
    from streetaqi.annotate import find_images, process

    image_paths = find_images(images)
    if not image_paths:
        raise click.ClickException(f"No images found matching pattern: {images}")

    click.echo(f"Found {len(image_paths)} images")

    process(
        images=image_paths,
        output_dir=output,
        model=model,
        manifest_path=manifest,
        batch_id=batch_id,
        use_batch=batch,
        poll_interval=poll_interval,
    )


@main.command()
@click.option(
    "--readings",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pollution_readings.json file from annotate command",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output HTML file path (default: same as input with .html extension)",
)
def viewer(readings: Path, output: Path | None):
    """Generate HTML viewer for OCR results with QC capabilities."""
    from streetaqi.viewer import process

    process(readings, output)


@main.command()
@click.option(
    "--readings",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pollution_logs.csv file",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/analysis"),
    help="Output directory for analysis results (default: output/analysis)",
)
def analyze(readings: Path, output: Path):
    """Run statistical analysis on air quality data."""
    from streetaqi.analyze import process

    process(readings, output)


if __name__ == "__main__":
    main()
