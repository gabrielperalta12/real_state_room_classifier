"""Extract images from Urbania listings using Selenium."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DEFAULT_OUTPUT_DIR
from ..utils import ensure_dir
from .urbania import UrbaniaScraper


def run_extraction(
    base_url: str = "https://urbania.pe",
    output_dir: str | Path = "data/scrapped",
    max_pages: int = 10,
) -> dict[str, Path]:
    """Run the image extraction pipeline with Selenium."""
    output_path = Path(output_dir)
    ensure_dir(output_path)

    config = {"base_url": base_url, "max_pages": max_pages}
    scraper = UrbaniaScraper(config, data_dir=output_path)

    listings = scraper.scrape_listings(max_pages)
    scraper.download_all_images()
    scraper.save_listings()

    print(f"Scraped listings: {len(listings)}")
    print(f"Images folder: {output_path}")

    return {"output_dir": output_path, "listings_file": output_path / "listings.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images from Urbania listings using Selenium.")
    parser.add_argument("--base_url", default="https://urbania.pe", help="Urbania base URL.")
    parser.add_argument("--output_dir", default="data/scrapped", help="Directory for downloaded images.")
    parser.add_argument("--max_pages", type=int, default=10, help="Maximum pages to scrape.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extraction(
        base_url=args.base_url,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
