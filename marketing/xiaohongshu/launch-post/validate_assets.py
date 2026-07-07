"""Validate Xiaohongshu launch assets."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

WIDTH = 1080
HEIGHT = 1440
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
EXPORT_DIR = ROOT / "export"
CONTACT_SHEET = ROOT / "contact-sheet.png"

EXPECTED = [
    "01-cover",
    "02-pain-points",
    "03-why-trainmedic",
    "04-nan-example",
    "05-features",
    "06-diagnostic-structure",
    "07-project-status",
    "08-call-to-action",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"(?:href|src)=['\"]https?://"),
    re.compile(r"file://"),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
]

REQUIRED_TEXT = {
    "01-cover": ["PyTorch", "TrainMedic", "NaN", "grad=None"],
    "04-nan-example": ["watch_forward", "TM3001", "invalid_log", "nan_count: 1"],
    "05-features": ["optimizer", "grad=None", "train / eval"],
    "07-project-status": ["Alpha", "PyPI", "DDP / FSDP / DeepSpeed"],
    "08-call-to-action": ["yiboban/TrainMedic", "Issue"],
}


def main() -> int:
    errors: list[str] = []
    for stem in EXPECTED:
        svg = SOURCE_DIR / f"{stem}.svg"
        png = EXPORT_DIR / f"{stem}.png"
        if not svg.exists():
            errors.append(f"Missing SVG: {svg}")
            continue
        if not png.exists():
            errors.append(f"Missing PNG: {png}")
            continue

        text = svg.read_text(encoding="utf-8")
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            errors.append(f"Invalid SVG XML {svg}: {exc}")
            continue

        if root.attrib.get("width") != str(WIDTH) or root.attrib.get("height") != str(HEIGHT):
            errors.append(f"Wrong SVG size: {svg}")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"Forbidden external/local reference in {svg}: {pattern.pattern}")
        for required in REQUIRED_TEXT.get(stem, []):
            if required not in text:
                errors.append(f"Missing required text in {svg}: {required}")

        with Image.open(png) as image:
            if image.size != (WIDTH, HEIGHT):
                errors.append(f"Wrong PNG size {png}: {image.size}")
            if image.format != "PNG":
                errors.append(f"Wrong PNG format {png}: {image.format}")

    if not CONTACT_SHEET.exists():
        errors.append(f"Missing contact sheet: {CONTACT_SHEET}")
    else:
        with Image.open(CONTACT_SHEET) as image:
            if image.format != "PNG":
                errors.append(f"Wrong contact sheet format: {image.format}")

    if errors:
        for error in errors:
            print(error)
        return 1

    print("asset validation passed")
    print(f"svg_count={len(EXPECTED)} png_count={len(EXPECTED)} png_size={WIDTH}x{HEIGHT}")
    print(f"contact_sheet={CONTACT_SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
