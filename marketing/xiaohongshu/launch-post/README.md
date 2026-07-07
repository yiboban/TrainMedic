# TrainMedic Xiaohongshu Launch Post Assets

This folder contains a ready-to-publish Xiaohongshu launch image set for
TrainMedic.

## Files

- `copywriting.md`: post titles, full post body, short post body, pinned comment,
  and hashtags.
- `image-plan.md`: text and design notes for each of the 8 images.
- `generate_assets.py`: generates editable SVG files and PNG exports from the same
  page definitions.
- `validate_assets.py`: checks file count, PNG size, SVG XML, core text, and
  external resource references.
- `source/`: editable SVG files.
- `export/`: upload-ready PNG files.
- `contact-sheet.png`: preview sheet for quick manual review.

## Image Specs

- Size: `1080 x 1440`
- Ratio: `3:4`
- Format for upload: PNG
- Source format: SVG
- Style: dark technical UI, medical diagnostic metaphor, terminal cards, clear
  evidence-oriented copy.

## Regenerate Assets

From the repository root:

```bash
python marketing/xiaohongshu/launch-post/generate_assets.py
python marketing/xiaohongshu/launch-post/validate_assets.py
```

The script does not download network assets. PNG rendering uses Pillow and local
system fonts.

## Font Fallbacks

SVG files use this font stack:

```css
"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif
```

The PNG generator looks for common local fonts such as Microsoft YaHei, SimHei,
PingFang, Noto Sans CJK, and DejaVu. No font files are stored in this repository.

## Recommended Posting Order

1. `01-cover.png`
2. `02-pain-points.png`
3. `03-why-trainmedic.png`
4. `04-nan-example.png`
5. `05-features.png`
6. `06-diagnostic-structure.png`
7. `07-project-status.png`
8. `08-call-to-action.png`

## How To Edit Text

Edit the page definitions in `generate_assets.py`, then regenerate assets. The SVG
and PNG files are produced from the same page definitions so they stay aligned.

## No External Copyright Assets

The images use self-drawn SVG/Pillow shapes only. They do not reference external
network images, downloaded illustrations, external logo files, or bundled fonts.

## Pre-Publish Checklist

- Confirm all PNG files are exactly `1080 x 1440`.
- Open `contact-sheet.png` and check the full sequence at a glance.
- Open the 8 exported PNG files on a phone-sized preview.
- Confirm no title is clipped and no text overlaps.
- Confirm `yiboban/TrainMedic` is spelled correctly.
- Confirm the public APIs are spelled correctly:
  `inspect_optimizer`, `watch_forward`, `watch_gradients`, `watch_updates`,
  `watch_modes`.
- Confirm the post does not claim PyPI, CLI, Web UI, automatic fixes, unified
  `watch()`, Lightning integration, Transformers Trainer integration, or full
  distributed support.
- Confirm the project is described as an Alpha test version.
- Confirm the NaN wording says "first observed abnormal output location", not
  guaranteed root cause.
