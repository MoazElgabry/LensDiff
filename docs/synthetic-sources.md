# LensDiff Synthetic Sources

## Purpose
- Generate deterministic linear-light validation plates for `LensDiff` without relying on ad hoc footage.
- Keep the plates simple enough to diagnose PSF truth, threshold behavior, spectral mapping, and backend parity.

## Dependencies
- Python 3.14+
- `numpy`
- `tifffile`

Install them with:

```powershell
python -m pip install -r .\tools\synthetic_requirements.txt
```

## Generate The Starter Suite

```powershell
python .\tools\generate_synthetic_sources.py
```

Default output:

- `.\testdata\synthetic`

## Useful Variants

Generate only a subset:

```powershell
python .\tools\generate_synthetic_sources.py --plates impulse_center point_grid spectral_points
```

Generate a larger suite:

```powershell
python .\tools\generate_synthetic_sources.py --width 2048 --height 2048 --odd-width 2047 --odd-height 1151
```

## Plate Guide

- `impulse_center`
  - Use for PSF-truth checks, kernel normalization, and backend parity.
- `impulse_edge`
  - Use for ROI expansion, edge safety, and wraparound detection.
- `point_grid`
  - Use for field consistency, repeated renders, and geometry drift.
- `exposure_bars`
  - Use for threshold, softness, and exposure-invariance tuning.
- `spectral_points`
  - Use for `Natural`, `Cyan-Magenta`, and `Warm-Cool` endpoint evaluation.
- `window_slits`
  - Use for directional structure, spike readability, and `anisotropyEmphasis`.
- `odd_dim_grid`
  - Use for odd-size host parity and cropped render-window regressions.
- `annular_probe`
  - Use for circular symmetry, obstruction readability, and warm/cool ring separation.
- `spider_vane_probe`
  - Use for vane count, vane rotation, and directional spike readability.
- `practicals_beauty`
  - Use for default tuning, shoulder behavior, and beauty-oriented look checks on a controlled scene.
- `diagonal_glints`
  - Use for rotation sensitivity, asymmetry, and directional stability checks.

## Format Notes

- The generator writes compressed `float32` RGB TIFF files in linear light.
- A `manifest.json` file is written beside the plates with descriptions, dimensions, peaks, and intended checks.
- TIFF was chosen for the first pass because it is stable to generate from Python 3.14 with lightweight dependencies, while still preserving linear float data for host tests.
- The second-pass suite adds more look-oriented but still deterministic plates so you can tune creative endpoints without immediately jumping to live-action footage.
