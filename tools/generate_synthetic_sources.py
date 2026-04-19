from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile

MIDDLE_GRAY = 0.18


@dataclass(frozen=True)
class PlateRecord:
    filename: str
    description: str
    width: int
    height: int
    peak: float
    checks: list[str]


def middle_gray_stop_value(stops: float) -> float:
    return float(MIDDLE_GRAY * (2.0 ** stops))


def empty_rgb(width: int, height: int, value: float = 0.0) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.float32)


def put_points(image: np.ndarray, positions: list[tuple[int, int]], rgb: tuple[float, float, float]) -> None:
    height, width, _ = image.shape
    value = np.asarray(rgb, dtype=np.float32)
    for x, y in positions:
        if 0 <= x < width and 0 <= y < height:
            image[y, x, :] = value


def put_disc(image: np.ndarray,
             center_x: float,
             center_y: float,
             radius: float,
             rgb: tuple[float, float, float]) -> None:
    height, width, _ = image.shape
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    image[mask] = np.asarray(rgb, dtype=np.float32)


def put_rect(image: np.ndarray,
             x0: int,
             y0: int,
             x1: int,
             y1: int,
             rgb: tuple[float, float, float]) -> None:
    image[max(0, y0):max(0, y1), max(0, x0):max(0, x1), :] = np.asarray(rgb, dtype=np.float32)


def put_line(image: np.ndarray,
             x0: float,
             y0: float,
             x1: float,
             y1: float,
             thickness: float,
             rgb: tuple[float, float, float]) -> None:
    height, width, _ = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    px = xx.astype(np.float32)
    py = yy.astype(np.float32)
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        put_disc(image, x0, y0, max(0.5, thickness * 0.5), rgb)
        return
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = np.clip(t, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    dist_sq = (px - cx) ** 2 + (py - cy) ** 2
    mask = dist_sq <= (thickness * 0.5) ** 2
    image[mask] = np.asarray(rgb, dtype=np.float32)


def put_gradient_rect(image: np.ndarray,
                      x0: int,
                      y0: int,
                      x1: int,
                      y1: int,
                      rgb0: tuple[float, float, float],
                      rgb1: tuple[float, float, float],
                      horizontal: bool = True) -> None:
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = max(x0, min(image.shape[1], x1))
    y1 = max(y0, min(image.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    a = np.asarray(rgb0, dtype=np.float32)
    b = np.asarray(rgb1, dtype=np.float32)
    if horizontal:
        ramp = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float32)[None, :, None]
        image[y0:y1, x0:x1, :] = a[None, None, :] * (1.0 - ramp) + b[None, None, :] * ramp
    else:
        ramp = np.linspace(0.0, 1.0, y1 - y0, dtype=np.float32)[:, None, None]
        image[y0:y1, x0:x1, :] = a[None, None, :] * (1.0 - ramp) + b[None, None, :] * ramp


def impulse_center(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(10.0)
    put_points(image, [(width // 2, height // 2)], (hot, hot, hot))
    return image, "Single white impulse at frame center for PSF-truth and backend parity checks.", [
        "PSF truth",
        "backend parity",
        "preserve-mode energy",
    ]


def impulse_edge(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(10.0)
    inset = max(4, min(width, height) // 32)
    positions = [
        (inset, inset),
        (width - inset - 1, inset),
        (inset, height - inset - 1),
        (width - inset - 1, height - inset - 1),
    ]
    put_points(image, positions, (hot, hot, hot))
    return image, "Hot corner impulses to stress ROI expansion, edge clipping, and wraparound safety.", [
        "edge handling",
        "ROI expansion",
        "wraparound safety",
    ]


def point_grid(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(9.0)
    cols = 7
    rows = 5
    xs = np.linspace(width * 0.15, width * 0.85, cols).astype(int)
    ys = np.linspace(height * 0.18, height * 0.82, rows).astype(int)
    put_points(image, [(int(x), int(y)) for y in ys for x in xs], (hot, hot, hot))
    return image, "Sparse white point grid for PSF consistency across the frame and repeated render checks.", [
        "field consistency",
        "backend parity",
        "repeated renders",
    ]


def exposure_bars(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    stops = [-4, -2, 0, 2, 4, 6, 8, 10]
    margin_x = max(8, width // 24)
    margin_y = max(8, height // 8)
    spacing = max(4, width // 256)
    bar_width = max(8, (width - 2 * margin_x - spacing * (len(stops) - 1)) // len(stops))
    x = margin_x
    for stop in stops:
        value = middle_gray_stop_value(float(stop))
        put_rect(image, x, margin_y, x + bar_width, height - margin_y, (value, value, value))
        x += bar_width + spacing
    return image, "Vertical exposure bars from -4 to +10 stops relative to 18% gray for threshold and softness tuning.", [
        "selection threshold",
        "softness",
        "exposure invariance",
    ]


def spectral_points(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(9.0)
    radius = max(3.0, min(width, height) / 48.0)
    centers = [
        (width * 0.20, height * 0.50, (hot, hot, hot)),
        (width * 0.40, height * 0.50, (hot, 0.0, 0.0)),
        (width * 0.60, height * 0.50, (0.0, hot, 0.0)),
        (width * 0.80, height * 0.50, (0.0, 0.0, hot)),
    ]
    for cx, cy, rgb in centers:
        put_disc(image, cx, cy, radius, rgb)
    return image, "White plus RGB primary discs for source-color interaction and spectral-style evaluation.", [
        "natural mode",
        "creative spectrum styles",
        "chromatic luma coupling",
    ]


def window_slits(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    frame_x0 = int(width * 0.28)
    frame_x1 = int(width * 0.72)
    frame_y0 = int(height * 0.16)
    frame_y1 = int(height * 0.84)
    bright = middle_gray_stop_value(8.0)
    hotter = middle_gray_stop_value(10.0)
    slit_count = 18
    slit_gap = max(3, (frame_y1 - frame_y0) // (slit_count * 2))
    slit_height = max(2, slit_gap)
    for slit in range(slit_count):
        y0 = frame_y0 + slit * (slit_height + slit_gap)
        y1 = min(frame_y1, y0 + slit_height)
        put_rect(image, frame_x0, y0, frame_x1, y1, (bright, bright, bright))
    hot_strip = max(4, width // 128)
    put_rect(image, frame_x1 - hot_strip, frame_y0, frame_x1, frame_y1, (hotter, hotter, hotter))
    return image, "Bright slit window with a hotter edge strip for directional structure and spike readability checks.", [
        "directional structure",
        "core versus structure split",
        "anisotropy emphasis",
    ]


def odd_dim_grid(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(9.0)
    xs = np.linspace(width * 0.12, width * 0.88, 6).astype(int)
    ys = np.linspace(height * 0.18, height * 0.82, 4).astype(int)
    put_points(image, [(int(x), int(y)) for y in ys for x in xs], (hot, hot, hot))
    return image, "Odd-dimension point grid for host geometry, tile, and parity regression checks.", [
        "odd dimensions",
        "cropped windows",
        "backend parity",
    ]


def annular_probe(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    center_x = width * 0.5
    center_y = height * 0.5
    ring_radius = min(width, height) * 0.22
    point_radius = max(3.0, min(width, height) / 96.0)
    hot = middle_gray_stop_value(9.0)
    warm = (hot, hot * 0.72, hot * 0.46)
    cool = (hot * 0.52, hot * 0.78, hot)
    put_disc(image, center_x, center_y, point_radius * 0.9, (hot, hot, hot))
    for i in range(12):
        angle = (2.0 * np.pi * i) / 12.0
        rgb = warm if i % 2 == 0 else cool
        put_disc(
            image,
            center_x + np.cos(angle) * ring_radius,
            center_y + np.sin(angle) * ring_radius,
            point_radius,
            rgb,
        )
    return image, "Central impulse plus annular ring of alternating warm and cool practicals for obstruction readability and circular spectral separation checks.", [
        "central obstruction readability",
        "circular symmetry",
        "creative spectrum tuning",
    ]


def spider_vane_probe(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(10.0)
    warm = (hot, hot * 0.75, hot * 0.48)
    center_x = width * 0.5
    center_y = height * 0.5
    line_len = min(width, height) * 0.36
    thickness = max(3.0, min(width, height) / 128.0)
    put_disc(image, center_x, center_y, max(3.0, thickness * 0.9), (hot, hot, hot))
    for angle_deg in (0.0, 45.0, 90.0, 135.0):
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle) * line_len
        dy = np.sin(angle) * line_len
        put_line(image, center_x - dx, center_y - dy, center_x + dx, center_y + dy, thickness, warm)
    ring_radius = min(width, height) * 0.18
    for angle_deg in range(0, 360, 30):
        angle = np.deg2rad(float(angle_deg))
        put_disc(
            image,
            center_x + np.cos(angle) * ring_radius,
            center_y + np.sin(angle) * ring_radius,
            max(2.5, thickness * 0.7),
            (hot * 0.9, hot * 0.9, hot * 0.9),
        )
    return image, "Central star probe with aligned bright arms and ring points for vane-count, rotation, and directional-spike readability tests.", [
        "spider vane stress",
        "rotation",
        "anisotropy emphasis",
    ]


def practicals_beauty(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height, value=middle_gray_stop_value(-6.0))
    shadow = middle_gray_stop_value(-7.5)
    warm_wall = (middle_gray_stop_value(-2.0), middle_gray_stop_value(-3.0), middle_gray_stop_value(-4.2))
    cool_window = (middle_gray_stop_value(6.5), middle_gray_stop_value(6.8), middle_gray_stop_value(7.1))
    ambient = (shadow * 1.1, shadow * 0.9, shadow * 1.2)
    image[:] = np.asarray(ambient, dtype=np.float32)

    put_gradient_rect(
        image,
        0,
        0,
        width,
        height,
        warm_wall,
        (warm_wall[0] * 0.32, warm_wall[1] * 0.30, warm_wall[2] * 0.42),
        horizontal=True,
    )
    win_x0 = int(width * 0.30)
    win_x1 = int(width * 0.54)
    win_y0 = int(height * 0.14)
    win_y1 = int(height * 0.78)
    put_gradient_rect(image, win_x0, win_y0, win_x1, win_y1, cool_window, (cool_window[0] * 0.9, cool_window[1], cool_window[2]), horizontal=False)

    slit_gap = max(5, height // 72)
    slit_height = max(2, slit_gap // 2)
    for y in range(win_y0, win_y1, slit_gap):
        put_rect(image, win_x0, y, win_x1, min(win_y1, y + slit_height), (0.0, 0.0, 0.0))

    put_disc(image, width * 0.81, height * 0.43, min(width, height) * 0.09, (middle_gray_stop_value(8.8), middle_gray_stop_value(7.6), middle_gray_stop_value(6.4)))
    put_disc(image, width * 0.15, height * 0.37, min(width, height) * 0.08, (middle_gray_stop_value(7.4), middle_gray_stop_value(6.4), middle_gray_stop_value(5.6)))

    string_y = height * 0.12
    for i in range(10):
        x = width * (0.08 + 0.084 * i)
        y = string_y + np.sin(i * 0.55) * height * 0.015
        hot = middle_gray_stop_value(7.6 + (i % 3) * 0.4)
        rgb = (hot, hot * 0.82, hot * 0.60) if i % 2 == 0 else (hot * 0.78, hot * 0.84, hot)
        put_disc(image, x, y, max(2.5, min(width, height) / 180.0), rgb)

    cymbal = (middle_gray_stop_value(6.2), middle_gray_stop_value(5.5), middle_gray_stop_value(4.4))
    put_disc(image, width * 0.43, height * 0.86, min(width, height) * 0.05, cymbal)
    put_disc(image, width * 0.56, height * 0.70, min(width, height) * 0.018, (middle_gray_stop_value(8.0), middle_gray_stop_value(8.0), middle_gray_stop_value(8.0)))
    return image, "Warm interior practical-light scene with blown window, string lights, and lamp sources for default tuning and beauty-oriented evaluation.", [
        "default look tuning",
        "mixed practicals",
        "shoulder behavior",
    ]


def diagonal_glints(width: int, height: int) -> tuple[np.ndarray, str, list[str]]:
    image = empty_rgb(width, height)
    hot = middle_gray_stop_value(9.5)
    cool = (hot * 0.64, hot * 0.78, hot)
    warm = (hot, hot * 0.74, hot * 0.52)
    margin = min(width, height) * 0.12
    thickness = max(2.0, min(width, height) / 160.0)
    put_line(image, margin, height - margin, width - margin, margin, thickness, warm)
    put_line(image, margin, margin, width - margin, height - margin, thickness * 0.8, cool)
    for t in np.linspace(0.1, 0.9, 9):
        put_disc(image, margin + (width - 2 * margin) * t, height * 0.5 + np.sin(t * np.pi * 2.0) * height * 0.18, thickness * 1.25, (hot, hot, hot))
    return image, "Diagonal glints and offset point accents for rotation sensitivity, directional stability, and asymmetry checks.", [
        "rotation sensitivity",
        "directional stability",
        "asymmetry checks",
    ]


PLATE_BUILDERS: dict[str, Callable[[int, int], tuple[np.ndarray, str, list[str]]]] = {
    "impulse_center": impulse_center,
    "impulse_edge": impulse_edge,
    "point_grid": point_grid,
    "exposure_bars": exposure_bars,
    "spectral_points": spectral_points,
    "window_slits": window_slits,
    "odd_dim_grid": odd_dim_grid,
    "annular_probe": annular_probe,
    "spider_vane_probe": spider_vane_probe,
    "practicals_beauty": practicals_beauty,
    "diagonal_glints": diagonal_glints,
}


def write_tiff(path: Path, image: np.ndarray, description: str) -> None:
    tifffile.imwrite(
        path,
        image.astype(np.float32, copy=False),
        photometric="rgb",
        compression="deflate",
        metadata=None,
        description=description,
    )


def build_suite(output_dir: Path,
                width: int,
                height: int,
                odd_width: int,
                odd_height: int,
                selected: list[str]) -> list[PlateRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[PlateRecord] = []
    for name in selected:
        builder = PLATE_BUILDERS[name]
        plate_width = odd_width if name == "odd_dim_grid" else width
        plate_height = odd_height if name == "odd_dim_grid" else height
        image, description, checks = builder(plate_width, plate_height)
        filename = f"{name}_{plate_width}x{plate_height}.tif"
        write_tiff(output_dir / filename, image, description)
        manifest.append(
            PlateRecord(
                filename=filename,
                description=description,
                width=plate_width,
                height=plate_height,
                peak=float(np.max(image)),
                checks=checks,
            )
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic float32 TIFF plates for LensDiff validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "synthetic",
        help="Destination directory for generated plates and manifest.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Default width for standard plates.")
    parser.add_argument("--height", type=int, default=1024, help="Default height for standard plates.")
    parser.add_argument("--odd-width", type=int, default=1023, help="Width used by the odd-dimension plate.")
    parser.add_argument("--odd-height", type=int, default=577, help="Height used by the odd-dimension plate.")
    parser.add_argument(
        "--plates",
        nargs="+",
        choices=sorted(PLATE_BUILDERS.keys()),
        default=list(PLATE_BUILDERS.keys()),
        help="Subset of plates to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_suite(
        output_dir=args.output_dir,
        width=max(16, args.width),
        height=max(16, args.height),
        odd_width=max(17, args.odd_width),
        odd_height=max(17, args.odd_height),
        selected=args.plates,
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generator": "LensDiff synthetic source generator",
                "linear_light": True,
                "format": "float32 TIFF",
                "middle_gray": MIDDLE_GRAY,
                "plates": [record.__dict__ for record in manifest],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} plate(s) in {args.output_dir}")
    for record in manifest:
        print(f" - {record.filename}: peak={record.peak:.4f}")


if __name__ == "__main__":
    main()
