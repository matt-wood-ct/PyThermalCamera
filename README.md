# pythermalcamera - Topdon TC001 Thermal Camera Library

[![PyPI version](https://badge.fury.io/py/pythermalcamera.svg)](https://pypi.org/project/pythermalcamera/)

A Python library for interacting with the **Topdon TC001** thermal camera. This library simplifies device discovery, provides a live interactive preview with temperature analysis, and supports capturing full-resolution thermal images with associated JSON metadata.

## Features

- **Auto-Detection**: Automatically finds the correct video device ID for the TC001 by scanning `/dev/video*`.
- **Live Preview**: Interactive window showing the thermal heatmap with a center crosshair, HUD, and real-time statistics (Min/Max/Avg).
- **Area of Interest (ROI)**: Interactively select a region in the preview to focus temperature analysis (statistics will only reflect the chosen box).
- **High-Quality Captures**: Save colorized heatmaps as PNG and full temperature metadata as JSON.
- **Marker Overlays**: Toggleable hotspot and coldspot markers on both the preview and captured images.
- **Background Threading**: Optionally run the preview in a non-blocking background thread while performing other tasks in the main script.
- **Extensive Metadata**: Captures include timestamps, raw temperature stats, ROI coordinates, and all rendering settings (colormap, alpha, blur, etc.).

## Prerequisites

- **Hardware**: Topdon TC001 Thermal Camera.
- **Operating System**: Linux (developed and tested on Linux with V4L2).
- **Dependencies**:
  - `opencv-python`
  - `numpy`

## Installation

Install the library directly from PyPI:

```bash
pip install pythermalcamera
```

## Usage

### Quick Start (CLI Demo)

Once installed, you can run the built-in demo to see the library in action. This will auto-detect your TC001 and start an interactive preview:

```bash
# Start interactive preview
pythermalcamera

# Alternatively, run as a module:
python3 -m pythermalcamera

# Use --preview to run in background and take a manual capture after 5s
pythermalcamera --preview

# Enable markers on the manual capture
pythermalcamera --markers
```

### Basic Library API

```python
from pythermalcamera import ThermalCamera
import cv2

# Initialize with auto-detection and non-blocking preview
with ThermalCamera(include_preview=True) as cam:
    # Do something else while the preview runs...
    import time
    time.sleep(5)
    
    # Take a manual snapshot with specific settings
    result = cam.capture(
        filename_prefix="Snapshot",
        colormap=cv2.COLORMAP_MAGMA,
        include_markers=True
    )
    
    if result:
        print(f"Captured {result['image']}")
        print(f"Max Temp: {result['metadata']['max_temp']}°C")
```

## Interactive Controls (Preview Window)

When the preview window is active, use the following keyboard shortcuts:

| Key | Action |
|-----|--------|
| **q** | Quit preview |
| **p** | Take snapshot (saves PNG + JSON) |
| **r** | Reset/Clear ROI (Area of Interest) |
| **m** | Cycle through available colormaps |
| **h** | Toggle HUD (On-screen statistics) |
| **k** | Toggle markers (hotspot/coldspot) for snapshots |
| **a/z** | Increase / Decrease Blur |
| **s/x** | Increase / Decrease Marker Threshold |
| **d/c** | Increase / Decrease Display Scale |
| **f/v** | Increase / Decrease Contrast (Alpha) |

**Mouse Controls:**
- **Left-Click & Drag**: Select an Region of Interest (ROI) box on the preview.

## Metadata Format

Snapshots generate a `.json` file containing:
- `timestamp`: Unix timestamp of the capture.
- `center_temp`, `max_temp`, `min_temp`, `avg_temp`: Temperature readings in Celsius.
- `max_pos`, `min_pos`: Pixel coordinates of the hotspot and coldspot.
- `roi`: Coordinates of the active ROI at the time of capture.
- `settings`: All rendering parameters used to generate the PNG (colormap, scale, blur, etc.).

## Credits and Attribution

This library is inspired by and based on the work of:
- **Les Wright's PyThermalCamera**: [https://github.com/leswright1977/PyThermalCamera](https://github.com/leswright1977/PyThermalCamera)
- **Researcher LeoDJ**: For their significant contributions and enhancements to the thermal camera research. Specifically, huge kudos to LeoDJ for reverse engineering the thermal image format to extract raw temperature data. 
  - [EEVBlog forum discussion](https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/)
  - [LeoDJ's P2Pro-Viewer GitHub](https://github.com/LeoDJ/P2Pro-Viewer/tree/main)

## License

This project is licensed under the MIT License - see the LICENSE file for details (if provided).
