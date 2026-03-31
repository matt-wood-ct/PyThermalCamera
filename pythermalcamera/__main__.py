"""
Thermal Camera Library Demo Script.

Demonstrates the use of the pythermalcamera library for interacting with 
the Topdon TC001 thermal camera.

Based on work by Les Wright (https://github.com/leswright1977/PyThermalCamera)
and downstream researcher LeoDJ, who reverse engineered the thermal image 
format to extract raw temperature data.
See LeoDJ's work here: https://github.com/LeoDJ/P2Pro-Viewer
"""

import argparse
import sys
import os
import time
import cv2

# Import from the package
from .camera import ThermalCamera

def main():
    parser = argparse.ArgumentParser(description="Thermal Camera Library Demo")
    parser.add_argument("--device", type=int, default=None, help="Video Device number")
    parser.add_argument("--preview", action="store_true", help="Enable live preview")
    parser.add_argument("--capture", action="store_true", help="Perform a manual capture")
    parser.add_argument("--markers", action="store_true", help="Include markers on capture")
    args = parser.parse_args()

    print(f"Initializing Thermal Camera on device {args.device}...")
    
    try:
        with ThermalCamera(device_id=args.device, include_preview=False) as cam:
            if args.preview:
                # Example 1: Use the built-in live preview (blocking)
                print("Starting live preview. Press 'q' to quit, 'p' to take a snapshot.")
                cam.live_preview(colormap=cv2.COLORMAP_JET, scale=3, blur=1)

            # Example 2: Manual capture via API
            if args.capture:
                result = cam.capture(filename_prefix="Manual_Capture", include_markers=args.markers)
                if result:
                    print(f"Success!")
                    print(f"Image saved to: {result['image']}")
                    print(f"Metadata saved to: {result['metadata_file']}")
                    print(f"Min Temp: {result['metadata']['min_temp']} C")
                    print(f"Max Temp: {result['metadata']['max_temp']} C")
                    print(f"Average Temp: {result['metadata']['avg_temp']} C")
                else:
                    print("Failed to capture frame.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
