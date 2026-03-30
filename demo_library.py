import argparse
import sys
import os

# Add src to path so we can import the library without installing it
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pythermalcamera import ThermalCamera
import cv2

def main():
    parser = argparse.ArgumentParser(description="Thermal Camera Library Demo")
    parser.add_argument("--device", type=int, default=None, help="Video Device number")
    args = parser.parse_args()

    print(f"Initializing Thermal Camera on device {args.device}...")
    
    with ThermalCamera(device_id=args.device) as cam:
        # Example 1: Use the built-in live preview
        print("Starting live preview. Press 'q' to quit, 'p' to take a snapshot.")
        cam.live_preview(colormap=cv2.COLORMAP_JET, scale=3, blur=1)
        
        # Example 2: Manual capture via API
        print("\nTaking a manual capture via API...")
        result = cam.capture(filename_prefix="Manual_Capture")
        if result:
            print(f"Success!")
            print(f"Image saved to: {result['image']}")
            print(f"Metadata saved to: {result['metadata_file']}")
            print(f"Min Temp: {result['metadata']['min_temp']} C")
            print(f"Max Temp: {result['metadata']['max_temp']} C")
            print(f"Average Temp: {result['metadata']['avg_temp']} C")
        else:
            print("Failed to capture frame.")

if __name__ == "__main__":
    main()
