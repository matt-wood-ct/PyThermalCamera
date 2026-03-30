import cv2
import numpy as np
import time
import io
import os
import json
import threading


class ThermalFrame:
    """Class representing a single frame from the thermal camera."""
    
    def __init__(self, raw_frame):
        self.raw_frame = raw_frame
        # The frame is 256x384. Top half (256x192) is image data, 
        # bottom half (256x192) is thermal data.
        self.imdata, self.thdata = np.array_split(raw_frame, 2)
        self.height, self.width = self.imdata.shape[:2]
        self._process_thermal()

    def _process_thermal(self):
        """Extract temperature statistics from thermal data."""
        # Convert thermal data to raw temperatures
        # thdata[..., 0] is hi, thdata[..., 1] is lo in the original code's naming,
        # but lo*256 suggests thdata[..., 1] is the MSB.
        thdata_int = self.thdata.astype(np.int32)
        self.raw_temps = thdata_int[..., 0] + thdata_int[..., 1] * 256
        
        # Center temperature
        cy, cx = self.height // 2, self.width // 2
        self.center_temp = self._raw_to_celsius(self.raw_temps[cy, cx])
        
        # Max temperature
        max_idx = np.argmax(self.raw_temps)
        m_col, m_row = divmod(max_idx, self.width)
        self.max_temp = self._raw_to_celsius(self.raw_temps[m_col, m_row])
        self.max_pos = (m_row, m_col)
        
        # Min temperature
        min_idx = np.argmin(self.raw_temps)
        l_col, l_row = divmod(min_idx, self.width)
        self.min_temp = self._raw_to_celsius(self.raw_temps[l_col, l_row])
        self.min_pos = (l_row, l_col)
        
        # Average temperature
        self.avg_temp = self._raw_to_celsius(np.mean(self.raw_temps))

    def _raw_to_celsius(self, raw):
        return round((raw / 64) - 273.15, 2)

    def get_heatmap(self, colormap=cv2.COLORMAP_JET, alpha=1.0, scale=3, blur=0):
        """Generate a colorized heatmap from the image data."""
        # Convert YUYV to BGR (OpenCV format)
        bgr = cv2.cvtColor(self.imdata, cv2.COLOR_YUV2BGR_YUYV)
        
        if alpha != 1.0:
            bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
        
        new_width = self.width * scale
        new_height = self.height * scale
        bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        if blur > 0:
            bgr = cv2.blur(bgr, (blur, blur))
        
        if colormap == "inv_rainbow":
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        else:
            heatmap = cv2.applyColorMap(bgr, colormap)
            
        return heatmap

class ThermalCamera:
    """Library for interacting with the Topdon TC001 Thermal Camera."""
    
    @staticmethod
    def detect_devices():
        """
        Scan /dev/video* devices to find a potential TC001 camera.
        Returns a list of matching device IDs.
        """
        matches = []
        for i in range(16):
            dev_path = f'/dev/video{i}'
            if not os.path.exists(dev_path):
                continue
                
            cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L)
            if not cap.isOpened():
                continue
            
            # Set to raw mode to check dimensions
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
            
            # Try a couple of frames to be sure
            for _ in range(5):
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                h, w = frame.shape[:2]
                # TC001 is 256x384 (combined image and thermal)
                if (h == 384 and w == 256) or (h == 256 and w == 384):
                    matches.append(i)
                    break
            
            cap.release()
        return matches

    def __init__(self, device_id=None, include_preview=False):
        if device_id is None:
            print("No device ID provided. Attempting to auto-detect Thermal Camera...")
            matches = self.detect_devices()
            if matches:
                device_id = matches[0]
                print(f"Auto-detected Thermal Camera on device {device_id}")
            else:
                device_id = 0
                print("Could not auto-detect. Falling back to device 0.")

        self.device_id = device_id
        # In Linux, VideoCapture can take a path or device ID
        # The original code uses '/dev/video' + str(dev)
        dev_path = f'/dev/video{device_id}'
        if os.path.exists(dev_path):
            self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L)
        else:
            self.cap = cv2.VideoCapture(device_id)
            
        if not self.cap.isOpened():
            print(f"Warning: Could not open video device {device_id}")
            
        # Crucial: Pull in video but do NOT automatically convert to RGB
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        # Default display settings
        self.colormap = cv2.COLORMAP_JET
        self.alpha = 1.0
        self.scale = 3
        self.blur = 0
        self.threshold = 2
        self.hud = True

        self._preview_thread = None
        self._stop_preview = threading.Event()
        
        if include_preview:
            self._preview_thread = threading.Thread(target=self.live_preview, daemon=True)
            self._preview_thread.start()

    def get_frame(self):
        """Capture a single frame and return a ThermalFrame object."""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return ThermalFrame(frame)

    def capture(self, filename_prefix="TC001", folder=".", colormap=None, 
                alpha=None, scale=None, blur=None, frame=None):
        """
        Take a snapshot and store both the image and temperature metadata.
        Returns a dictionary with file paths and metadata.
        """
        if frame is None:
            frame = self.get_frame()
            
        if frame is None:
            return None
        
        # Use instance variables if parameters are not provided
        colormap = colormap if colormap is not None else self.colormap
        alpha = alpha if alpha is not None else self.alpha
        scale = scale if scale is not None else self.scale
        blur = blur if blur is not None else self.blur

        now = time.strftime("%Y%m%d-%H%M%S")
        img_filename = os.path.join(folder, f"{filename_prefix}_{now}.png")
        meta_filename = os.path.join(folder, f"{filename_prefix}_{now}.json")
        
        heatmap = frame.get_heatmap(colormap=colormap, alpha=alpha, scale=scale, blur=blur)
        cv2.imwrite(img_filename, heatmap)
        
        metadata = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "center_temp": float(frame.center_temp),
            "max_temp": float(frame.max_temp),
            "min_temp": float(frame.min_temp),
            "avg_temp": float(frame.avg_temp),
            "max_pos": [int(x) for x in frame.max_pos],
            "min_pos": [int(x) for x in frame.min_pos],
            "settings": {
                "colormap": str(colormap),
                "alpha": alpha,
                "scale": scale,
                "blur": blur
            }
        }
        
        with open(meta_filename, "w") as f:
            json.dump(metadata, f, indent=4)
        
        return {
            "image": img_filename,
            "metadata_file": meta_filename,
            "metadata": metadata
        }

    def live_preview(self, colormap=None, alpha=None, scale=None, blur=None, 
                    threshold=None, hud=None):
        """Start a live preview window with interactive controls."""
        cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
        
        # Override instance variables with parameters if provided
        if colormap is not None: self.colormap = colormap
        if alpha is not None: self.alpha = alpha
        if scale is not None: self.scale = scale
        if blur is not None: self.blur = blur
        if threshold is not None: self.threshold = threshold
        if hud is not None: self.hud = hud

        colormaps = [
            (cv2.COLORMAP_JET, "Jet"),
            (cv2.COLORMAP_HOT, "Hot"),
            (cv2.COLORMAP_MAGMA, "Magma"),
            (cv2.COLORMAP_INFERNO, "Inferno"),
            (cv2.COLORMAP_PLASMA, "Plasma"),
            (cv2.COLORMAP_BONE, "Bone"),
            (cv2.COLORMAP_SPRING, "Spring"),
            (cv2.COLORMAP_AUTUMN, "Autumn"),
            (cv2.COLORMAP_VIRIDIS, "Viridis"),
            (cv2.COLORMAP_PARULA, "Parula"),
            ("inv_rainbow", "Inv Rainbow")
        ]
        
        # Find initial colormap index
        cmap_idx = 0
        for i, (cmap, name) in enumerate(colormaps):
            if cmap == self.colormap:
                cmap_idx = i
                break

        print("\nInteractive Controls:")
        print("  a/z : Increase/Decrease Blur")
        print("  s/x : Increase/Decrease Floating Label Threshold")
        print("  d/c : Increase/Decrease Scale")
        print("  f/v : Increase/Decrease Contrast (Alpha)")
        print("  m   : Cycle ColorMaps")
        print("  h   : Toggle HUD")
        print("  p   : Take Snapshot")
        print("  q   : Quit")
        
        while not self._stop_preview.is_set():
            frame = self.get_frame()
            if frame is None:
                break
            
            current_cmap, cmap_name = colormaps[cmap_idx]
            self.colormap = current_cmap # Sync with instance variable for cycle colormap
            
            heatmap = frame.get_heatmap(colormap=self.colormap, alpha=self.alpha, scale=self.scale, blur=self.blur)
            
            # Draw Crosshair
            h, w = heatmap.shape[:2]
            cv2.line(heatmap, (w//2, h//2-20), (w//2, h//2+20), (255,255,255), 2)
            cv2.line(heatmap, (w//2-20, h//2), (w//2+20, h//2), (255,255,255), 2)
            cv2.putText(heatmap, f"{frame.center_temp} C", (w//2+10, h//2-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{frame.center_temp} C", (w//2+10, h//2-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            if self.hud:
                # Stats box
                cv2.rectangle(heatmap, (0, 0), (180, 120), (0,0,0), -1)
                cv2.putText(heatmap, f"Avg: {frame.avg_temp} C", (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Max: {frame.max_temp} C", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Min: {frame.min_temp} C", (10, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Colormap: {cmap_name}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Blur: {self.blur}", (10, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Scale: {self.scale}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
                cv2.putText(heatmap, f"Contrast: {self.alpha:.1f}", (10, 105), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

            # Floating markers
            if frame.max_temp > frame.avg_temp + self.threshold:
                mx, my = frame.max_pos
                cv2.circle(heatmap, (mx*self.scale, my*self.scale), 5, (0,0,255), -1)
                cv2.putText(heatmap, f"{frame.max_temp} C", (mx*self.scale+10, my*self.scale+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            if frame.min_temp < frame.avg_temp - self.threshold:
                lx, ly = frame.min_pos
                cv2.circle(heatmap, (lx*self.scale, ly*self.scale), 5, (255,0,0), -1)
                cv2.putText(heatmap, f"{frame.min_temp} C", (lx*self.scale+10, ly*self.scale+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow('Thermal', heatmap)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                res = self.capture(frame=frame)
                if res:
                    print(f"Captured: {res['image']}")
            elif key == ord('a'):
                self.blur += 1
            elif key == ord('z'):
                self.blur = max(0, self.blur - 1)
            elif key == ord('s'):
                self.threshold += 1
            elif key == ord('x'):
                self.threshold = max(0, self.threshold - 1)
            elif key == ord('d'):
                self.scale = min(5, self.scale + 1)
            elif key == ord('c'):
                self.scale = max(1, self.scale - 1)
            elif key == ord('f'):
                self.alpha = min(3.0, self.alpha + 0.1)
            elif key == ord('v'):
                self.alpha = max(0.1, self.alpha - 0.1)
            elif key == ord('m'):
                cmap_idx = (cmap_idx + 1) % len(colormaps)
                self.colormap, _ = colormaps[cmap_idx] # Update instance variable
            elif key == ord('h'):
                self.hud = not self.hud
                
        cv2.destroyAllWindows()
        self._stop_preview.set()

    def close(self):
        """Release the camera resources."""
        self._stop_preview.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
