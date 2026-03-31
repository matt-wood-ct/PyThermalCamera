import cv2
import numpy as np
import time
import io
import os
import json
import threading

# Based on work by Les Wright (https://github.com/leswright1977/PyThermalCamera)
# and downstream researcher LeoDJ, who reverse engineered the thermal image 
# format to extract raw temperature data.
# See LeoDJ's work here: https://github.com/LeoDJ/P2Pro-Viewer


class ThermalFrame:
    """Class representing a single frame from the thermal camera."""
    
    def __init__(self, raw_frame, roi=None):
        """
        Initialize a ThermalFrame.

        :param raw_frame: Raw frame data from the video device.
        :param roi: Optional (x, y, w, h) tuple defining the Area of Interest for statistics.
        """
        self.raw_frame = raw_frame
        self.roi = roi # (x, y, w, h)
        
        # TC001 resolution
        TARGET_WIDTH = 256
        TARGET_HEIGHT = 384
        
        # Handle different input formats (especially on Windows)
        if len(raw_frame.shape) == 2 and raw_frame.shape[0] == 1:
            # 1D buffer (common with MSMF and CONVERT_RGB=0)
            if raw_frame.shape[1] == TARGET_WIDTH * TARGET_HEIGHT * 2:
                # 16-bit raw data
                raw_frame = raw_frame.reshape((TARGET_HEIGHT, TARGET_WIDTH, 2))
            elif raw_frame.shape[1] == TARGET_WIDTH * TARGET_HEIGHT * 3:
                # 24-bit raw/BGR data
                raw_frame = raw_frame.reshape((TARGET_HEIGHT, TARGET_WIDTH, 3))
        
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
        
        # Determine analysis region
        if self.roi:
            x, y, w, h = self.roi
            # Ensure ROI is within bounds
            x1 = max(0, min(x, self.width - 1))
            y1 = max(0, min(y, self.height - 1))
            x2 = max(0, min(x + w, self.width))
            y2 = max(0, min(y + h, self.height))
            
            roi_temps = self.raw_temps[y1:y2, x1:x2]
            
            # Center temperature (relative to ROI center)
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            self.center_temp = self._raw_to_celsius(self.raw_temps[cy, cx])
            
            if roi_temps.size > 0:
                # Max temperature in ROI
                max_idx = np.argmax(roi_temps)
                m_roi_col, m_roi_row = divmod(max_idx, roi_temps.shape[1])
                m_col, m_row = y1 + m_roi_col, x1 + m_roi_row
                self.max_temp = self._raw_to_celsius(self.raw_temps[m_col, m_row])
                self.max_pos = (m_row, m_col)
                
                # Min temperature in ROI
                min_idx = np.argmin(roi_temps)
                l_roi_col, l_roi_row = divmod(min_idx, roi_temps.shape[1])
                l_col, l_row = y1 + l_roi_col, x1 + l_roi_row
                self.min_temp = self._raw_to_celsius(self.raw_temps[l_col, l_row])
                self.min_pos = (l_row, l_col)
                
                # Average temperature in ROI
                self.avg_temp = self._raw_to_celsius(np.mean(roi_temps))
            else:
                # Fallback if ROI is invalid
                self.center_temp = 0
                self.max_temp = 0
                self.max_pos = (0, 0)
                self.min_temp = 0
                self.min_pos = (0, 0)
                self.avg_temp = 0
        else:
            # Full frame stats
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
        """
        Generate a colorized heatmap from the image data.

        :param colormap: OpenCV colormap to apply.
        :param alpha: Contrast/Brightness adjustment (default 1.0).
        :param scale: Zoom factor for the display (default 3x).
        :param blur: Gaussian blur factor (default 0).
        :return: Colorized BGR image.
        """
        # Convert YUYV to BGR (OpenCV format)
        if len(self.imdata.shape) == 3 and self.imdata.shape[2] == 3:
            # Already 3 channels (e.g. backend failed to provide raw)
            bgr = self.imdata
        else:
            try:
                bgr = cv2.cvtColor(self.imdata, cv2.COLOR_YUV2BGR_YUYV)
            except cv2.error:
                # Fallback if cvtColor fails
                bgr = cv2.cvtColor(self.imdata, cv2.COLOR_GRAY2BGR) if len(self.imdata.shape) == 2 else self.imdata

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
        Scan available video devices to find a potential TC001 camera.
        Returns a list of tuples (device_id, backend).
        """
        matches = []
        
        # Determine the search range and potential backends
        if os.name == 'nt':
            # Windows: Try MSMF first for better raw data support, then DSHOW, then ANY
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
            search_range = range(10)
        else:
            # Linux/POSIX: Try V4L2
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            search_range = range(16)

        for i in search_range:
            for backend in backends:
                cap = cv2.VideoCapture(i, backend)
                if not cap.isOpened():
                    continue
                
                # Try setting the expected resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
                
                # Crucial: Pull in video but do NOT automatically convert to RGB
                # This is essential to get the raw thermal data
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
                
                # Try a couple of frames to be sure
                detected = False
                for _ in range(5):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    
                    h, w = frame.shape[:2]
                    # TC001 is 256x384 (combined image and thermal)
                    # Check for standard 2D shape or flat buffer from some backends (like MSMF)
                    is_tc001 = (h == 384 and w == 256) or (h == 256 and w == 384)
                    if not is_tc001 and h == 1:
                        # 256 * 384 * 2 = 196608 (16-bit raw)
                        # 256 * 384 * 3 = 294912 (24-bit raw/BGR)
                        is_tc001 = (w == 196608 or w == 294912)

                    if is_tc001:
                        matches.append((i, backend))
                        detected = True
                        break
                
                cap.release()
                if detected:
                    break # Found it with this backend, no need to try others for this index
                    
        return matches

    def __init__(self, device_id=None, include_preview=False):
        """
        Initialize the Thermal Camera.

        :param device_id: Video device index. If None, the library will attempt auto-detection.
        :param include_preview: If True, starts an interactive live preview in a background thread.
        """
        backend = None
        if device_id is None:
            print("No device ID provided. Attempting to auto-detect Thermal Camera...")
            matches = self.detect_devices()
            if matches:
                device_id, backend = matches[0]
                print(f"Auto-detected Thermal Camera on device {device_id}")
            else:
                device_id = 0
                print("Could not auto-detect. Falling back to device 0.")

        self.device_id = device_id
        
        # Determine preferred backend if not already found during detection
        if backend is None:
            backend = cv2.CAP_ANY
            if os.name == 'nt':
                # Prefer MSMF on Windows for better raw support
                backend = cv2.CAP_MSMF
            elif os.name == 'posix':
                backend = cv2.CAP_V4L2
            
        self.cap = cv2.VideoCapture(device_id, backend)
            
        if not self.cap.isOpened():
            # Fallback to default backend if specified one failed
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                print(f"Warning: Could not open video device {device_id}")
            
        # Set expected resolution for TC001
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
            
        # Crucial: Pull in video but do NOT automatically convert to RGB
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        # Default display settings
        self.colormap = cv2.COLORMAP_JET
        self.alpha = 1.0
        self.scale = 3
        self.blur = 0
        self.threshold = 2
        self.hud = True
        self.include_markers = False
        self.roi = None

        self._preview_thread = None
        self._stop_preview = threading.Event()
        
        if include_preview:
            self._preview_thread = threading.Thread(target=self.live_preview, daemon=True)
            self._preview_thread.start()

    def get_frame(self, roi=None):
        """
        Capture a single frame and return a ThermalFrame object.

        :param roi: Optional (x, y, w, h) tuple to override the default ROI for this frame.
        :return: ThermalFrame object or None if capture fails.
        """
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Use provided ROI or fallback to instance ROI
        target_roi = roi if roi is not None else self.roi
        return ThermalFrame(frame, roi=target_roi)

    def capture(self, filename_prefix="Capture", folder=".", colormap=None,
                alpha=None, scale=None, blur=None, include_markers=None, frame=None):
        """
        Take a snapshot and store both the image and temperature metadata.

        :param filename_prefix: Prefix for the generated filenames.
        :param folder: Directory where files should be saved.
        :param colormap: Colormap to use (overrides instance default).
        :param alpha: Contrast factor (overrides instance default).
        :param scale: Image scale (overrides instance default).
        :param blur: Blur factor (overrides instance default).
        :param include_markers: Whether to overlay hotspots/coldspots on the saved image.
        :param frame: Optional ThermalFrame to use instead of capturing a new one.
        :return: Dictionary containing 'image', 'metadata_file', and 'metadata' dict, or None.
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
        include_markers = include_markers if include_markers is not None else self.include_markers

        now = time.strftime("%Y%m%d-%H%M%S")
        img_filename = os.path.join(folder, f"{filename_prefix}_{now}.png")
        meta_filename = os.path.join(folder, f"{filename_prefix}_{now}.json")
        
        heatmap = frame.get_heatmap(colormap=colormap, alpha=alpha, scale=scale, blur=blur)
        
        if include_markers:
            self._draw_markers(heatmap, frame, scale)

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
            "roi": frame.roi,
            "settings": {
                "colormap": str(colormap),
                "alpha": alpha,
                "scale": scale,
                "blur": blur,
                "include_markers": include_markers,
                "threshold": self.threshold
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
        """
        Start a live preview window with interactive controls.

        Note: If `include_preview=True` was passed to `__init__`, this is already running 
        in a background thread.

        :param colormap: Initial colormap.
        :param alpha: Initial contrast factor.
        :param scale: Initial display scale.
        :param blur: Initial blur factor.
        :param threshold: Initial marker sensitivity threshold.
        :param hud: Boolean to toggle the on-screen display (HUD).
        """
        cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
        
        # Mouse callback for ROI selection
        selection = {"start": None, "current": None, "selecting": False}
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selection["start"] = (x, y)
                selection["current"] = (x, y)
                selection["selecting"] = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if selection["selecting"]:
                    selection["current"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if selection["selecting"]:
                    x1, y1 = selection["start"]
                    x2, y2 = x, y
                    # Normalize coordinates and scale back to thermal frame size
                    ix1, ix2 = min(x1, x2) // self.scale, max(x1, x2) // self.scale
                    iy1, iy2 = min(y1, y2) // self.scale, max(y1, y2) // self.scale
                    
                    if ix2 - ix1 > 2 and iy2 - iy1 > 2:
                        self.roi = (ix1, iy1, ix2 - ix1, iy2 - iy1)
                    selection["selecting"] = False
                    selection["start"] = None

        cv2.setMouseCallback('Thermal', mouse_callback)

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
        print("  k   : Toggle Markers in Snapshot")
        print("  r   : Clear ROI")
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
            self._draw_markers(heatmap, frame, self.scale)

            # Draw active ROI box
            if self.roi:
                rx, ry, rw, rh = self.roi
                cv2.rectangle(heatmap, (rx*self.scale, ry*self.scale), 
                             ((rx+rw)*self.scale, (ry+rh)*self.scale), (255, 255, 255), 2)
            
            # Draw ongoing selection
            if selection["selecting"] and selection["start"] and selection["current"]:
                cv2.rectangle(heatmap, selection["start"], selection["current"], (0, 255, 0), 1)

            cv2.imshow('Thermal', heatmap)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                res = self.capture(frame=frame, filename_prefix="Preview")
                if res:
                    print(f"Captured: {res['image']} && {res['metadata_file']}")
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
            elif key == ord('r'):
                self.roi = None
            elif key == ord('k'):
                self.include_markers = not self.include_markers
                print(f"Markers in captures: {'Enabled' if self.include_markers else 'Disabled'}")
                
        cv2.destroyAllWindows()
        self._stop_preview.set()

    def _draw_markers(self, heatmap, frame, scale):
        """Draw hotspot and coldspot markers on the given heatmap."""
        if frame.max_temp > frame.avg_temp + self.threshold:
            mx, my = frame.max_pos
            cv2.circle(heatmap, (mx*scale, my*scale), 5, (0,0,255), -1)
            cv2.putText(heatmap, f"{frame.max_temp} C", (mx*scale+10, my*scale+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

        if frame.min_temp < frame.avg_temp - self.threshold:
            lx, ly = frame.min_pos
            cv2.circle(heatmap, (lx*scale, ly*scale), 5, (255,0,0), -1)
            cv2.putText(heatmap, f"{frame.min_temp} C", (lx*scale+10, ly*scale+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

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
