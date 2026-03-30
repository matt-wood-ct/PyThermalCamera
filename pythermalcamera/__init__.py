"""
Topdon TC001 Thermal Camera Library.

This library provides a high-level interface for the Topdon TC001 thermal camera,
including auto-detection, live preview, and temperature analysis.

Based on work by Les Wright (https://github.com/leswright1977/PyThermalCamera)
and downstream researcher LeoDJ, who reverse engineered the thermal image 
format to extract raw temperature data.
See LeoDJ's work here: https://github.com/LeoDJ/P2Pro-Viewer
"""
__version__ = "0.1.0"

from .camera import ThermalCamera, ThermalFrame
