"""
Configuration settings for OASIS application
"""

import os
from typing import Dict, List

# Application Settings
APP_TITLE = "OASIS - Open Accessibility Spatial Impact System"
APP_DESCRIPTION = "Discover the hidden impact of parks on urban connectivity"

# Default Map Settings  
DEFAULT_LOCATION = (51.9225, 4.47917)  # Rotterdam
DEFAULT_ZOOM = 12
SEARCH_RADIUS = 2000  # meters

# OSM Network Settings
NETWORK_TYPE = 'walk'
WALKING_SPEEDS = {
    'slow': 3.0,      # km/h
    'normal': 4.5,    # km/h  
    'fast': 6.0       # km/h
}

# Analysis Parameters
MAX_WALKING_TIMES = [5, 10, 15, 20, 30]  # minutes
ANALYSIS_DEPTHS = {
    'quick': {'sample_size': 100, 'centrality_k': 100},
    'standard': {'sample_size': 500, 'centrality_k': 500}, 
    'deep': {'sample_size': 1000, 'centrality_k': 1000}
}

# Park/Green Space Tags
PARK_TAGS = {
    "leisure": ["park", "recreation_ground", "playground", "garden"],
    "landuse": ["grass", "recreation_ground", "village_green"],
    "natural": ["grassland"]
}

# Visualization Settings
COLORS = {
    'remaining_edges': '#2E86AB',    # Blue
    'removed_edges': '#A23B72',      # Red/Pink
    'parks': '#31B870',              # Green
    'critical_edges': '#F18F01',     # Orange
    'background': '#F8F9FA'          # Light gray
}

MAP_STYLES = {
    'default': 'OpenStreetMap',
    'satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'terrain': 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'
}

# Cache Settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = "cache"

# Export Settings
EXPORT_FORMATS = ['PDF', 'HTML', 'PNG', 'SVG']
REPORT_TEMPLATES = ['executive', 'technical', 'policy']

# Performance Settings
MAX_EDGES_FOR_FULL_ANALYSIS = 50000
PARALLEL_PROCESSING = True
MAX_WORKERS = 4

# API Settings (for future microservices)
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 300  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"