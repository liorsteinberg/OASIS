#!/usr/bin/env python3
"""Debug script to test the exact amenity data flow"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
import numpy as np

# Use a location that should have amenities (let's use a typical city location)
lat, lng = 40.7831, -73.9712  # Central Park area, NYC
radius = 1000

print(f"Testing at location: ({lat}, {lng})")

# Load amenities exactly like the app does
print("\n=== LOADING AMENITIES ===")

# Supermarkets
try:
    supermarkets_gdf = ox.features_from_point(
        (lat, lng), 
        dist=radius,
        tags={'shop': ['supermarket', 'grocery', 'convenience', 'department_store', 'general', 'food'], 'amenity': ['marketplace', 'food_court']}
    )
    if not supermarkets_gdf.empty:
        supermarkets_gdf = supermarkets_gdf.to_crs('EPSG:4326')
    print(f"Loaded {len(supermarkets_gdf) if not supermarkets_gdf.empty else 0} supermarkets (CRS: {supermarkets_gdf.crs if not supermarkets_gdf.empty else 'N/A'})")
except Exception as e:
    print(f"Failed to load supermarkets: {e}")
    supermarkets_gdf = gpd.GeoDataFrame()

# Test the conditional logic exactly like the app
print(f"\nTesting conditional logic:")
print(f"supermarkets_gdf is not None: {supermarkets_gdf is not None}")
print(f"supermarkets_gdf.empty: {supermarkets_gdf.empty}")
print(f"Condition result: {supermarkets_gdf is not None and not supermarkets_gdf.empty}")

# This is what gets passed to the filtering function
passed_data = supermarkets_gdf if supermarkets_gdf is not None and not supermarkets_gdf.empty else None
print(f"Data passed to filtering: {type(passed_data)} (None: {passed_data is None})")

if passed_data is not None:
    print(f"Passed data length: {len(passed_data)}")
    print(f"Sample entry: {passed_data.iloc[0].get('name', 'Unnamed')} at ({passed_data.iloc[0].geometry.centroid.y:.6f}, {passed_data.iloc[0].geometry.centroid.x:.6f})")

# Test the filtering logic with a simple polygon
print(f"\n=== TESTING SPATIAL FILTERING ===")
if passed_data is not None and not passed_data.empty:
    # Create a test polygon around the center point
    # Simple square around the center
    buffer_size = 0.01  # ~1km in degrees
    test_polygon = Polygon([
        (lng - buffer_size, lat - buffer_size),
        (lng + buffer_size, lat - buffer_size),
        (lng + buffer_size, lat + buffer_size),
        (lng - buffer_size, lat + buffer_size)
    ])
    
    count = 0
    for idx, amenity in passed_data.iterrows():
        try:
            if hasattr(amenity.geometry, 'centroid'):
                point = amenity.geometry.centroid
            else:
                point = amenity.geometry
            
            amenity_point = Point(point.x, point.y)
            if test_polygon.contains(amenity_point):
                count += 1
                
        except Exception as e:
            print(f"Error processing amenity: {e}")
            continue
    
    print(f"Amenities within test polygon: {count} out of {len(passed_data)}")
    print(f"Test polygon bounds: {test_polygon.bounds}")
    if len(passed_data) > 0:
        first_amenity = passed_data.iloc[0]
        first_point = first_amenity.geometry.centroid if hasattr(first_amenity.geometry, 'centroid') else first_amenity.geometry
        print(f"First amenity location: ({first_point.y:.6f}, {first_point.x:.6f})")
        print(f"First amenity in polygon: {test_polygon.contains(Point(first_point.x, first_point.y))}")
        
else:
    print("No data to test spatial filtering with")

print("\nDone!")