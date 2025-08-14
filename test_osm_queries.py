#!/usr/bin/env python3
"""Test script to verify OSM queries are working"""

import osmnx as ox

# Test location (should be somewhere with amenities)
lat, lng = 40.7831, -73.9712  # Central Park area, NYC - lots of amenities nearby
radius = 1000  # 1km radius

print(f"Testing OSM queries at ({lat}, {lng}) with {radius}m radius")

# Test supermarkets
print("\n=== SUPERMARKETS ===")
try:
    supermarkets_gdf = ox.features_from_point(
        (lat, lng), 
        dist=radius,
        tags={'shop': ['supermarket', 'grocery', 'convenience', 'department_store', 'general', 'food'], 'amenity': ['marketplace', 'food_court']}
    )
    print(f"Found {len(supermarkets_gdf)} supermarkets")
    if not supermarkets_gdf.empty:
        print("Sample supermarkets:")
        for idx, row in supermarkets_gdf.head(3).iterrows():
            name = row.get('name', 'Unnamed')
            print(f"  - {name} at ({row.geometry.centroid.y:.6f}, {row.geometry.centroid.x:.6f})")
except Exception as e:
    print(f"Supermarkets query failed: {e}")

# Test schools  
print("\n=== SCHOOLS ===")
try:
    schools_gdf = ox.features_from_point((lat, lng), dist=radius, tags={'amenity': 'school'})
    print(f"Found {len(schools_gdf)} schools")
    if not schools_gdf.empty:
        print("Sample schools:")
        for idx, row in schools_gdf.head(3).iterrows():
            name = row.get('name', 'Unnamed')
            print(f"  - {name} at ({row.geometry.centroid.y:.6f}, {row.geometry.centroid.x:.6f})")
except Exception as e:
    print(f"Schools query failed: {e}")

# Test playgrounds
print("\n=== PLAYGROUNDS ===")
try:
    playgrounds_gdf = ox.features_from_point((lat, lng), dist=radius, tags={'leisure': 'playground'})
    print(f"Found {len(playgrounds_gdf)} playgrounds")
    if not playgrounds_gdf.empty:
        print("Sample playgrounds:")
        for idx, row in playgrounds_gdf.head(3).iterrows():
            name = row.get('name', 'Unnamed')
            print(f"  - {name} at ({row.geometry.centroid.y:.6f}, {row.geometry.centroid.x:.6f})")
except Exception as e:
    print(f"Playgrounds query failed: {e}")

# Test cafes/bars
print("\n=== CAFES/BARS ===")
try:
    cafes_bars_gdf = ox.features_from_point(
        (lat, lng), 
        dist=radius, 
        tags={'amenity': ['cafe', 'bar', 'pub', 'restaurant'], 'shop': ['coffee']}
    )
    print(f"Found {len(cafes_bars_gdf)} cafes/bars")
    if not cafes_bars_gdf.empty:
        print("Sample cafes/bars:")
        for idx, row in cafes_bars_gdf.head(3).iterrows():
            name = row.get('name', 'Unnamed')
            print(f"  - {name} at ({row.geometry.centroid.y:.6f}, {row.geometry.centroid.x:.6f})")
except Exception as e:
    print(f"Cafes/bars query failed: {e}")

# Test transit
print("\n=== TRANSIT ===")
try:
    transit_gdf = ox.features_from_point(
        (lat, lng), 
        dist=radius, 
        tags={'public_transport': ['station', 'stop_position', 'platform'], 'railway': ['station', 'halt', 'tram_stop'], 'highway': 'bus_stop'}
    )
    print(f"Found {len(transit_gdf)} transit stops")
    if not transit_gdf.empty:
        print("Sample transit stops:")
        for idx, row in transit_gdf.head(3).iterrows():
            name = row.get('name', 'Unnamed')
            print(f"  - {name} at ({row.geometry.centroid.y:.6f}, {row.geometry.centroid.x:.6f})")
except Exception as e:
    print(f"Transit query failed: {e}")

print("\nTest complete!")