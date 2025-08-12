from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional, Tuple
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull

app = FastAPI(title="Fast Park Accessibility", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Global cache for loaded data
DATA_CACHE = {}

class ParkAccessibilityAnalyzer:
    def __init__(self):
        self.cache = {}
    
    async def load_area_data(self, lat: float, lng: float, radius: int = 2000) -> Dict:
        """Load parks and street network for an area"""
        cache_key = f"{lat:.4f}_{lng:.4f}_{radius}"
        
        if cache_key in DATA_CACHE:
            return DATA_CACHE[cache_key]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, self._load_data_sync, lat, lng, radius)
        
        DATA_CACHE[cache_key] = data
        return data
    
    def _load_data_sync(self, lat: float, lng: float, radius: int) -> Dict:
        """Synchronous data loading"""
        print(f"Starting data load for {lat}, {lng}, {radius}")
        
        try:
            # Load parks
            print("Loading parks...")
            tags = {"leisure": ["park", "recreation_ground"], "landuse": ["grass", "recreation_ground"]}
            parks = ox.features_from_point((lat, lng), tags=tags, dist=radius)
            print(f"Raw parks loaded: {len(parks)}")
            
            parks = parks[parks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            print(f"Filtered parks: {len(parks)}")
            
            if not parks.empty:
                parks["area_m2"] = parks.geometry.to_crs(3857).area
                # parks = parks[parks.area_m2 > 500].head(50)  # Limit for performance
                print(f"Final parks: {len(parks)}")
            
            # Load street network
            print("Loading street network...")
            graph = ox.graph_from_point((lat, lng), dist=radius, network_type='walk')
            print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Add edge lengths using current OSMnx API
            print("Adding edge lengths...")
            graph = ox.distance.add_edge_lengths(graph)
            print("Edge lengths added")
                
            print("Converting to undirected...")
            graph_undirected = graph.to_undirected()
            print("Converted to undirected")
            
            print("Converting to GeoDataFrames...")
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
            print(f"GDFs created: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
            
            # Convert to serializable format
            print("Converting parks to serializable format...")
            parks_data = []
            for idx, park in parks.iterrows():
                try:
                    if park.geometry.geom_type == 'Polygon':
                        coords = list(park.geometry.exterior.coords)
                        centroid = park.geometry.centroid
                        area_m2 = float(park.area_m2)
                        
                        # Check for invalid values
                        if not (np.isfinite(centroid.y) and np.isfinite(centroid.x) and np.isfinite(area_m2)):
                            print(f"Skipping park {idx} due to invalid coordinates or area")
                            continue
                            
                        parks_data.append({
                            'id': str(idx),
                            'name': str(park.get('name', 'Unnamed Park')),
                            'coords': [[float(lat_coord), float(lon_coord)] for lon_coord, lat_coord in coords 
                                      if np.isfinite(lat_coord) and np.isfinite(lon_coord)],
                            'centroid': [float(centroid.y), float(centroid.x)],
                            'area_m2': area_m2
                        })
                except Exception as e:
                    print(f"Error processing park {idx}: {e}")
                    continue
            
            print("Converting nodes to serializable format...")
            nodes_data = []
            for node_id, node_data in nodes_gdf.iterrows():
                try:
                    lat = float(node_data.geometry.y)
                    lng = float(node_data.geometry.x)
                    
                    # Check for invalid coordinates
                    if not (np.isfinite(lat) and np.isfinite(lng)):
                        continue
                        
                    nodes_data.append({
                        'id': str(node_id),
                        'lat': lat,
                        'lng': lng
                    })
                except Exception as e:
                    print(f"Error processing node {node_id}: {e}")
                    continue
            
            print(f"Data preparation complete: {len(parks_data)} parks, {len(nodes_data)} nodes")
            
            return {
                'parks': parks_data,
                # 'nodes': nodes_data[:200],  # Limit nodes for UI performance
                'nodes': nodes_data,  # Limit nodes for UI performance
                'graph': graph_undirected,
                'parks_gdf': parks,
                'nodes_gdf': nodes_gdf,
                'edges_gdf': edges_gdf,
                'center': [lat, lng]
            }
            
        except Exception as e:
            print(f"Exception in _load_data_sync: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load area data: {str(e)}")
    
    async def calculate_accessibility(self, lat: float, lng: float, radius: int, 
                                   park_id: str, node_id: str, walk_time: float = 10.0, 
                                   walk_speed: float = 4.5, viz_method: str = "convex_hull") -> Dict:
        """Calculate accessibility with and without park"""
        # Get cached data
        cache_key = f"{lat:.4f}_{lng:.4f}_{radius}"
        if cache_key not in DATA_CACHE:
            await self.load_area_data(lat, lng, radius)
        
        data = DATA_CACHE[cache_key]
        
        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, self._calculate_accessibility_sync, 
            data, park_id, node_id, walk_time, walk_speed, viz_method
        )
        
        return result
    
    def _calculate_accessibility_sync(self, data: Dict, park_id: str, node_id: str, 
                                     walk_time: float, walk_speed: float, viz_method: str) -> Dict:
        """Synchronous accessibility calculation"""
        print(f"Starting accessibility calculation for park {park_id}, node {node_id}")
        
        try:
            graph = data['graph']
            parks_gdf = data['parks_gdf']
            nodes_gdf = data['nodes_gdf']
            
            # Convert park_id and node_id to proper types
            try:
                # Handle different park ID formats
                if park_id.isdigit():
                    park_key = int(park_id)
                else:
                    park_key = park_id
                    
                # Try to find the park in different ways
                if park_key not in parks_gdf.index:
                    # Try converting park_key to different types
                    for idx in parks_gdf.index:
                        if str(idx) == str(park_key):
                            park_key = idx
                            break
                    else:
                        raise KeyError(f"Park {park_id} not found in parks_gdf")
                        
                # Similar handling for node_id
                node_key = int(node_id)
                if node_key not in nodes_gdf.index:
                    raise KeyError(f"Node {node_id} not found in nodes_gdf")
                    
            except (ValueError, KeyError) as e:
                raise Exception(f"Invalid park_id ({park_id}) or node_id ({node_id}): {e}")
            
            # Check if node exists in graph
            if node_key not in graph:
                raise Exception(f"Node {node_key} not found in graph")
            
            # Walking parameters
            walking_speed_ms = (walk_speed * 1000) / 60  # km/h to m/min
            max_distance = walk_time * walking_speed_ms  # walking time in minutes
            
            print(f"Calculating WITH park (park_key: {park_key}, node_key: {node_key})...")
            # Get park geometry for enhanced analysis
            park_geom = parks_gdf.loc[park_key].geometry
            
            # Calculate WITH park
            with_park = self._calculate_isochrone(graph, nodes_gdf, data['edges_gdf'], node_key, max_distance, viz_method)
            
            print("Calculating WITHOUT park...")
            # Calculate WITHOUT park
            filtered_graph = self._remove_park_edges(graph, nodes_gdf, park_geom)
            without_park = self._calculate_isochrone(filtered_graph, nodes_gdf, data['edges_gdf'], node_key, max_distance, viz_method)
            
            print("Accessibility calculation complete")
            
            # Calculate enhanced differences and metrics
            area_difference = without_park['area_km2'] - with_park['area_km2']
            street_network_difference = without_park['street_network_stats']['total_length_km'] - with_park['street_network_stats']['total_length_km']
            node_difference = without_park['reachable_nodes'] - with_park['reachable_nodes']
            edge_difference = without_park['street_network_stats']['edge_count'] - with_park['street_network_stats']['edge_count']
            
            # Calculate percentage changes
            area_change_pct = (area_difference / with_park['area_km2'] * 100) if with_park['area_km2'] > 0 else 0
            street_network_change_pct = (street_network_difference / with_park['street_network_stats']['total_length_km'] * 100) if with_park['street_network_stats']['total_length_km'] > 0 else 0
            
            # Determine impact category (5 levels) based on street network change
            if street_network_change_pct < -5:
                impact_category = 'highly_positive'
                impact_label = 'Highly Positive'
            elif street_network_change_pct < -1:
                impact_category = 'positive'
                impact_label = 'Positive'
            elif street_network_change_pct > 5:
                impact_category = 'highly_negative'
                impact_label = 'Highly Negative'
            elif street_network_change_pct > 1:
                impact_category = 'negative'
                impact_label = 'Negative'
            else:
                impact_category = 'neutral'
                impact_label = 'Neutral'
            
            return {
                'with_park': with_park,
                'without_park': without_park,
                'difference_km2': area_difference,
                'street_network_difference_km': street_network_difference,
                'node_difference': node_difference,
                'edge_difference': edge_difference,
                'area_change_pct': area_change_pct,
                'street_network_change_pct': street_network_change_pct,
                'impact_category': impact_category,
                'impact_label': impact_label,
                'node_id': node_id,
                'park_id': park_id,
                'enhanced_metrics': {
                    'with_connectivity': with_park['connectivity_stats']['connectivity_ratio'],
                    'without_connectivity': without_park['connectivity_stats']['connectivity_ratio'],
                    'connectivity_change': without_park['connectivity_stats']['connectivity_ratio'] - with_park['connectivity_stats']['connectivity_ratio'],
                    'with_avg_distance': with_park['distance_stats']['mean_distance'],
                    'without_avg_distance': without_park['distance_stats']['mean_distance'],
                    'with_street_length_km': with_park['street_network_stats']['total_length_km'],
                    'without_street_length_km': without_park['street_network_stats']['total_length_km'],
                    'with_street_density': with_park['street_network_stats']['street_density_km_per_km2'],
                    'without_street_density': without_park['street_network_stats']['street_density_km_per_km2']
                }
            }
            
        except Exception as e:
            print(f"Exception in _calculate_accessibility_sync: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to calculate accessibility: {str(e)}")
    
    def _calculate_isochrone(self, graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                           edges_gdf: gpd.GeoDataFrame, start_node: int, max_distance: float, 
                           viz_method: str = "convex_hull", park_geometry=None) -> Dict:
        """Calculate isochrone for a node with enhanced metrics"""
        if start_node not in graph:
            print(f"Start node {start_node} not found in graph")
            return self._empty_isochrone_result()
        
        try:
            print(f"Finding reachable nodes from {start_node} within {max_distance}m")
            
            # Find reachable nodes using Dijkstra with error handling
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    graph, start_node, weight='length', cutoff=max_distance
                )
            except KeyError as e:
                print(f"KeyError during Dijkstra: {e}")
                return self._empty_isochrone_result()
            
            reachable_nodes = list(lengths.keys())
            distances = list(lengths.values())
            print(f"Found {len(reachable_nodes)} reachable nodes")
            
            if len(reachable_nodes) < 4:
                return self._empty_isochrone_result(len(reachable_nodes))
            
            # Calculate distance statistics
            distance_stats = {
                'min_distance': min(distances),
                'max_distance': max(distances),
                'mean_distance': np.mean(distances),
                'median_distance': np.median(distances),
                'std_distance': np.std(distances)
            }
            
            # Analyze connectivity
            reachable_subgraph = graph.subgraph(reachable_nodes)
            connected_components = list(nx.connected_components(reachable_subgraph))
            connectivity_stats = {
                'connected_components': len(connected_components),
                'largest_component_size': max(len(comp) for comp in connected_components) if connected_components else 0,
                'connectivity_ratio': len(connected_components[0]) / len(reachable_nodes) if connected_components else 0
            }
            
            # Get coordinates of reachable nodes
            points = []
            node_coords = {}
            for node in reachable_nodes:
                if node in nodes_gdf.index:
                    try:
                        geom = nodes_gdf.loc[node].geometry
                        lat, lng = float(geom.y), float(geom.x)
                        if np.isfinite(lat) and np.isfinite(lng):
                            points.append([lat, lng])
                            node_coords[node] = (lat, lng)
                    except Exception as e:
                        print(f"Error processing node {node}: {e}")
                        continue
            
            print(f"Got coordinates for {len(points)} nodes")
            
            if len(points) < 4:
                return self._empty_isochrone_result(len(reachable_nodes))
            
            # Choose visualization method and calculate area
            if viz_method == "buffered_network":
                area_result = self._create_buffered_network_isochrone(graph, edges_gdf, reachable_nodes, max_distance)
            else:
                points_array = np.array(points)
                area_result = self._create_convex_hull_isochrone(points_array, reachable_nodes)
            
            # Calculate reachable street network metrics
            street_network_stats = self._calculate_street_network_stats(graph, edges_gdf, reachable_nodes, area_result['area_km2'])
            
            # Compile enhanced result
            result = {
                **area_result,
                'distance_stats': distance_stats,
                'connectivity_stats': connectivity_stats,
                'street_network_stats': street_network_stats,
                'node_coordinates': node_coords,
                'reachable_distances': dict(zip(reachable_nodes, distances))
            }
            
            return result
                
        except Exception as e:
            print(f"Error in _calculate_isochrone: {e}")
            return self._empty_isochrone_result()
    
    def _empty_isochrone_result(self, reachable_nodes=0):
        """Return empty result with all metrics initialized"""
        return {
            'boundary': [], 
            'area_km2': 0, 
            'reachable_nodes': reachable_nodes,
            'distance_stats': {
                'min_distance': 0, 'max_distance': 0, 'mean_distance': 0, 
                'median_distance': 0, 'std_distance': 0
            },
            'connectivity_stats': {
                'connected_components': 0, 'largest_component_size': 0, 'connectivity_ratio': 0
            },
            'street_network_stats': {
                'total_length_km': 0, 'edge_count': 0, 'street_density_km_per_km2': 0, 'avg_edge_length_m': 0
            },
            'node_coordinates': {},
            'reachable_distances': {}
        }
    
    def _calculate_street_network_stats(self, graph: nx.Graph, edges_gdf: gpd.GeoDataFrame, 
                                       reachable_nodes: list, accessible_area_km2: float) -> Dict:
        """Calculate street network statistics for reachable nodes"""
        try:
            reachable_node_set = set(reachable_nodes)
            total_length_m = 0
            edge_count = 0
            edge_lengths = []
            
            # Get all edges where both endpoints are reachable
            for edge_idx, edge_data in edges_gdf.iterrows():
                try:
                    u, v, k = edge_idx
                    if u in reachable_node_set and v in reachable_node_set:
                        # Get edge length from graph data (already calculated in meters)
                        if graph.has_edge(u, v, k):
                            edge_length = graph[u][v][k].get('length', 0)
                            if edge_length > 0:
                                total_length_m += edge_length
                                edge_count += 1
                                edge_lengths.append(edge_length)
                except Exception:
                    continue
            
            # Convert to kilometers
            total_length_km = total_length_m / 1000
            
            # Calculate derived metrics
            street_density_km_per_km2 = total_length_km / accessible_area_km2 if accessible_area_km2 > 0 else 0
            avg_edge_length_m = np.mean(edge_lengths) if edge_lengths else 0
            
            return {
                'total_length_km': total_length_km,
                'edge_count': edge_count,
                'street_density_km_per_km2': street_density_km_per_km2,
                'avg_edge_length_m': avg_edge_length_m
            }
            
        except Exception as e:
            print(f"Error calculating street network stats: {e}")
            return {
                'total_length_km': 0, 'edge_count': 0, 'street_density_km_per_km2': 0, 'avg_edge_length_m': 0
            }
    
    def _create_convex_hull_isochrone(self, points_array: np.ndarray, reachable_nodes: list) -> Dict:
        """Create convex hull isochrone (fast but can overlap parks)"""
        print(f"Creating convex hull with {len(points_array)} points")
        print(f"Points array shape: {points_array.shape}")
        print(f"First few points: {points_array[:3].tolist() if len(points_array) > 3 else points_array.tolist()}")
        
        try:
            hull = ConvexHull(points_array)
            print(f"ConvexHull created successfully with {len(hull.vertices)} vertices")
            
            # Get boundary coordinates
            boundary = []
            for vertex in hull.vertices:
                boundary.append(points_array[vertex].tolist())
            boundary.append(boundary[0])  # Close polygon
            
            # Calculate area (rough approximation)
            # hull.volume is the area in square degrees
            # 1 degree latitude ≈ 111 km, but longitude varies by latitude
            # For Rotterdam (≈52°N), 1 degree longitude ≈ 69 km
            area_deg_sq = hull.volume
            area_km2 = area_deg_sq * 111 * 69  # More accurate for Rotterdam latitude
            
            print(f"Convex hull area: {area_deg_sq:.9f} deg² → {area_km2:.6f} km²")
            print(f"Convex hull boundary points: {len(boundary)}")
            print(f"Final boundary: {boundary}")
            
            return {
                'boundary': boundary,
                'area_km2': float(area_km2) if np.isfinite(area_km2) else 0,
                'reachable_nodes': len(reachable_nodes)
            }
        except Exception as e:
            print(f"Error creating convex hull: {e}")
            import traceback
            traceback.print_exc()
            return {'boundary': [], 'area_km2': 0, 'reachable_nodes': len(reachable_nodes)}
    
    def _create_buffered_network_isochrone(self, graph: nx.Graph, edges_gdf: gpd.GeoDataFrame, 
                                         reachable_nodes: list, max_distance: float) -> Dict:
        """Create buffered network isochrone (realistic, follows streets)"""
        try:
            print(f"Creating buffered network isochrone for {len(reachable_nodes)} reachable nodes")
            
            # Get edges where both endpoints are reachable
            reachable_node_set = set(reachable_nodes)
            reachable_edges = []
            
            for edge_idx, edge_data in edges_gdf.iterrows():
                try:
                    u, v, k = edge_idx
                    if u in reachable_node_set and v in reachable_node_set:
                        reachable_edges.append(edge_data.geometry)
                except Exception:
                    continue
            
            if len(reachable_edges) == 0:
                print("No reachable edges found")
                return {'boundary': [], 'area_km2': 0, 'reachable_nodes': len(reachable_nodes)}
            
            print(f"Found {len(reachable_edges)} reachable edges")
            
            # Create GeoSeries of reachable edges
            edges_series = gpd.GeoSeries(reachable_edges, crs=edges_gdf.crs)
            
            # Project to metric CRS for buffering
            edges_projected = edges_series.to_crs('EPSG:3857')
            
            # Buffer edges by a reasonable amount (e.g., 25m on each side)
            buffer_distance = 25  # meters
            buffered_edges = edges_projected.buffer(buffer_distance, cap_style=2, join_style=2)
            
            # Union all buffered edges
            unified_buffer = buffered_edges.unary_union
            
            if unified_buffer.is_empty:
                return {'boundary': [], 'area_km2': 0, 'reachable_nodes': len(reachable_nodes)}
            
            # Handle MultiPolygon - keep largest part
            if hasattr(unified_buffer, 'geoms'):
                unified_buffer = max(unified_buffer.geoms, key=lambda p: p.area)
            
            # Calculate area in km²
            area_m2 = unified_buffer.area
            area_km2 = area_m2 / 1e6
            
            print(f"Buffered network area: {area_m2:.0f} m² → {area_km2:.6f} km²")
            
            # Convert back to WGS84 for display
            buffer_gdf = gpd.GeoSeries([unified_buffer], crs='EPSG:3857').to_crs('EPSG:4326')
            boundary_geom = buffer_gdf.iloc[0]
            
            # Extract boundary coordinates
            if boundary_geom.geom_type == 'Polygon':
                coords = list(boundary_geom.exterior.coords)
                boundary = [[lat, lon] for lon, lat in coords]
            else:
                boundary = []
            
            return {
                'boundary': boundary,
                'area_km2': float(area_km2) if np.isfinite(area_km2) else 0,
                'reachable_nodes': len(reachable_nodes)
            }
            
        except Exception as e:
            print(f"Error creating buffered network isochrone: {e}")
            import traceback
            traceback.print_exc()
            return {'boundary': [], 'area_km2': 0, 'reachable_nodes': len(reachable_nodes)}
    
    def _remove_park_edges(self, graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                          park_geom) -> nx.Graph:
        """Remove edges that intersect with park"""
        filtered_graph = graph.copy()
        
        try:
            # Get edges as GeoDataFrame
            _, edges_gdf = ox.graph_to_gdfs(graph)
            
            # Project to metric CRS for intersection
            edges_proj = edges_gdf.to_crs('EPSG:3857')
            park_proj = gpd.GeoSeries([park_geom], crs=edges_gdf.crs).to_crs('EPSG:3857').iloc[0]
            
            # Find intersecting edges
            intersects = edges_proj.geometry.intersects(park_proj)
            intersecting_edges = edges_proj[intersects]
            
            # Remove from graph
            for edge_idx in intersecting_edges.index:
                try:
                    u, v, k = edge_idx
                    if filtered_graph.has_edge(u, v, k):
                        filtered_graph.remove_edge(u, v, k)
                except:
                    continue
                    
        except Exception as e:
            pass  # Return original graph if filtering fails
        
        return filtered_graph

# Global analyzer instance
analyzer = ParkAccessibilityAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve the main HTML page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Fast Park Accessibility</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .controls { 
            display: flex; 
            gap: 20px; 
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            font-weight: bold;
            color: #333;
        }
        .control-group input, .control-group select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        #map { 
            height: 500px; 
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
            background-color: #e5e5e5; /* Fallback background */
            z-index: 1;
        }
        .btn {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.loading {
            background-color: #fff3cd;
            color: #856404;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .results {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .step {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            border-radius: 0 8px 8px 0;
        }
        .step h3 {
            margin-top: 0;
            color: #007bff;
        }
        .center-marker {
            background: none;
            border: none;
            font-size: 16px;
            text-align: center;
        }
        .impact-highly-positive { color: #28a745; font-weight: bold; }
        .impact-positive { color: #6f9c3d; }
        .impact-neutral { color: #6c757d; }
        .impact-negative { color: #dc3545; }
        .impact-highly-negative { color: #a71d2a; font-weight: bold; }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e9ecef;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .enhanced-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        .enhanced-metric {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 12px;
        }
        .enhanced-metric-value {
            font-size: 14px;
            font-weight: bold;
            color: #495057;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 280px;
            background-color: #555;
            color: white;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -140px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            line-height: 1.4;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Fast Park Accessibility Analysis</h1>
            <p>Analyze how parks affect 10-minute walkability in Rotterdam</p>
        </div>

        <div class="step">
            <h3>Step 1: Set Location & Load Data</h3>
            <p><strong>💡 Tip:</strong> Choose a city or click anywhere on the map to set location!</p>
            <div class="controls">
                <div class="control-group">
                    <label for="citySelect">Quick City Select:</label>
                    <select id="citySelect" onchange="selectCity()">
                        <option value="">Custom Location</option>
                        <option value="rotterdam" selected>Rotterdam, Netherlands</option>
                        <option value="munich">Munich, Germany</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="lat">Latitude:</label>
                    <input type="number" id="lat" value="51.9225" step="0.0001">
                </div>
                <div class="control-group">
                    <label for="lng">Longitude:</label>
                    <input type="number" id="lng" value="4.47917" step="0.0001">
                </div>
                <div class="control-group">
                    <label for="radius">Radius (m):</label>
                    <input type="number" id="radius" value="2000" min="500" max="5000" step="500">
                </div>
                <div class="control-group">
                    <label>&nbsp;</label>
                    <button class="btn" onclick="loadData()">Load Parks & Network</button>
                </div>
            </div>
        </div>

        <div id="status" class="status" style="display: none;"></div>
        
        <div class="step">
            <h3>Step 2: Select Park & Node</h3>
            <div class="controls">
                <div class="control-group">
                    <label for="parkSelect">Select Park:</label>
                    <select id="parkSelect" disabled>
                        <option>Load data first</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="nodeSelect">Select Node:</label>
                    <select id="nodeSelect" disabled>
                        <option>Select park first</option>
                    </select>
                </div>
            </div>
        </div>

        <div id="map"></div>
        <div id="mapError" style="display: none; text-align: center; padding: 10px; margin-top: 10px;">
            <p style="color: red;">⚠️ Map failed to load automatically</p>
            <button class="btn" onclick="forceInitMap()" style="background-color: #dc3545;">🔄 Force Initialize Map</button>
        </div>

        <div id="layerControls" class="step" style="display: none;">
            <h4>🗂️ Layer Visibility</h4>
            <div id="layerCheckboxes" class="controls">
                <!-- Layer checkboxes will be dynamically added here -->
            </div>
        </div>

        <div class="step">
            <h3>Step 3: Configure Analysis</h3>
            <div class="controls">
                <div class="control-group">
                    <label for="walkTime">Walking Time (minutes):</label>
                    <input type="range" id="walkTime" min="5" max="20" value="10" oninput="updateDistance()">
                    <span id="walkTimeValue">10 minutes</span>
                </div>
                <div class="control-group">
                    <label for="walkSpeed">Walking Speed (km/h):</label>
                    <input type="range" id="walkSpeed" min="3" max="6" step="0.1" value="4.5" oninput="updateDistance()">
                    <span id="walkSpeedValue">4.5 km/h</span>
                </div>
                <div class="control-group">
                    <label>Maximum Distance:</label>
                    <span id="maxDistance" class="metric-value" style="color: #007bff;">750m</span>
                </div>
                <div class="control-group">
                    <label for="vizMethod">Visualization Method:</label>
                    <select id="vizMethod">
                        <option value="convex_hull">Convex Hull (Fast)</option>
                        <option value="buffered_network">Buffered Streets (Realistic)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>&nbsp;</label>
                    <button class="btn" onclick="runAnalysis()" disabled id="analyzeBtn">Single Node Analysis</button>
                </div>
            </div>
        </div>

        <div class="step">
            <h3>Step 4: Mass Analysis (Optional)</h3>
            <p>Analyze multiple nodes around the selected park to get comprehensive statistics about the park's accessibility impact.</p>
            <div class="controls">
                <div class="control-group">
                    <label for="massNodeCount">Number of nodes to analyze:</label>
                    <select id="massNodeCount">
                        <option value="5">5 nodes (Quick)</option>
                        <option value="10" selected>10 nodes (Balanced)</option>
                        <option value="20">20 nodes (Comprehensive)</option>
                        <option value="30">30 nodes (Detailed)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="massRadius">Node search radius (m):</label>
                    <input type="range" id="massRadius" min="400" max="1200" value="600" step="100" oninput="updateMassRadius()">
                    <span id="massRadiusValue">600m</span>
                </div>
                <div class="control-group">
                    <label>&nbsp;</label>
                    <button class="btn" onclick="runMassAnalysis()" disabled id="massAnalyzeBtn" style="background-color: #28a745;">🔍 Mass Analysis</button>
                </div>
            </div>
            <div id="massProgress" style="display: none;">
                <div style="background-color: #e9ecef; border-radius: 10px; padding: 3px; margin: 10px 0;">
                    <div id="massProgressBar" style="background-color: #28a745; height: 20px; border-radius: 7px; width: 0%; transition: width 0.3s;"></div>
                </div>
                <div id="massProgressText">Analyzing node 0/10...</div>
            </div>
        </div>

        <div id="results" class="results">
            <h3>📊 Single Node Results</h3>
            <div id="metrics" class="metrics"></div>
            
            <div class="step">
                <h4>🔍 Enhanced Metrics</h4>
                <div id="enhancedMetrics" class="enhanced-metrics"></div>
            </div>
            
            <div class="step">
                <h4>📈 Visual Analysis</h4>
                <div class="charts-grid">
                    <div class="chart-container">
                        <canvas id="comparisonChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="detailedMetricsChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div id="massResults" class="results" style="display: none;">
            <h3>📈 Mass Analysis Results</h3>
            <div id="massOverview" class="metrics"></div>
            
            <div class="step">
                <h4>📊 Statistical Summary</h4>
                <div id="massStats" class="metrics"></div>
            </div>
            
            <div class="step">
                <h4>📈 Interactive Visualizations</h4>
                <div class="charts-grid">
                    <div class="chart-container">
                        <canvas id="massComparisonChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="impactDistributionChart"></canvas>
                    </div>
                </div>
                <div class="charts-grid">
                    <div class="chart-container">
                        <canvas id="scatterPlotChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="impactCategoriesChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="step">
                <h4>🎯 Park Impact Classification</h4>
                <div id="parkImpact" style="margin: 15px 0;"></div>
            </div>
            
            <div class="step">
                <h4>📋 Detailed Results</h4>
                <div id="massTable" style="overflow-x: auto; margin: 15px 0;">
                    <!-- Detailed results table will be inserted here -->
                </div>
                <button class="btn" onclick="downloadMassResults()" style="background-color: #6f42c1; margin-top: 10px;">
                    📥 Download Results (CSV)
                </button>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        let map;
        let parksData = [];
        let nodesData = [];
        let layerGroups = {
            parks: [],
            nodes: [],
            withPark: null,
            withoutPark: null
        };
        let layerVisibility = {
            parks: true,
            nodes: true,
            withPark: true,
            withoutPark: true
        };
        let centerMarker = null;

        // Initialize map
        function initMap() {
            try {
                console.log('Starting map initialization...');
                console.log('Leaflet version:', L.version);
                
                const lat = parseFloat(document.getElementById('lat').value) || 51.9225;
                const lng = parseFloat(document.getElementById('lng').value) || 4.47917;
                
                console.log('Map coordinates:', lat, lng);
                
                // Clear any existing map
                if (map) {
                    map.remove();
                }
                
                map = L.map('map', {
                    preferCanvas: false,
                    zoomControl: true
                }).setView([lat, lng], 13);
                
                console.log('Map object created');
                
                // Multiple tile layer providers for better reliability
                const tileProviders = [
                    {
                        name: 'OpenStreetMap',
                        url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                        attribution: '© OpenStreetMap contributors',
                        subdomains: 'abc'
                    },
                    {
                        name: 'CartoDB Light',
                        url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                        attribution: '© OpenStreetMap contributors © CARTO',
                        subdomains: 'abcd'
                    },
                    {
                        name: 'OpenTopoMap',
                        url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                        attribution: '© OpenStreetMap contributors, © OpenTopoMap (CC-BY-SA)',
                        subdomains: 'abc'
                    }
                ];
                
                let currentProviderIndex = 0;
                let tileErrorCount = 0;
                let tileLayer;
                
                function tryTileProvider(providerIndex) {
                    if (providerIndex >= tileProviders.length) {
                        showStatus('Unable to load any map tiles. Please check your internet connection.', 'error');
                        return;
                    }
                    
                    const provider = tileProviders[providerIndex];
                    console.log(`Trying tile provider: ${provider.name}`);
                    
                    if (tileLayer) {
                        map.removeLayer(tileLayer);
                    }
                    
                    tileLayer = L.tileLayer(provider.url, {
                        attribution: provider.attribution,
                        maxZoom: 19,
                        subdomains: provider.subdomains,
                        crossOrigin: true,
                        timeout: 10000
                    });
                    
                    let providerErrors = 0;
                    
                    tileLayer.on('tileerror', function(error) {
                        providerErrors++;
                        console.warn(`Tile error with ${provider.name}:`, error);
                        
                        if (providerErrors > 3) {
                            console.log(`Too many errors with ${provider.name}, trying next provider`);
                            setTimeout(() => tryTileProvider(providerIndex + 1), 1000);
                        }
                    });
                    
                    tileLayer.on('tileload', function() {
                        console.log(`Tiles loading successfully from ${provider.name}`);
                        tileErrorCount = 0;
                        hideStatus();
                    });
                    
                    tileLayer.addTo(map);
                }
                
                // Start with first provider
                tryTileProvider(0);
                
                console.log('Tile layer added');
                
                // Add center marker to show current location
                updateCenterMarker(lat, lng);
                
                // Add click handler to update lat/lng
                map.on('click', function(e) {
                    const newLat = e.latlng.lat;
                    const newLng = e.latlng.lng;
                    
                    // Update input fields
                    document.getElementById('lat').value = newLat.toFixed(6);
                    document.getElementById('lng').value = newLng.toFixed(6);
                    
                    // Update center marker
                    updateCenterMarker(newLat, newLng);
                    
                    // Reset city selector to custom since user clicked manually
                    document.getElementById('citySelect').value = '';
                    
                    // Update page title to generic
                    document.querySelector('.header h1').innerHTML = '🌳 Fast Park Accessibility Analysis';
                    
                    // Show brief feedback
                    showStatus(`Custom location set: ${newLat.toFixed(4)}, ${newLng.toFixed(4)}`, 'success');
                    setTimeout(hideStatus, 2000);
                    
                    // Clear any existing data since location changed
                    clearMarkers();
                });
                
                console.log('Map initialization complete');
                
            } catch (error) {
                console.error('Map initialization failed:', error);
                showStatus('Map initialization failed: ' + error.message, 'error');
                
                // Fallback: show error in map container
                document.getElementById('map').innerHTML = 
                    '<div style="padding: 20px; text-align: center; color: red;">' +
                    '<h3>Map Loading Error</h3>' +
                    '<p>' + error.message + '</p>' +
                    '<p>Please refresh the page and try again.</p>' +
                    '</div>';
            }
        }

        function showStatus(message, type = 'loading') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
        }

        function hideStatus() {
            document.getElementById('status').style.display = 'none';
        }

        function updateCenterMarker(lat, lng) {
            // Remove existing center marker
            if (centerMarker) {
                map.removeLayer(centerMarker);
            }
            
            // Add new center marker
            centerMarker = L.marker([lat, lng], {
                icon: L.divIcon({
                    html: '📍',
                    iconSize: [20, 20],
                    className: 'center-marker'
                })
            }).addTo(map);
            
            centerMarker.bindPopup(`Center: ${lat.toFixed(4)}, ${lng.toFixed(4)}<br><small>Click anywhere to move</small>`);
        }

        function selectCity() {
            const citySelect = document.getElementById('citySelect');
            const selectedCity = citySelect.value;
            
            const cities = {
                rotterdam: {
                    name: "Rotterdam, Netherlands",
                    lat: 51.9225,
                    lng: 4.47917,
                    zoom: 13
                },
                munich: {
                    name: "Munich, Germany", 
                    lat: 48.1351,
                    lng: 11.5820,
                    zoom: 12
                }
            };
            
            if (selectedCity && cities[selectedCity]) {
                const city = cities[selectedCity];
                
                // Update input fields
                document.getElementById('lat').value = city.lat.toFixed(6);
                document.getElementById('lng').value = city.lng.toFixed(6);
                
                // Clear any existing data since location changed
                clearMarkers();
                
                // Update map center with smooth transition and marker
                console.log(`Moving map to ${city.name}: ${city.lat}, ${city.lng}`);
                map.flyTo([city.lat, city.lng], city.zoom, {
                    animate: true,
                    duration: 1.5
                });
                
                // Update marker after map movement
                setTimeout(() => {
                    updateCenterMarker(city.lat, city.lng);
                }, 100);
                
                // Update page title
                document.querySelector('.header h1').innerHTML = `🌳 Fast Park Accessibility Analysis - ${city.name}`;
                
                // Show feedback
                showStatus(`Switched to ${city.name}`, 'success');
                setTimeout(hideStatus, 2500);
            }
        }

        function updateDistance() {
            const walkTime = parseFloat(document.getElementById('walkTime').value);
            const walkSpeed = parseFloat(document.getElementById('walkSpeed').value);
            
            // Update display values
            document.getElementById('walkTimeValue').textContent = `${walkTime} minutes`;
            document.getElementById('walkSpeedValue').textContent = `${walkSpeed} km/h`;
            
            // Calculate maximum walking distance
            const speedMs = (walkSpeed * 1000) / 60; // Convert km/h to m/min
            const maxDistanceM = Math.round(walkTime * speedMs);
            
            document.getElementById('maxDistance').textContent = `${maxDistanceM}m`;
        }

        function updateMassRadius() {
            const radius = document.getElementById('massRadius').value;
            document.getElementById('massRadiusValue').textContent = `${radius}m`;
        }

        function updateLayerControls() {
            const container = document.getElementById('layerCheckboxes');
            const layerConfigs = [];

            // Add layer configs for existing layers
            if (layerGroups.parks.length > 0) {
                layerConfigs.push({
                    key: 'parks',
                    name: 'Parks',
                    color: 'green',
                    count: layerGroups.parks.length
                });
            }
            
            if (layerGroups.nodes.length > 0) {
                layerConfigs.push({
                    key: 'nodes', 
                    name: 'Nodes',
                    color: 'blue',
                    count: layerGroups.nodes.length
                });
            }
            
            if (layerGroups.withPark) {
                layerConfigs.push({
                    key: 'withPark',
                    name: 'WITH Park Area',
                    color: 'green'
                });
            }
            
            if (layerGroups.withoutPark) {
                layerConfigs.push({
                    key: 'withoutPark',
                    name: 'WITHOUT Park Area', 
                    color: 'red'
                });
            }

            // Clear and rebuild checkboxes
            container.innerHTML = '';
            
            if (layerConfigs.length > 0) {
                layerConfigs.forEach(config => {
                    const controlGroup = document.createElement('div');
                    controlGroup.className = 'control-group';
                    
                    const label = document.createElement('label');
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = `show_${config.key}`;
                    checkbox.checked = layerVisibility[config.key];
                    
                    const colorIndicator = document.createElement('span');
                    colorIndicator.style.color = config.color;
                    colorIndicator.textContent = '■ ';
                    
                    const text = document.createElement('span');
                    text.textContent = config.name;
                    if (config.count) {
                        text.textContent += ` (${config.count})`;
                    }
                    
                    label.appendChild(checkbox);
                    label.appendChild(colorIndicator);
                    label.appendChild(text);
                    controlGroup.appendChild(label);
                    container.appendChild(controlGroup);
                    
                    // Add event listener
                    checkbox.addEventListener('change', function() {
                        toggleLayer(config.key, this.checked);
                    });
                });
                
                // Show layer controls
                document.getElementById('layerControls').style.display = 'block';
            } else {
                // Hide if no layers
                document.getElementById('layerControls').style.display = 'none';
            }
        }

        function toggleLayer(layerKey, show) {
            layerVisibility[layerKey] = show;
            
            if (layerKey === 'parks') {
                layerGroups.parks.forEach(layer => {
                    if (show) {
                        map.addLayer(layer);
                    } else {
                        map.removeLayer(layer);
                    }
                });
            } else if (layerKey === 'nodes') {
                layerGroups.nodes.forEach(layer => {
                    if (show) {
                        map.addLayer(layer);
                    } else {
                        map.removeLayer(layer);
                    }
                });
            } else if (layerKey === 'withPark' && layerGroups.withPark) {
                if (show) {
                    map.addLayer(layerGroups.withPark);
                } else {
                    map.removeLayer(layerGroups.withPark);
                }
            } else if (layerKey === 'withoutPark' && layerGroups.withoutPark) {
                if (show) {
                    map.addLayer(layerGroups.withoutPark);
                } else {
                    map.removeLayer(layerGroups.withoutPark);
                }
            }
        }

        async function loadData() {
            const lat = parseFloat(document.getElementById('lat').value);
            const lng = parseFloat(document.getElementById('lng').value);
            const radius = parseInt(document.getElementById('radius').value);

            showStatus('Loading parks and street network...', 'loading');
            
            try {
                const response = await fetch(`/load-data?lat=${lat}&lng=${lng}&radius=${radius}`);
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Failed to load data');
                }

                parksData = data.parks;
                nodesData = data.nodes;

                // Update map center
                map.setView([lat, lng], 13);
                
                // Clear existing markers
                clearMarkers();
                
                // Add parks to map and selector
                const parkSelect = document.getElementById('parkSelect');
                parkSelect.innerHTML = '<option value="">Select a park...</option>';
                
                parksData.forEach(park => {
                    // Add to map with click handler
                    const polygon = L.polygon(park.coords, {
                        color: 'green',
                        fillColor: 'green',
                        fillOpacity: 0.3
                    });
                    
                    // Only add to map if parks are visible
                    if (layerVisibility.parks) {
                        polygon.addTo(map);
                    }
                    
                    polygon.bindPopup(`<b>${park.name}</b><br>Area: ${Math.round(park.area_m2)} m²<br><i>Click to select</i>`);
                    
                    // Add click handler to select park
                    polygon.on('click', function() {
                        selectPark(park.id);
                    });
                    
                    layerGroups.parks.push(polygon);
                    
                    // Add to selector
                    const option = document.createElement('option');
                    option.value = park.id;
                    option.textContent = `${park.name} (${Math.round(park.area_m2/1000)}k m²)`;
                    parkSelect.appendChild(option);
                });

                updateLayerControls();

                parkSelect.disabled = false;
                showStatus(`Loaded ${parksData.length} parks and ${nodesData.length} nodes`, 'success');
                
                setTimeout(hideStatus, 3000);
                
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
                setTimeout(hideStatus, 5000);
            }
        }

        function clearMarkers() {
            // Clear all layer groups
            Object.values(layerGroups).forEach(layerGroup => {
                if (Array.isArray(layerGroup)) {
                    layerGroup.forEach(layer => map.removeLayer(layer));
                } else if (layerGroup) {
                    map.removeLayer(layerGroup);
                }
            });
            
            // Reset layer groups
            layerGroups.parks = [];
            layerGroups.nodes = [];
            layerGroups.withPark = null;
            layerGroups.withoutPark = null;
            
            // Update controls
            updateLayerControls();
        }

        function selectPark(parkId) {
            // Update dropdown
            const parkSelect = document.getElementById('parkSelect');
            parkSelect.value = parkId;
            
            // Update map display
            updateParkSelection(parkId);
        }

        function selectNode(nodeId) {
            // Update dropdown  
            const nodeSelect = document.getElementById('nodeSelect');
            nodeSelect.value = nodeId;
            
            // Update map display
            updateNodeSelection(nodeId);
        }

        function updateParkSelection(selectedParkId) {
            // Clear existing nodes
            layerGroups.nodes.forEach(marker => map.removeLayer(marker));
            layerGroups.nodes = [];
            
            if (!selectedParkId) {
                updateLayerControls();
                return;
            }
            
            // Find selected park
            const selectedPark = parksData.find(p => p.id === selectedParkId);
            if (!selectedPark) return;
            
            // Update park colors
            layerGroups.parks.forEach((marker, index) => {
                const park = parksData[index];
                if (park.id === selectedParkId) {
                    marker.setStyle({color: 'red', fillColor: 'red', fillOpacity: 0.5});
                } else {
                    marker.setStyle({color: 'green', fillColor: 'green', fillOpacity: 0.3});
                }
            });
            
            // Center map on park
            map.setView(selectedPark.centroid, 15);
            
            // Load nearby nodes
            loadNodesForPark(selectedPark);
        }

        function updateNodeSelection(selectedNodeId) {
            // Update node colors
            layerGroups.nodes.forEach(marker => {
                if (marker.options.nodeId === selectedNodeId) {
                    marker.setStyle({color: 'orange', fillColor: 'orange'});
                } else {
                    marker.setStyle({color: 'blue', fillColor: 'blue'});
                }
            });
        }

        function loadNodesForPark(selectedPark) {
            // Add nearby nodes to map and selector
            const nodeSelect = document.getElementById('nodeSelect');
            nodeSelect.innerHTML = '<option value="">Select a node...</option>';
            
            // Filter nodes near park (within reasonable distance)
            const nearbyNodes = nodesData.filter(node => {
                const distance = getDistance(
                    selectedPark.centroid[0], selectedPark.centroid[1],
                    node.lat, node.lng
                );
                return distance < 800; // Within 800m of park
            });
            
            nearbyNodes.forEach(node => {
                // Add to map with click handler
                const marker = L.circleMarker([node.lat, node.lng], {
                    radius: 4,
                    color: 'blue',
                    fillColor: 'blue',
                    fillOpacity: 0.8,
                    nodeId: node.id  // Store node ID for reference
                });
                
                // Only add to map if nodes are visible
                if (layerVisibility.nodes) {
                    marker.addTo(map);
                }
                
                marker.bindPopup(`Node ${node.id}<br><i>Click to select</i>`);
                
                // Add click handler to select node
                marker.on('click', function() {
                    selectNode(node.id);
                });
                
                layerGroups.nodes.push(marker);
                
                // Add to selector
                const option = document.createElement('option');
                option.value = node.id;
                option.textContent = `Node ${node.id}`;
                nodeSelect.appendChild(option);
            });
            
            updateLayerControls();
            
            nodeSelect.disabled = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('massAnalyzeBtn').disabled = false;
        }

        document.getElementById('parkSelect').addEventListener('change', function() {
            updateParkSelection(this.value);
        });

        document.getElementById('nodeSelect').addEventListener('change', function() {
            updateNodeSelection(this.value);
        });


        function getDistance(lat1, lon1, lat2, lon2) {
            const R = 6371e3; // Earth's radius in meters
            const φ1 = lat1 * Math.PI/180;
            const φ2 = lat2 * Math.PI/180;
            const Δφ = (lat2-lat1) * Math.PI/180;
            const Δλ = (lon2-lon1) * Math.PI/180;

            const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
                    Math.cos(φ1) * Math.cos(φ2) *
                    Math.sin(Δλ/2) * Math.sin(Δλ/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

            return R * c;
        }

        async function runAnalysis() {
            const lat = parseFloat(document.getElementById('lat').value);
            const lng = parseFloat(document.getElementById('lng').value);
            const radius = parseInt(document.getElementById('radius').value);
            const parkId = document.getElementById('parkSelect').value;
            const nodeId = document.getElementById('nodeSelect').value;
            const walkTime = parseFloat(document.getElementById('walkTime').value);
            const walkSpeed = parseFloat(document.getElementById('walkSpeed').value);
            const vizMethod = document.getElementById('vizMethod').value;
            
            if (!parkId || !nodeId) {
                alert('Please select both a park and a node');
                return;
            }

            const methodName = vizMethod === 'buffered_network' ? 'Buffered Streets' : 'Convex Hull';
            showStatus(`Calculating ${walkTime}-minute accessibility (${methodName})...`, 'loading');
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat, lng, radius, park_id: parkId, node_id: nodeId,
                        walk_time: walkTime, walk_speed: walkSpeed, viz_method: vizMethod
                    })
                });
                
                const results = await response.json();
                
                if (!response.ok) {
                    throw new Error(results.detail || 'Analysis failed');
                }

                displayResults(results);
                showStatus('Analysis complete!', 'success');
                setTimeout(hideStatus, 3000);
                
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
                setTimeout(hideStatus, 5000);
            } finally {
                document.getElementById('analyzeBtn').disabled = false;
            }
        }

        function displayResults(results) {
            // Auto-hide parks and nodes when showing results
            layerVisibility.parks = false;
            layerVisibility.nodes = false;
            
            // Hide existing parks and nodes
            layerGroups.parks.forEach(layer => map.removeLayer(layer));
            layerGroups.nodes.forEach(layer => map.removeLayer(layer));
            
            // Clear previous result layers
            if (layerGroups.withPark) {
                map.removeLayer(layerGroups.withPark);
                layerGroups.withPark = null;
            }
            if (layerGroups.withoutPark) {
                map.removeLayer(layerGroups.withoutPark);
                layerGroups.withoutPark = null;
            }
            
            // Add isochrones to map
            if (results.with_park.boundary && results.with_park.boundary.length > 0) {
                layerGroups.withPark = L.polygon(results.with_park.boundary, {
                    color: 'green',
                    fillOpacity: 0,
                    weight: 3
                });
                
                if (layerVisibility.withPark) {
                    layerGroups.withPark.addTo(map);
                }
                
                layerGroups.withPark.bindPopup(`WITH park: ${results.with_park.area_km2.toFixed(3)} km²`);
            }
            
            if (results.without_park.boundary && results.without_park.boundary.length > 0) {
                layerGroups.withoutPark = L.polygon(results.without_park.boundary, {
                    color: 'red',
                    fillOpacity: 0,
                    weight: 3,
                    dashArray: '5, 5'
                });
                
                if (layerVisibility.withoutPark) {
                    layerGroups.withoutPark.addTo(map);
                }
                
                layerGroups.withoutPark.bindPopup(`WITHOUT park: ${results.without_park.area_km2.toFixed(3)} km²`);
            }
            
            updateLayerControls();

            // Display main metrics
            const metrics = document.getElementById('metrics');
            const impactClass = `impact-${results.impact_category.replace('_', '-')}`;
            
            metrics.innerHTML = `
                <div class="metric">
                    <div class="metric-value">${results.with_park.area_km2.toFixed(3)} km²</div>
                    <div class="metric-label">WITH Park (Area)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${results.enhanced_metrics.with_street_length_km.toFixed(3)} km</div>
                    <div class="metric-label">WITH Park (Streets)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${results.without_park.area_km2.toFixed(3)} km²</div>
                    <div class="metric-label">WITHOUT Park (Area)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${results.enhanced_metrics.without_street_length_km.toFixed(3)} km</div>
                    <div class="metric-label">WITHOUT Park (Streets)</div>
                </div>
                <div class="metric">
                    <div class="metric-value ${impactClass}">${results.street_network_difference_km > 0 ? '+' : ''}${results.street_network_difference_km.toFixed(3)} km</div>
                    <div class="metric-label">Street Impact (${results.street_network_change_pct.toFixed(1)}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value ${impactClass} tooltip">
                        ${results.impact_label}
                        <span class="tooltiptext">${getImpactTooltip(results.impact_category, results.impact_label)}</span>
                    </div>
                    <div class="metric-label">Overall Assessment</div>
                </div>
            `;

            // Display enhanced metrics
            displayEnhancedMetrics(results);
            
            // Create charts
            createSingleAnalysisCharts(results);

            document.getElementById('results').style.display = 'block';
        }

        function displayEnhancedMetrics(results) {
            const enhanced = document.getElementById('enhancedMetrics');
            const metrics = results.enhanced_metrics;
            
            enhanced.innerHTML = `
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${results.with_park.reachable_nodes}</div>
                    <div>WITH Park Nodes</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${results.without_park.reachable_nodes}</div>
                    <div>WITHOUT Park Nodes</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${results.with_park.street_network_stats.edge_count}</div>
                    <div>WITH Park Edges</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${results.without_park.street_network_stats.edge_count}</div>
                    <div>WITHOUT Park Edges</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${metrics.with_street_density.toFixed(1)} km/km²</div>
                    <div>WITH Street Density</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${metrics.without_street_density.toFixed(1)} km/km²</div>
                    <div>WITHOUT Street Density</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${(metrics.with_connectivity * 100).toFixed(1)}%</div>
                    <div>WITH Connectivity</div>
                </div>
                <div class="enhanced-metric">
                    <div class="enhanced-metric-value">${(metrics.without_connectivity * 100).toFixed(1)}%</div>
                    <div>WITHOUT Connectivity</div>
                </div>
            `;
        }

        let singleCharts = {};

        function createSingleAnalysisCharts(results) {
            // Destroy existing charts
            Object.values(singleCharts).forEach(chart => {
                if (chart) chart.destroy();
            });
            
            // Comparison Chart
            const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
            singleCharts.comparison = new Chart(comparisonCtx, {
                type: 'bar',
                data: {
                    labels: ['Accessible Area (km²)', 'Street Length (km)', 'Reachable Nodes', 'Street Edges'],
                    datasets: [{
                        label: 'WITH Park',
                        data: [
                            results.with_park.area_km2,
                            results.enhanced_metrics.with_street_length_km,
                            results.with_park.reachable_nodes / 100, // Scale for visibility
                            results.with_park.street_network_stats.edge_count / 100 // Scale for visibility
                        ],
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1
                    }, {
                        label: 'WITHOUT Park',
                        data: [
                            results.without_park.area_km2,
                            results.enhanced_metrics.without_street_length_km,
                            results.without_park.reachable_nodes / 100,
                            results.without_park.street_network_stats.edge_count / 100
                        ],
                        backgroundColor: 'rgba(220, 53, 69, 0.8)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'WITH vs WITHOUT Park Comparison'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    if (context.dataIndex === 2 || context.dataIndex === 3) {
                                        // Scale back for nodes and edges in tooltip
                                        label += (context.parsed.y * 100).toFixed(0);
                                        label += context.dataIndex === 2 ? ' nodes' : ' edges';
                                    } else {
                                        label += context.parsed.y.toFixed(3);
                                        label += context.dataIndex === 0 ? ' km²' : ' km';
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Detailed Metrics Chart
            const detailedCtx = document.getElementById('detailedMetricsChart').getContext('2d');
            singleCharts.detailed = new Chart(detailedCtx, {
                type: 'radar',
                data: {
                    labels: ['Area Impact (%)', 'Street Network Impact (%)', 'Connectivity Change (%)', 'Node Change (%)', 'Edge Change (%)'],
                    datasets: [{
                        label: 'Park Impact',
                        data: [
                            results.area_change_pct,
                            results.street_network_change_pct,
                            (results.enhanced_metrics.connectivity_change * 100),
                            (results.node_difference / results.with_park.reachable_nodes * 100),
                            (results.edge_difference / results.with_park.street_network_stats.edge_count * 100)
                        ],
                        backgroundColor: 'rgba(0, 123, 255, 0.2)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        pointBackgroundColor: 'rgba(0, 123, 255, 1)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgba(0, 123, 255, 1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Impact Analysis (% Changes)'
                        }
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        }
                    }
                }
            });
        }

        let massResultsData = [];
        let massCharts = {};

        function getImpactTooltip(impactCategory, impactLabel) {
            const tooltips = {
                'highly_positive': 'This park significantly improves street network accessibility (>5% increase). The park provides important pedestrian connections, pathways, or shortcuts that enhance walkability and reduce travel distances to destinations.',
                'positive': 'This park moderately improves street network accessibility (1-5% increase). The park offers some beneficial pedestrian routes or connections that make the area more walkable.',
                'neutral': 'This park has minimal impact on street network accessibility (-1% to +1% change). The park neither significantly helps nor hinders pedestrian movement in the area.',
                'negative': 'This park moderately reduces street network accessibility (1-5% decrease). The park may block some optimal pedestrian routes, requiring people to walk slightly longer distances.',
                'highly_negative': 'This park significantly reduces street network accessibility (>5% decrease). The park creates substantial barriers to pedestrian movement, forcing much longer walking routes to reach destinations.'
            };
            
            return tooltips[impactCategory] || 'Impact assessment based on changes to reachable street network length when the park is present versus removed.';
        }

        async function runMassAnalysis() {
            const parkId = document.getElementById('parkSelect').value;
            const nodeCount = parseInt(document.getElementById('massNodeCount').value);
            const searchRadius = parseInt(document.getElementById('massRadius').value);
            const walkTime = parseFloat(document.getElementById('walkTime').value);
            const walkSpeed = parseFloat(document.getElementById('walkSpeed').value);
            const vizMethod = document.getElementById('vizMethod').value;
            
            if (!parkId) {
                alert('Please select a park first');
                return;
            }

            // Find selected park
            const selectedPark = parksData.find(p => p.id === parkId);
            if (!selectedPark) return;

            // Get nearby nodes for mass analysis
            const nearbyNodes = nodesData.filter(node => {
                const distance = getDistance(
                    selectedPark.centroid[0], selectedPark.centroid[1],
                    node.lat, node.lng
                );
                return distance < searchRadius;
            });

            if (nearbyNodes.length === 0) {
                alert('No nodes found in the specified radius. Try increasing the search radius.');
                return;
            }

            // Select nodes for analysis (random sampling if more than requested)
            const selectedNodes = nearbyNodes.length <= nodeCount ? 
                nearbyNodes : 
                nearbyNodes.sort(() => 0.5 - Math.random()).slice(0, nodeCount);

            // Show progress
            document.getElementById('massProgress').style.display = 'block';
            document.getElementById('massAnalyzeBtn').disabled = true;

            massResultsData = [];
            const lat = parseFloat(document.getElementById('lat').value);
            const lng = parseFloat(document.getElementById('lng').value);
            const radius = parseInt(document.getElementById('radius').value);

            for (let i = 0; i < selectedNodes.length; i++) {
                const node = selectedNodes[i];
                
                // Update progress
                const progress = ((i + 1) / selectedNodes.length) * 100;
                document.getElementById('massProgressBar').style.width = progress + '%';
                document.getElementById('massProgressText').textContent = 
                    `Analyzing node ${i + 1}/${selectedNodes.length}... (${node.id})`;

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            lat, lng, radius, 
                            park_id: parkId, 
                            node_id: node.id,
                            walk_time: walkTime, 
                            walk_speed: walkSpeed, 
                            viz_method: vizMethod
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        massResultsData.push({
                            node_id: node.id,
                            node_lat: node.lat,
                            node_lng: node.lng,
                            with_area: result.with_park.area_km2,
                            without_area: result.without_park.area_km2,
                            with_street_length: result.enhanced_metrics.with_street_length_km,
                            without_street_length: result.enhanced_metrics.without_street_length_km,
                            difference: result.difference_km2,
                            street_network_difference: result.street_network_difference_km,
                            area_change_pct: result.area_change_pct,
                            street_network_change_pct: result.street_network_change_pct,
                            with_nodes: result.with_park.reachable_nodes,
                            without_nodes: result.without_park.reachable_nodes,
                            node_difference: result.node_difference,
                            with_edges: result.with_park.street_network_stats.edge_count,
                            without_edges: result.without_park.street_network_stats.edge_count,
                            edge_difference: result.edge_difference,
                            impact_category: result.impact_category,
                            impact_label: result.impact_label,
                            connectivity_with: result.enhanced_metrics.with_connectivity,
                            connectivity_without: result.enhanced_metrics.without_connectivity,
                            connectivity_change: result.enhanced_metrics.connectivity_change,
                            distance_with: result.enhanced_metrics.with_avg_distance,
                            distance_without: result.enhanced_metrics.without_avg_distance,
                            street_density_with: result.enhanced_metrics.with_street_density,
                            street_density_without: result.enhanced_metrics.without_street_density
                        });
                    }
                } catch (error) {
                    console.error('Error analyzing node', node.id, ':', error);
                }
            }

            // Hide progress and show results
            document.getElementById('massProgress').style.display = 'none';
            document.getElementById('massAnalyzeBtn').disabled = false;
            
            displayMassResults();
        }

        function displayMassResults() {
            if (massResultsData.length === 0) {
                alert('No results to display');
                return;
            }

            // Calculate enhanced statistics
            const differences = massResultsData.map(r => r.difference);
            const streetNetworkDifferences = massResultsData.map(r => r.street_network_difference);
            const areaChangePcts = massResultsData.map(r => r.area_change_pct);
            const streetNetworkChangePcts = massResultsData.map(r => r.street_network_change_pct);
            const withAreas = massResultsData.map(r => r.with_area);
            const withoutAreas = massResultsData.map(r => r.without_area);
            const withStreetLengths = massResultsData.map(r => r.with_street_length);
            const withoutStreetLengths = massResultsData.map(r => r.without_street_length);

            const avgDifference = differences.reduce((a, b) => a + b, 0) / differences.length;
            const avgStreetNetworkDifference = streetNetworkDifferences.reduce((a, b) => a + b, 0) / streetNetworkDifferences.length;
            const avgAreaChangePct = areaChangePcts.reduce((a, b) => a + b, 0) / areaChangePcts.length;
            const avgStreetNetworkChangePct = streetNetworkChangePcts.reduce((a, b) => a + b, 0) / streetNetworkChangePcts.length;
            const maxDifference = Math.max(...differences);
            const minDifference = Math.min(...differences);
            const avgWithArea = withAreas.reduce((a, b) => a + b, 0) / withAreas.length;
            const avgWithoutArea = withoutAreas.reduce((a, b) => a + b, 0) / withoutAreas.length;
            const avgWithStreetLength = withStreetLengths.reduce((a, b) => a + b, 0) / withStreetLengths.length;
            const avgWithoutStreetLength = withoutStreetLengths.reduce((a, b) => a + b, 0) / withoutStreetLengths.length;

            // Enhanced impact classification (5 levels)
            const impactCounts = {
                'highly_positive': massResultsData.filter(d => d.impact_category === 'highly_positive').length,
                'positive': massResultsData.filter(d => d.impact_category === 'positive').length,
                'neutral': massResultsData.filter(d => d.impact_category === 'neutral').length,
                'negative': massResultsData.filter(d => d.impact_category === 'negative').length,
                'highly_negative': massResultsData.filter(d => d.impact_category === 'highly_negative').length
            };

            // Legacy counts for compatibility
            const positiveImpact = impactCounts.highly_positive + impactCounts.positive;
            const negativeImpact = impactCounts.negative + impactCounts.highly_negative;
            const neutralImpact = impactCounts.neutral;

            // Display overview
            document.getElementById('massOverview').innerHTML = `
                <div class="metric">
                    <div class="metric-value">${massResultsData.length}</div>
                    <div class="metric-label">Nodes Analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgWithArea.toFixed(3)} km²</div>
                    <div class="metric-label">Avg WITH Park (Area)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgWithStreetLength.toFixed(3)} km</div>
                    <div class="metric-label">Avg WITH Park (Streets)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgWithoutArea.toFixed(3)} km²</div>
                    <div class="metric-label">Avg WITHOUT Park (Area)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgWithoutStreetLength.toFixed(3)} km</div>
                    <div class="metric-label">Avg WITHOUT Park (Streets)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgStreetNetworkDifference > 0 ? '+' : ''}${avgStreetNetworkDifference.toFixed(3)} km</div>
                    <div class="metric-label">Average Street Impact (${avgStreetNetworkChangePct.toFixed(1)}%)</div>
                </div>
            `;

            // Display statistics
            document.getElementById('massStats').innerHTML = `
                <div class="metric">
                    <div class="metric-value">${maxDifference.toFixed(3)} km²</div>
                    <div class="metric-label">Max Impact</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${minDifference.toFixed(3)} km²</div>
                    <div class="metric-label">Min Impact</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${avgAreaChangePct.toFixed(1)}%</div>
                    <div class="metric-label">Avg % Change</div>
                </div>
                <div class="metric">
                    <div class="metric-value impact-highly-positive">${impactCounts.highly_positive}</div>
                    <div class="metric-label">Highly Positive</div>
                </div>
                <div class="metric">
                    <div class="metric-value impact-positive">${impactCounts.positive}</div>
                    <div class="metric-label">Positive</div>
                </div>
                <div class="metric">
                    <div class="metric-value impact-neutral">${impactCounts.neutral}</div>
                    <div class="metric-label">Neutral</div>
                </div>
                <div class="metric">
                    <div class="metric-value impact-negative">${impactCounts.negative}</div>
                    <div class="metric-label">Negative</div>
                </div>
                <div class="metric">
                    <div class="metric-value impact-highly-negative">${impactCounts.highly_negative}</div>
                    <div class="metric-label">Highly Negative</div>
                </div>
            `;

            // Create interactive charts
            createMassAnalysisCharts(massResultsData, impactCounts);

            // Determine overall park impact based on street network change
            let parkImpactText = '';
            let parkImpactClass = '';
            
            if (avgStreetNetworkChangePct < -5) {
                parkImpactText = '🌟 <strong>HIGHLY BENEFICIAL</strong> - This park significantly improves street network accessibility by providing important pedestrian connections.';
                parkImpactClass = 'success';
            } else if (avgStreetNetworkChangePct < -1) {
                parkImpactText = '✅ <strong>BENEFICIAL</strong> - This park generally improves street network accessibility with valuable pedestrian routes.';
                parkImpactClass = 'success';
            } else if (avgStreetNetworkChangePct > 5) {
                parkImpactText = '⚠️ <strong>PROBLEMATIC</strong> - This park significantly blocks optimal street network access. Consider adding paths through or around it.';
                parkImpactClass = 'error';
            } else if (avgStreetNetworkChangePct > 1) {
                parkImpactText = '🔶 <strong>MIXED IMPACT</strong> - This park has some negative impact on street network accessibility. Consider improving connectivity.';
                parkImpactClass = 'loading';
            } else {
                parkImpactText = '➖ <strong>NEUTRAL</strong> - This park has minimal impact on overall street network accessibility.';
                parkImpactClass = 'loading';
            }

            document.getElementById('parkImpact').innerHTML = `
                <div class="status ${parkImpactClass}" style="display: block;">
                    ${parkImpactText}
                    <br><br>
                    <strong>Enhanced Summary:</strong> ${impactCounts.highly_positive} highly positive, ${impactCounts.positive} positive, 
                    ${impactCounts.neutral} neutral, ${impactCounts.negative} negative, ${impactCounts.highly_negative} highly negative impacts.
                    <br>
                    <strong>Street Network Impact:</strong> ${avgStreetNetworkChangePct.toFixed(1)}% change in reachable street length.
                    <br>
                    <strong>Area Impact:</strong> ${avgAreaChangePct.toFixed(1)}% change in accessible area.
                </div>
            `;

            // Create detailed table with enhanced metrics
            const tableHtml = `
                <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                    <thead>
                        <tr style="background-color: #f8f9fa;">
                            <th style="border: 1px solid #dee2e6; padding: 6px;">Node</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">WITH Area (km²)</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">WITH Streets (km)</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">WITHOUT Area (km²)</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">WITHOUT Streets (km)</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">Street Impact</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">% Change</th>
                            <th style="border: 1px solid #dee2e6; padding: 6px;">Impact Level</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${massResultsData.map(row => `
                            <tr>
                                <td style="border: 1px solid #dee2e6; padding: 6px;">${row.node_id}</td>
                                <td style="border: 1px solid #dee2e6; padding: 6px;">${row.with_area.toFixed(3)}</td>
                                <td style="border: 1px solid #dee2e6; padding: 6px;">${row.with_street_length.toFixed(3)}</td>
                                <td style="border: 1px solid #dee2e6; padding: 6px;">${row.without_area.toFixed(3)}</td>
                                <td style="border: 1px solid #dee2e6; padding: 6px;">${row.without_street_length.toFixed(3)}</td>
                                <td style="border: 1px solid #dee2e6; padding: 6px; color: ${row.street_network_difference > 0 ? 'red' : row.street_network_difference < 0 ? 'green' : 'gray'};">
                                    ${row.street_network_difference > 0 ? '+' : ''}${row.street_network_difference.toFixed(3)} km
                                </td>
                                <td style="border: 1px solid #dee2e6; padding: 6px; color: ${row.street_network_change_pct > 0 ? 'red' : row.street_network_change_pct < 0 ? 'green' : 'gray'};">
                                    ${row.street_network_change_pct > 0 ? '+' : ''}${row.street_network_change_pct.toFixed(1)}%
                                </td>
                                <td style="border: 1px solid #dee2e6; padding: 6px;" class="impact-${row.impact_category.replace('_', '-')} tooltip">
                                    ${row.impact_label}
                                    <span class="tooltiptext">${getImpactTooltip(row.impact_category, row.impact_label)}</span>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            
            document.getElementById('massTable').innerHTML = tableHtml;
            document.getElementById('massResults').style.display = 'block';
            
            // Scroll to results
            document.getElementById('massResults').scrollIntoView({ behavior: 'smooth' });
        }

        function createMassAnalysisCharts(data, impactCounts) {
            // Destroy existing charts
            Object.values(massCharts).forEach(chart => {
                if (chart) chart.destroy();
            });

            // 1. Mass Comparison Chart - WITH vs WITHOUT areas and streets
            const massComparisonCtx = document.getElementById('massComparisonChart').getContext('2d');
            massCharts.massComparison = new Chart(massComparisonCtx, {
                type: 'bar',
                data: {
                    labels: data.map(d => `Node ${d.node_id}`),
                    datasets: [{
                        label: 'WITH Park (Area km²)',
                        data: data.map(d => d.with_area),
                        backgroundColor: 'rgba(40, 167, 69, 0.6)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'WITH Park (Streets km)',
                        data: data.map(d => d.with_street_length),
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }, {
                        label: 'WITHOUT Park (Area km²)',
                        data: data.map(d => d.without_area),
                        backgroundColor: 'rgba(220, 53, 69, 0.6)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'WITHOUT Park (Streets km)',
                        data: data.map(d => d.without_street_length),
                        backgroundColor: 'rgba(220, 53, 69, 0.8)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: { display: true, text: 'Area & Street Network Comparison Across All Nodes' },
                        legend: { position: 'top' }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Analysis Points' } },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Area (km²)' },
                            beginAtZero: true
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Street Length (km)' },
                            beginAtZero: true,
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });

            // 2. Impact Distribution Histogram - based on street network changes
            const impactDistCtx = document.getElementById('impactDistributionChart').getContext('2d');
            const impactBuckets = [-10, -5, -1, 0, 1, 5, 10];
            const bucketCounts = new Array(impactBuckets.length - 1).fill(0);
            
            data.forEach(d => {
                for (let i = 0; i < impactBuckets.length - 1; i++) {
                    if (d.street_network_change_pct >= impactBuckets[i] && d.street_network_change_pct < impactBuckets[i + 1]) {
                        bucketCounts[i]++;
                        break;
                    }
                }
            });

            massCharts.impactDistribution = new Chart(impactDistCtx, {
                type: 'bar',
                data: {
                    labels: ['-10% to -5%', '-5% to -1%', '-1% to 0%', '0% to 1%', '1% to 5%', '5% to 10%'],
                    datasets: [{
                        label: 'Number of Nodes',
                        data: bucketCounts,
                        backgroundColor: bucketCounts.map((_, i) => {
                            if (i < 2) return 'rgba(40, 167, 69, 0.8)';  // Positive impact
                            if (i === 2 || i === 3) return 'rgba(108, 117, 125, 0.8)';  // Neutral
                            return 'rgba(220, 53, 69, 0.8)';  // Negative impact
                        })
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Street Network Impact Distribution' },
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Number of Nodes' } },
                        x: { title: { display: true, text: 'Street Network Change (%)' } }
                    }
                }
            });

            // 3. Scatter Plot - Street Length vs Street Network Impact
            const scatterCtx = document.getElementById('scatterPlotChart').getContext('2d');
            massCharts.scatter = new Chart(scatterCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Street Network Impact vs Street Length',
                        data: data.map(d => ({ x: d.with_street_length, y: d.street_network_change_pct })),
                        backgroundColor: data.map(d => {
                            switch(d.impact_category) {
                                case 'highly_positive': return 'rgba(40, 167, 69, 0.8)';
                                case 'positive': return 'rgba(111, 156, 61, 0.8)';
                                case 'neutral': return 'rgba(108, 117, 125, 0.8)';
                                case 'negative': return 'rgba(220, 53, 69, 0.8)';
                                case 'highly_negative': return 'rgba(167, 29, 42, 0.8)';
                                default: return 'rgba(0, 123, 255, 0.8)';
                            }
                        }),
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Street Network Impact vs Baseline Street Length' },
                        legend: { display: false }
                    },
                    scales: {
                        x: { title: { display: true, text: 'WITH Park Street Length (km)' } },
                        y: { title: { display: true, text: 'Street Network Impact (% Change)' } }
                    }
                }
            });

            // 4. Impact Categories Pie Chart
            const categoriesCtx = document.getElementById('impactCategoriesChart').getContext('2d');
            massCharts.categories = new Chart(categoriesCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Highly Negative'],
                    datasets: [{
                        data: [
                            impactCounts.highly_positive,
                            impactCounts.positive,
                            impactCounts.neutral,
                            impactCounts.negative,
                            impactCounts.highly_negative
                        ],
                        backgroundColor: [
                            'rgba(40, 167, 69, 0.8)',
                            'rgba(111, 156, 61, 0.8)',
                            'rgba(108, 117, 125, 0.8)',
                            'rgba(220, 53, 69, 0.8)',
                            'rgba(167, 29, 42, 0.8)'
                        ],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Impact Category Distribution' },
                        legend: { position: 'right' }
                    }
                }
            });
        }

        function downloadMassResults() {
            if (massResultsData.length === 0) {
                alert('No data to download');
                return;
            }

            const csvContent = [
                'Node ID,Latitude,Longitude,WITH Park Area (km²),WITH Park Streets (km),WITHOUT Park Area (km²),WITHOUT Park Streets (km),Area Difference (km²),Street Network Difference (km),Area Change (%),Street Network Change (%),WITH Nodes,WITHOUT Nodes,Node Difference,WITH Edges,WITHOUT Edges,Edge Difference,Impact Category,Connectivity WITH (%),Connectivity WITHOUT (%),Distance WITH (m),Distance WITHOUT (m),Street Density WITH (km/km²),Street Density WITHOUT (km/km²)',
                ...massResultsData.map(row => 
                    `${row.node_id},${row.node_lat},${row.node_lng},${row.with_area},${row.with_street_length},${row.without_area},${row.without_street_length},${row.difference},${row.street_network_difference},${row.area_change_pct.toFixed(2)},${row.street_network_change_pct.toFixed(2)},${row.with_nodes},${row.without_nodes},${row.node_difference},${row.with_edges},${row.without_edges},${row.edge_difference},"${row.impact_label}",${(row.connectivity_with * 100).toFixed(2)},${(row.connectivity_without * 100).toFixed(2)},${row.distance_with.toFixed(0)},${row.distance_without.toFixed(0)},${row.street_density_with.toFixed(2)},${row.street_density_without.toFixed(2)}`
                )
            ].join('\\n');

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mass_analysis_${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            window.URL.revokeObjectURL(url);
        }

        // Initialize when page loads - multiple fallback methods
        let mapInitialized = false;
        
        function safeInitialize() {
            if (mapInitialized) return;
            
            console.log('🗺️ Checking initialization conditions...');
            console.log('- Leaflet available:', typeof L !== 'undefined');
            console.log('- Leaflet version:', typeof L !== 'undefined' ? L.version : 'N/A');
            console.log('- Map container exists:', document.getElementById('map') !== null);
            console.log('- Document ready state:', document.readyState);
            console.log('- Window location:', window.location.href);
            
            if (typeof L !== 'undefined' && document.getElementById('map')) {
                try {
                    mapInitialized = true;
                    console.log('✅ Starting map initialization...');
                    document.getElementById('mapError').style.display = 'none';
                    
                    // Show loading status
                    showStatus('Initializing map...', 'loading');
                    
                    initMap();
                    updateDistance();
                    updateMassRadius();
                    
                    console.log('✅ Map initialization completed successfully');
                    
                } catch (error) {
                    console.error('❌ Map initialization failed:', error);
                    mapInitialized = false;
                    showStatus('Map initialization failed: ' + error.message, 'error');
                    document.getElementById('mapError').style.display = 'block';
                }
            } else {
                console.log('⏳ Not ready yet, will retry...');
                // Show error button after multiple failed attempts
                setTimeout(function() {
                    if (!mapInitialized) {
                        console.log('⚠️ Showing manual initialization option');
                        document.getElementById('mapError').style.display = 'block';
                    }
                }, 3000);
                setTimeout(safeInitialize, 500);
            }
        }
        
        function forceInitMap() {
            console.log('🔄 Force initialization triggered by user');
            mapInitialized = false; // Reset flag
            document.getElementById('mapError').style.display = 'none';
            
            // Clear any existing map
            const mapContainer = document.getElementById('map');
            mapContainer.innerHTML = '';
            
            setTimeout(safeInitialize, 100);
        }
        
        // Method 1: DOMContentLoaded
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM Content Loaded event fired');
            setTimeout(safeInitialize, 100);
        });
        
        // Method 2: Window load (fallback)
        window.addEventListener('load', function() {
            console.log('Window load event fired');
            setTimeout(safeInitialize, 100);
        });
        
        // Method 3: Immediate check if already loaded
        if (document.readyState === 'loading') {
            console.log('Document still loading...');
        } else {
            console.log('Document already loaded, initializing immediately');
            setTimeout(safeInitialize, 100);
        }
    </script>
</body>
</html>
    """

@app.get("/load-data")
async def load_data(lat: float, lng: float, radius: int = 2000):
    """Load parks and street network data for an area"""
    try:
        data = await analyzer.load_area_data(lat, lng, radius)
        return {
            'parks': data['parks'],
            'nodes': data['nodes'],
            'center': data['center']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_accessibility(request: dict):
    """Analyze park accessibility impact"""
    try:
        lat = request['lat']
        lng = request['lng'] 
        radius = request['radius']
        park_id = request['park_id']
        node_id = request['node_id']
        walk_time = request.get('walk_time', 10.0)  # Default 10 minutes
        walk_speed = request.get('walk_speed', 4.5)  # Default 4.5 km/h
        viz_method = request.get('viz_method', 'convex_hull')  # Default convex hull
        
        result = await analyzer.calculate_accessibility(
            lat, lng, radius, park_id, node_id, walk_time, walk_speed, viz_method
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)