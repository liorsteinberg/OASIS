from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional, Tuple, Union
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull
try:
    import networkit as nk
    NETWORKIT_AVAILABLE = True
except ImportError:
    nk = None
    NETWORKIT_AVAILABLE = False

class NetworkAdapter:
    """Abstract base class for network analysis backends"""
    
    def __init__(self, graph: nx.Graph):
        self.nx_graph = graph
        
    def shortest_path_lengths(self, start_node: int, max_distance: float) -> Dict[int, float]:
        """Calculate shortest path lengths from start_node within max_distance"""
        raise NotImplementedError
    
    def connected_components(self, nodes: List[int]) -> List[List[int]]:
        """Find connected components in subgraph containing given nodes"""
        raise NotImplementedError
    
    def remove_edges(self, edges_to_remove: List[Tuple[int, int]]) -> 'NetworkAdapter':
        """Create new adapter with specified edges removed"""
        raise NotImplementedError

class NetworkXAdapter(NetworkAdapter):
    """NetworkX-based network analysis"""
    
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        self._shortest_path_cache = {}  # Cache for reusing calculations
        self._max_cache_size = 50   # Reduced cache size for better memory management
        
    def shortest_path_lengths(self, start_node: int, max_distance: float) -> Dict[int, float]:
        """Calculate shortest path lengths using NetworkX with caching optimization"""
        try:
            cache_key = (start_node, max_distance)
            
            # Check cache first
            if cache_key in self._shortest_path_cache:
                return self._shortest_path_cache[cache_key]
            
            # Calculate and cache result
            result = nx.single_source_dijkstra_path_length(
                self.nx_graph, start_node, weight='length', cutoff=max_distance
            )
            
            # Manage cache size
            if len(self._shortest_path_cache) >= self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._shortest_path_cache))
                del self._shortest_path_cache[oldest_key]
            
            self._shortest_path_cache[cache_key] = result
            return result
            
        except KeyError as e:
            print(f"NetworkX KeyError during Dijkstra: {e}")
            return {}
    
    def connected_components(self, nodes: List[int]) -> List[List[int]]:
        """Find connected components using NetworkX"""
        subgraph = self.nx_graph.subgraph(nodes)
        return [list(component) for component in nx.connected_components(subgraph)]
    
    def remove_edges(self, edges_to_remove: List[Tuple[int, int]]) -> 'NetworkXAdapter':
        """Create new NetworkX adapter with edges removed"""
        new_graph = self.nx_graph.copy()
        for edge in edges_to_remove:
            if new_graph.has_edge(edge[0], edge[1]):
                new_graph.remove_edge(edge[0], edge[1])
        return NetworkXAdapter(new_graph)

class NetworKitAdapter(NetworkAdapter):
    """NetworKit-based network analysis (faster for large networks)"""
    
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        if not NETWORKIT_AVAILABLE:
            raise ImportError("NetworKit is not installed")
        
        self._shortest_path_cache = {}  # Cache for reusing calculations
        self._max_cache_size = 50   # Reduced cache size for better memory management
        
        # Convert NetworkX graph to NetworKit with fallback handling
        self.has_weights = False  # Track if weights were successfully converted
        
        # NetworKit has issues with MultiGraph, so convert to simple Graph if needed
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            print("NetworKit: Converting MultiGraph to simple Graph for compatibility")
            simple_graph = nx.Graph()
            # Copy nodes with their attributes
            simple_graph.add_nodes_from(graph.nodes(data=True))
            # Copy edges, taking the first parallel edge if multiple exist
            for u, v, data in graph.edges(data=True):
                if not simple_graph.has_edge(u, v):
                    simple_graph.add_edge(u, v, **data)
            graph_for_nk = simple_graph
        else:
            graph_for_nk = graph
        
        try:
            # First attempt: try with 'length' attribute
            self.nk_graph = nk.nxadapter.nx2nk(graph_for_nk, weightAttr='length')
            self.has_weights = True
            print("NetworKit: Successfully converted with 'length' weights")
        except KeyError as e:
            print(f"NetworKit: Some edges missing 'length' attribute, trying fallback conversion...")
            try:
                # Second attempt: convert without weights
                self.nk_graph = nk.nxadapter.nx2nk(graph_for_nk)
                self.has_weights = False
                print("NetworKit: Converted without edge weights (using topology only)")
            except Exception as e2:
                print(f"NetworKit: Conversion failed entirely: {e2}")
                raise ImportError(f"Failed to convert NetworkX graph to NetworKit: {e2}")
        except Exception as e:
            print(f"NetworKit: Unexpected conversion error: {e}")
            raise ImportError(f"Failed to convert NetworkX graph to NetworKit: {e}")
            
        # Create node mapping (NetworkX node IDs to NetworKit continuous IDs)
        self.nx_to_nk = {node: i for i, node in enumerate(graph.nodes())}
        self.nk_to_nx = {i: node for node, i in self.nx_to_nk.items()}
        
    def shortest_path_lengths(self, start_node: int, max_distance: float) -> Dict[int, float]:
        """Calculate shortest path lengths using NetworKit with caching optimization"""
        try:
            cache_key = (start_node, max_distance)
            
            # Check cache first
            if cache_key in self._shortest_path_cache:
                return self._shortest_path_cache[cache_key]
                
            if start_node not in self.nx_to_nk:
                print(f"NetworKit: Start node {start_node} not found")
                return {}
            
            # If we don't have proper edge weights, fallback to NetworkX for accurate distances
            if not self.has_weights:
                print("NetworKit: No edge weights available, falling back to NetworkX for distance calculation")
                try:
                    result = nx.single_source_dijkstra_path_length(
                        self.nx_graph, start_node, weight='length', cutoff=max_distance
                    )
                    
                    # Cache NetworkX fallback result too
                    if len(self._shortest_path_cache) >= self._max_cache_size:
                        oldest_key = next(iter(self._shortest_path_cache))
                        del self._shortest_path_cache[oldest_key]
                    self._shortest_path_cache[cache_key] = result
                    return result
                    
                except KeyError as e:
                    print(f"NetworkX fallback also failed: {e}")
                    return {}
            else:
                nk_start = self.nx_to_nk[start_node]
                
                # Use NetworKit's Dijkstra algorithm with weights
                dijkstra = nk.distance.Dijkstra(self.nk_graph, nk_start, storePaths=False)
                dijkstra.run()
                
                # Convert back to NetworkX node IDs and filter by distance
                result = {}
                for nk_node in range(self.nk_graph.numberOfNodes()):
                    distance = dijkstra.distance(nk_node)
                    if distance <= max_distance and distance != float('inf'):
                        nx_node = self.nk_to_nx[nk_node]
                        result[nx_node] = distance
            
            # Cache result before returning
            if len(self._shortest_path_cache) >= self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._shortest_path_cache))
                del self._shortest_path_cache[oldest_key]
            
            self._shortest_path_cache[cache_key] = result        
            return result
        except Exception as e:
            print(f"NetworKit error during Dijkstra: {e}")
            # Fallback to NetworkX if NetworKit fails
            print("Falling back to NetworkX due to NetworKit error")
            try:
                return nx.single_source_dijkstra_path_length(
                    self.nx_graph, start_node, weight='length', cutoff=max_distance
                )
            except Exception as e2:
                print(f"NetworkX fallback also failed: {e2}")
                return {}
    
    def connected_components(self, nodes: List[int]) -> List[List[int]]:
        """Find connected components using NetworKit"""
        try:
            # Create subgraph with only specified nodes
            nk_nodes = [self.nx_to_nk[node] for node in nodes if node in self.nx_to_nk]
            
            if not nk_nodes:
                return []
                
            # Create induced subgraph
            subgraph = nk.graphtools.subgraphFromNodes(self.nk_graph, nk_nodes)
            
            # Find connected components
            cc = nk.components.ConnectedComponents(subgraph)
            cc.run()
            
            # Convert back to NetworkX node IDs
            components = []
            component_map = cc.getPartition()
            
            # Group nodes by component
            component_dict = {}
            for i, nk_node in enumerate(nk_nodes):
                comp_id = component_map[i]
                if comp_id not in component_dict:
                    component_dict[comp_id] = []
                component_dict[comp_id].append(self.nk_to_nx[nk_node])
            
            return list(component_dict.values())
        except Exception as e:
            print(f"NetworKit error during connected components: {e}")
            # Fallback to NetworkX
            subgraph = self.nx_graph.subgraph(nodes)
            return [list(component) for component in nx.connected_components(subgraph)]
    
    def remove_edges(self, edges_to_remove: List[Tuple[int, int]]) -> 'NetworKitAdapter':
        """Create new NetworKit adapter with edges removed"""
        # For edge removal, we need to work with the NetworkX graph and recreate
        new_nx_graph = self.nx_graph.copy()
        for edge in edges_to_remove:
            if new_nx_graph.has_edge(edge[0], edge[1]):
                new_nx_graph.remove_edge(edge[0], edge[1])
        return NetworKitAdapter(new_nx_graph)

def validate_graph_for_networkit(graph: nx.Graph) -> Dict:
    """Validate graph edges for NetworKit compatibility"""
    total_edges = graph.number_of_edges()
    edges_with_length = 0
    edges_without_length = 0
    sample_edge_attrs = []
    
    for u, v, data in graph.edges(data=True):
        if 'length' in data:
            edges_with_length += 1
        else:
            edges_without_length += 1
            if len(sample_edge_attrs) < 3:  # Collect sample of problematic edges
                sample_edge_attrs.append((u, v, list(data.keys())))
        
        if edges_with_length + edges_without_length >= 1000:  # Sample first 1000 edges
            break
    
    result = {
        'total_edges_sampled': min(total_edges, 1000),
        'edges_with_length': edges_with_length,
        'edges_without_length': edges_without_length,
        'percentage_with_length': (edges_with_length / min(total_edges, 1000)) * 100,
        'sample_problematic_edges': sample_edge_attrs
    }
    
    print(f"Graph validation: {edges_with_length}/{min(total_edges, 1000)} edges have 'length' attribute ({result['percentage_with_length']:.1f}%)")
    if edges_without_length > 0:
        print(f"Sample edges without 'length': {sample_edge_attrs[:2]}")
    
    return result

def create_network_adapter(graph: nx.Graph, backend: str = "networkx") -> NetworkAdapter:
    """Factory function to create appropriate network adapter"""
    if backend.lower() == "networkit":
        if not NETWORKIT_AVAILABLE:
            print("Warning: NetworKit not available, falling back to NetworkX")
            return NetworkXAdapter(graph)
        
        # Add diagnostic validation
        validation = validate_graph_for_networkit(graph)
        
        try:
            return NetworKitAdapter(graph)
        except ImportError as e:
            print(f"Warning: NetworKit initialization failed ({e}), falling back to NetworkX")
            return NetworkXAdapter(graph)
    else:
        return NetworkXAdapter(graph)

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

# Global progress tracking
LOADING_PROGRESS = {}
MASS_ANALYSIS_PROGRESS = {}

class ParkAccessibilityAnalyzer:
    def __init__(self):
        self.cache = {}
        # Cache for polygon objects to avoid recreating them
        self._polygon_cache = {}
    
    def _get_or_create_polygon(self, area_result, points):
        """Get polygon from cache or create and cache it"""
        # Create a cache key based on the boundary or points
        if 'boundary' in area_result and area_result['boundary']:
            # Use boundary coordinates as cache key
            cache_key = tuple(tuple(coord) for coord in area_result['boundary'])
        else:
            # Use points as cache key
            if len(points) >= 3:
                cache_key = tuple(tuple(point) for point in points)
            else:
                return None
        
        # Check if polygon is already cached
        if cache_key in self._polygon_cache:
            return self._polygon_cache[cache_key]
        
        # Create polygon
        polygon = None
        if 'boundary' in area_result and len(area_result['boundary']) > 3:
            # Use boundary if available (more accurate)
            boundary_coords = [(coord[1], coord[0]) for coord in area_result['boundary']]
            polygon = Polygon(boundary_coords)
        else:
            # Fallback: create convex hull polygon from points
            if len(points) >= 3:
                hull_points = []
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                for vertex in hull.vertices:
                    # points are [lat, lng] format, convert to [lng, lat] for Shapely
                    hull_points.append((points[vertex][1], points[vertex][0]))
                polygon = Polygon(hull_points)
        
        # Cache the polygon
        if polygon is not None:
            self._polygon_cache[cache_key] = polygon
        
        return polygon
    
    def _extract_local_subgraph(self, graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                               center_node: int, max_distance: float) -> nx.Graph:
        """Extract a local subgraph around a center node within walking distance"""
        
        if center_node not in graph:
            raise ValueError(f"Center node {center_node} not found in graph")
        
        # Use Dijkstra-like approach to find all nodes within walking distance
        distances = {center_node: 0.0}
        visited = set()
        queue = [(0.0, center_node)]  # (distance, node)
        local_nodes = set([center_node])
        
        while queue:
            current_dist, current_node = min(queue, key=lambda x: x[0])
            queue.remove((current_dist, current_node))
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Explore neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in distances:
                    # Get edge length
                    edge_data = graph[current_node][neighbor]
                    edge_length = edge_data.get('length', 50)  # Default 50m if no length
                    
                    new_distance = current_dist + edge_length
                    
                    if new_distance <= max_distance:
                        distances[neighbor] = new_distance
                        local_nodes.add(neighbor)
                        queue.append((new_distance, neighbor))
        
        # Extract subgraph with only the local nodes
        subgraph = graph.subgraph(local_nodes).copy()
        
        return subgraph
    
    def _filter_amenities_for_subgraph(self, subgraph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                                     amenities_gdf: gpd.GeoDataFrame, buffer_meters: float = 100) -> gpd.GeoDataFrame:
        """Filter amenities to only those near the subgraph area"""
        if amenities_gdf is None or amenities_gdf.empty:
            return amenities_gdf
        
        # Get bounds of the subgraph nodes
        subgraph_node_ids = list(subgraph.nodes())
        if not subgraph_node_ids:
            return gpd.GeoDataFrame()  # Empty result
        
        # Filter nodes_gdf to subgraph nodes and get their bounds
        subgraph_nodes_gdf = nodes_gdf.loc[nodes_gdf.index.intersection(subgraph_node_ids)]
        
        if subgraph_nodes_gdf.empty:
            return gpd.GeoDataFrame()
        
        # Get bounding box of subgraph nodes
        bounds = subgraph_nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # Add buffer (convert meters to approximate degrees)
        buffer_deg = buffer_meters / 111000  # Very rough conversion: 1 degree ≈ 111km
        buffered_bounds = [
            bounds[0] - buffer_deg,  # minx
            bounds[1] - buffer_deg,  # miny  
            bounds[2] + buffer_deg,  # maxx
            bounds[3] + buffer_deg   # maxy
        ]
        
        # Filter amenities to those within the buffered bounds
        amenities_in_bounds = amenities_gdf.cx[
            buffered_bounds[0]:buffered_bounds[2],  # longitude range
            buffered_bounds[1]:buffered_bounds[3]   # latitude range
        ]
        
        return amenities_in_bounds
    
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
        """Synchronous data loading with progress tracking"""
        print(f"Starting data load for {lat}, {lng}, {radius}")
        
        # Create unique progress key for this request
        progress_key = f"{lat:.4f}_{lng:.4f}_{radius}"
        
        total_steps = 15
        current_step = 0
        
        def log_progress(step_name: str):
            nonlocal current_step
            current_step += 1
            progress_info = {
                'current_step': current_step,
                'total_steps': total_steps,
                'step_name': step_name,
                'sub_progress': 0,
                'current_operation': step_name,
                'estimated_time_remaining': None,
                'completed': False,
                'start_time': __import__('time').time() if current_step == 1 else LOADING_PROGRESS.get(progress_key, {}).get('start_time', __import__('time').time())
            }
            LOADING_PROGRESS[progress_key] = progress_info
            print(f"Step {current_step}/{total_steps}: {step_name}")
        
        def update_sub_progress(sub_progress: int, operation: str, estimate_remaining: float = None):
            """Update sub-progress within the current step"""
            if progress_key in LOADING_PROGRESS:
                LOADING_PROGRESS[progress_key]['sub_progress'] = sub_progress
                LOADING_PROGRESS[progress_key]['current_operation'] = operation
                if estimate_remaining:
                    LOADING_PROGRESS[progress_key]['estimated_time_remaining'] = estimate_remaining
                print(f"  → {operation} ({sub_progress}%)")
        
        try:
            # Step 1: Load parks from OpenStreetMap
            log_progress("Loading parks from OpenStreetMap...")
            tags = {"leisure": ["park", "recreation_ground"], "landuse": ["grass", "recreation_ground"]}
            parks = ox.features_from_point((lat, lng), tags=tags, dist=radius)
            # Ensure WGS84 CRS for consistency
            parks = parks.to_crs('EPSG:4326')
            print(f"Raw parks loaded: {len(parks)} (CRS: {parks.crs})")
            
            # Step 2: Filter and process park geometries
            log_progress("Filtering park geometries...")
            parks = parks[parks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            print(f"Filtered parks: {len(parks)}")
            
            if not parks.empty:
                parks["area_m2"] = parks.geometry.to_crs(3857).area
                print(f"Final parks: {len(parks)}")
            
            # Step 3: Load street network from OpenStreetMap
            log_progress("Loading street network from OpenStreetMap...")
            import time
            network_start = time.time()
            
            # Estimate time based on radius (rough estimates)
            if radius >= 4000:
                estimated_time = 120  # 2 minutes for 4km
                update_sub_progress(10, "Downloading large street network (this may take 2-3 minutes)...", estimated_time)
            elif radius >= 2000:
                estimated_time = 30   # 30 seconds for 2km  
                update_sub_progress(10, "Downloading street network...", estimated_time)
            else:
                update_sub_progress(10, "Downloading street network...")
            
            graph = ox.graph_from_point((lat, lng), dist=radius, network_type='walk')
            network_time = time.time() - network_start
            print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges (took {network_time:.1f}s)")
            
            # Step 4: Calculate edge lengths and weights
            log_progress("Calculating street lengths and weights...")
            graph = ox.distance.add_edge_lengths(graph)
            print("Edge lengths added")
                
            # Step 5: Convert to undirected graph
            log_progress("Converting to undirected graph...")
            graph_undirected = graph.to_undirected()
            print("Converted to undirected")
            
            # Step 5b: Ensure all edges in undirected graph have lengths
            log_progress("Validating edge lengths in undirected graph...")
            missing_lengths = 0
            total_edges = 0
            for u, v, data in graph_undirected.edges(data=True):
                total_edges += 1
                if 'length' not in data:
                    missing_lengths += 1
                    # Calculate length from node coordinates as fallback
                    try:
                        node_u = graph_undirected.nodes[u]
                        node_v = graph_undirected.nodes[v]
                        if 'x' in node_u and 'y' in node_u and 'x' in node_v and 'y' in node_v:
                            # Calculate Euclidean distance as approximation
                            import math
                            dx = node_v['x'] - node_u['x']
                            dy = node_v['y'] - node_u['y']
                            # Convert degrees to approximate meters (rough approximation)
                            # 1 degree ≈ 111,000 meters at equator
                            distance_m = math.sqrt(dx*dx + dy*dy) * 111000
                            graph_undirected[u][v]['length'] = distance_m
                    except Exception as e:
                        # Last resort: set a default length
                        graph_undirected[u][v]['length'] = 50.0  # 50 meter default
            
            if missing_lengths > 0:
                print(f"Fixed {missing_lengths}/{total_edges} edges missing length attributes")
            else:
                print(f"All {total_edges} edges have length attributes")
            
            # Step 6: Create geographic data frames
            log_progress("Creating geographic data frames...")
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
            # Ensure consistent CRS for all geodataframes
            nodes_gdf = nodes_gdf.to_crs('EPSG:4326')
            edges_gdf = edges_gdf.to_crs('EPSG:4326')
            print(f"GDFs created: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges (CRS: {nodes_gdf.crs})")
            
            # Step 7: Process parks for visualization
            log_progress("Processing parks for web display...")
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
            
            # Step 8: Process nodes for web interface
            log_progress("Processing street nodes for selection...")
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
            
            # Step 9: Load supermarkets from OpenStreetMap
            # Use larger radius for amenities to capture walkable POIs beyond park area
            poi_radius = radius * 2  # Double the radius for POI search
            log_progress("Loading supermarkets from OpenStreetMap...")
            
            # Provide time estimates for amenity loading
            if poi_radius >= 8000:
                update_sub_progress(20, "Downloading supermarkets data (large area - may take 30-60s)...")
            elif poi_radius >= 4000:
                update_sub_progress(20, "Downloading supermarkets data...")
            else:
                update_sub_progress(20, "Downloading supermarkets...")
            
            try:
                amenity_start = time.time()
                supermarkets_gdf = ox.features_from_point(
                    (lat, lng), 
                    dist=poi_radius,
                    tags={'shop': ['supermarket', 'grocery', 'convenience', 'department_store', 'general', 'food'], 'amenity': ['marketplace', 'food_court']}
                )
                if not supermarkets_gdf.empty:
                    # Ensure WGS84 CRS for consistency
                    supermarkets_gdf = supermarkets_gdf.to_crs('EPSG:4326')
                amenity_time = time.time() - amenity_start
                print(f"Loaded {len(supermarkets_gdf) if not supermarkets_gdf.empty else 0} supermarkets (CRS: {supermarkets_gdf.crs if not supermarkets_gdf.empty else 'N/A'}, took {amenity_time:.1f}s)")
            except Exception as e:
                print(f"Failed to load supermarkets: {e}")
                import geopandas as gpd
                supermarkets_gdf = gpd.GeoDataFrame()
            
            # Step 10: Load schools from OpenStreetMap
            log_progress("Loading schools from OpenStreetMap...")
            try:
                schools_gdf = ox.features_from_point((lat, lng), dist=poi_radius, tags={'amenity': 'school'})
                if not schools_gdf.empty:
                    schools_gdf = schools_gdf.to_crs('EPSG:4326')
                print(f"Loaded {len(schools_gdf) if not schools_gdf.empty else 0} schools (CRS: {schools_gdf.crs if not schools_gdf.empty else 'N/A'})")
            except Exception as e:
                print(f"Failed to load schools: {e}")
                import geopandas as gpd
                schools_gdf = gpd.GeoDataFrame()
            
            # Step 11: Load playgrounds from OpenStreetMap
            log_progress("Loading playgrounds from OpenStreetMap...")
            try:
                playgrounds_gdf = ox.features_from_point((lat, lng), dist=poi_radius, tags={'leisure': 'playground'})
                if not playgrounds_gdf.empty:
                    playgrounds_gdf = playgrounds_gdf.to_crs('EPSG:4326')
                print(f"Loaded {len(playgrounds_gdf) if not playgrounds_gdf.empty else 0} playgrounds (CRS: {playgrounds_gdf.crs if not playgrounds_gdf.empty else 'N/A'})")
            except Exception as e:
                print(f"Failed to load playgrounds: {e}")
                import geopandas as gpd
                playgrounds_gdf = gpd.GeoDataFrame()
            
            # Step 12: Load cafes/bars from OpenStreetMap
            log_progress("Loading cafes and bars from OpenStreetMap...")
            
            # Cafes/bars can be very numerous and slow to load for large areas
            if poi_radius >= 8000:
                update_sub_progress(40, "Downloading cafes/bars data (large area - this is often the slowest step, may take 1-2 minutes)...")
            elif poi_radius >= 4000:
                update_sub_progress(40, "Downloading cafes/bars data (may take 30-60s)...")
            else:
                update_sub_progress(40, "Downloading cafes/bars...")
            
            try:
                cafes_start = time.time()
                cafes_bars_gdf = ox.features_from_point(
                    (lat, lng), 
                    dist=poi_radius, 
                    tags={'amenity': ['cafe', 'bar', 'pub', 'restaurant'], 'shop': ['coffee']}
                )
                if not cafes_bars_gdf.empty:
                    cafes_bars_gdf = cafes_bars_gdf.to_crs('EPSG:4326')
                cafes_time = time.time() - cafes_start
                print(f"Loaded {len(cafes_bars_gdf) if not cafes_bars_gdf.empty else 0} cafes/bars (CRS: {cafes_bars_gdf.crs if not cafes_bars_gdf.empty else 'N/A'}, took {cafes_time:.1f}s)")
            except Exception as e:
                print(f"Failed to load cafes/bars: {e}")
                import geopandas as gpd
                cafes_bars_gdf = gpd.GeoDataFrame()
            
            # Step 13: Load public transit from OpenStreetMap
            log_progress("Loading public transit stops from OpenStreetMap...")
            
            if poi_radius >= 8000:
                update_sub_progress(60, "Downloading transit data (large area - may take 30-60s)...")
            else:
                update_sub_progress(60, "Downloading transit data...")
                
            try:
                transit_gdf = ox.features_from_point(
                    (lat, lng), 
                    dist=poi_radius, 
                    tags={'public_transport': ['station', 'stop_position', 'platform'], 'railway': ['station', 'halt', 'tram_stop'], 'highway': 'bus_stop'}
                )
                if not transit_gdf.empty:
                    transit_gdf = transit_gdf.to_crs('EPSG:4326')
                print(f"Loaded {len(transit_gdf) if not transit_gdf.empty else 0} transit stops (CRS: {transit_gdf.crs if not transit_gdf.empty else 'N/A'})")
            except Exception as e:
                print(f"Failed to load transit stops: {e}")
                import geopandas as gpd
                transit_gdf = gpd.GeoDataFrame()
            
            print(f"Amenities loaded (radius {poi_radius}m): {len(supermarkets_gdf)} supermarkets, {len(schools_gdf)} schools, {len(playgrounds_gdf)} playgrounds, {len(cafes_bars_gdf)} cafes/bars, {len(transit_gdf)} transit stops")
            
            # Debug: Show first few entries of each amenity type if available
            if len(supermarkets_gdf) > 0:
                print(f"Sample supermarket: {supermarkets_gdf.iloc[0].get('name', 'Unnamed')} at ({supermarkets_gdf.iloc[0].geometry.centroid.y:.6f}, {supermarkets_gdf.iloc[0].geometry.centroid.x:.6f})")
            if len(schools_gdf) > 0:
                print(f"Sample school: {schools_gdf.iloc[0].get('name', 'Unnamed')} at ({schools_gdf.iloc[0].geometry.centroid.y:.6f}, {schools_gdf.iloc[0].geometry.centroid.x:.6f})")
            if len(transit_gdf) > 0:
                print(f"Sample transit: {transit_gdf.iloc[0].get('name', 'Unnamed')} at ({transit_gdf.iloc[0].geometry.centroid.y:.6f}, {transit_gdf.iloc[0].geometry.centroid.x:.6f})")
            
            # Step 14: Create spatial indexes for fast amenity lookup during mass analysis
            log_progress("Creating spatial indexes for amenities...")
            update_sub_progress(80, "Building spatial indexes for fast amenity filtering...")
            try:
                from rtree import index
                print("✓ rtree library available for spatial indexing")
                # Create spatial indexes for each amenity type
                amenity_indexes = {}
                amenity_data = {
                    'supermarkets': supermarkets_gdf,
                    'schools': schools_gdf, 
                    'playgrounds': playgrounds_gdf,
                    'cafes_bars': cafes_bars_gdf,
                    'transit': transit_gdf
                }
                
                for amenity_type, gdf in amenity_data.items():
                    if not gdf.empty:
                        spatial_idx = index.Index()
                        for i, (idx, row) in enumerate(gdf.iterrows()):
                            geom = row.geometry
                            if hasattr(geom, 'centroid'):
                                point = geom.centroid
                            else:
                                point = geom
                            # Insert (integer_id, (minx, miny, maxx, maxy))
                            # Store mapping from integer id to original index
                            spatial_idx.insert(i, (point.x, point.y, point.x, point.y))
                        amenity_indexes[amenity_type] = {'index': spatial_idx, 'id_mapping': dict(enumerate(gdf.index))}
                        print(f"✓ Created spatial index for {len(gdf)} {amenity_type}")
                    else:
                        amenity_indexes[amenity_type] = None
                        print(f"✗ No {amenity_type} data, spatial index set to None")
                        
            except ImportError:
                print("❌ rtree not available, falling back to brute force spatial filtering")
                print("   This may cause performance issues on Heroku deployments")
                amenity_indexes = {}
            except Exception as e:
                print(f"❌ Failed to create spatial indexes: {e}")
                print("   This may cause performance issues on Heroku deployments")
                amenity_indexes = {}
            
            
            # Mark progress as completed
            LOADING_PROGRESS[progress_key] = {
                'current_step': total_steps,
                'total_steps': total_steps,
                'step_name': 'Data loading complete!',
                'completed': True
            }
            
            # Clean up old progress entries to prevent memory leaks
            import time
            current_time = time.time()
            for key in list(LOADING_PROGRESS.keys()):
                progress = LOADING_PROGRESS[key]
                if progress.get('completed') and not hasattr(progress, 'cleanup_time'):
                    progress['cleanup_time'] = current_time
                elif progress.get('cleanup_time') and (current_time - progress['cleanup_time']) > 300:  # 5 minutes
                    del LOADING_PROGRESS[key]
            
            return {
                'parks': parks_data,
                # 'nodes': nodes_data[:200],  # Limit nodes for UI performance
                'nodes': nodes_data,  # Limit nodes for UI performance
                'graph': graph_undirected,
                'parks_gdf': parks,
                'nodes_gdf': nodes_gdf,
                'edges_gdf': edges_gdf,
                'center': [lat, lng],
                'supermarkets_gdf': supermarkets_gdf,
                'schools_gdf': schools_gdf,
                'playgrounds_gdf': playgrounds_gdf,
                'cafes_bars_gdf': cafes_bars_gdf,
                'transit_gdf': transit_gdf,
                'amenity_indexes': amenity_indexes  # Add spatial indexes for fast lookup
            }
            
        except Exception as e:
            # Mark progress as failed
            LOADING_PROGRESS[progress_key] = {
                'current_step': current_step,
                'total_steps': total_steps,
                'step_name': f'Error: {str(e)}',
                'completed': True,
                'error': True
            }
            print(f"Exception in _load_data_sync: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load area data: {str(e)}")
    
    async def calculate_accessibility(self, lat: float, lng: float, radius: int, 
                                   park_id: str, node_id: str, walk_time: float = 10.0, 
                                   walk_speed: float = 4.5, viz_method: str = "convex_hull",
                                   network_backend: str = "networkit") -> Dict:
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
            data, park_id, node_id, walk_time, walk_speed, viz_method, network_backend
        )
        
        return result
    
    async def calculate_mass_accessibility(self, lat: float, lng: float, radius: int, 
                                         park_id: str, node_ids: List[str], walk_time: float = 10.0, 
                                         walk_speed: float = 4.5, viz_method: str = "convex_hull",
                                         network_backend: str = "networkit") -> Dict:
        """Calculate accessibility for multiple nodes in parallel with shared resources"""
        # Get cached data
        cache_key = f"{lat:.4f}_{lng:.4f}_{radius}"
        if cache_key not in DATA_CACHE:
            await self.load_area_data(lat, lng, radius)
        
        data = DATA_CACHE[cache_key]
        
        # Generate progress key immediately
        import time
        progress_key = f"mass_{park_id}_{int(time.time())}"
        
        # Initialize progress tracking
        MASS_ANALYSIS_PROGRESS[progress_key] = {
            'total_nodes': len(node_ids),
            'completed_nodes': 0,
            'current_node': None,
            'completed': False,
            'start_time': time.time()
        }
        
        # Run batch analysis in thread pool
        print(f"Starting mass analysis thread pool execution for {len(node_ids)} nodes")
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor, self._calculate_mass_accessibility_sync, 
                data, park_id, node_ids, walk_time, walk_speed, viz_method, network_backend, progress_key
            )
            print(f"Mass analysis thread pool completed successfully with {len(results) if results else 0} results")
        except Exception as e:
            print(f"Mass analysis thread pool failed: {e}")
            raise
        
        return results
    
    async def calculate_mass_accessibility_background(self, lat: float, lng: float, radius: int, 
                                         park_id: str, node_ids: List[str], walk_time: float = 10.0, 
                                         walk_speed: float = 4.5, viz_method: str = "convex_hull",
                                         network_backend: str = "networkit", progress_key: str = None):
        """Background task for mass accessibility calculation"""
        try:
            # Get cached data
            cache_key = f"{lat:.4f}_{lng:.4f}_{radius}"
            if cache_key not in DATA_CACHE:
                await self.load_area_data(lat, lng, radius)
            
            data = DATA_CACHE[cache_key]
            
            # Run batch analysis in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor, self._calculate_mass_accessibility_sync, 
                data, park_id, node_ids, walk_time, walk_speed, viz_method, network_backend, progress_key
            )
            
            # Store results in progress tracking
            MASS_ANALYSIS_PROGRESS[progress_key]['results'] = results
            
        except Exception as e:
            print(f"Background task error: {e}")
            import traceback
            traceback.print_exc()
            # Mark progress as failed
            if progress_key and progress_key in MASS_ANALYSIS_PROGRESS:
                MASS_ANALYSIS_PROGRESS[progress_key]['completed'] = True
                MASS_ANALYSIS_PROGRESS[progress_key]['error'] = str(e)
    
    def _calculate_mass_accessibility_sync(self, data: Dict, park_id: str, node_ids: List[str],
                                         walk_time: float, walk_speed: float, viz_method: str,
                                         network_backend: str, progress_key: str) -> List[Dict]:
        """Synchronous mass accessibility calculation with parallel processing and shared resources"""
        print(f"Starting mass accessibility calculation for park {park_id}, {len(node_ids)} nodes")
        
        import time
        
        try:
            graph = data['graph']
            parks_gdf = data['parks_gdf']
            nodes_gdf = data['nodes_gdf']
            
            # Validate park exists
            park_key = None
            for idx, park_data in parks_gdf.iterrows():
                if str(idx) == park_id:
                    park_key = idx
                    break
            
            if park_key is None:
                raise KeyError(f"Park {park_id} not found in parks_gdf")
            
            # Get park geometry for dynamic filtering
            park_geom = parks_gdf.loc[park_key].geometry
            print(f"Using dynamic subgraph approach - no preprocessing required")
            
            # Walking parameters
            walking_speed_ms = (walk_speed * 1000) / 60  # km/h to m/min
            max_distance = walk_time * walking_speed_ms  # walking time in minutes
            
            # Process nodes in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = []
            
            def process_single_node(node_id: str) -> Dict:
                """Process a single node using dynamic subgraphs"""
                try:
                    # Validate node exists
                    node_key = int(node_id)
                    if node_key not in nodes_gdf.index:
                        raise KeyError(f"Node {node_id} not found in nodes_gdf")
                    
                    if node_key not in graph:
                        raise Exception(f"Node {node_key} not found in graph")
                    
                    print(f"Processing node {node_id}...")
                    
                    # Memory check before processing (important for Heroku)
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > 450:  # Heroku free tier ~512MB limit
                            print(f"WARNING: High memory usage before node {node_id}: {memory_mb:.1f} MB")
                    except ImportError:
                        pass  # psutil not available
                    
                    # Extract dynamic subgraph around this node
                    local_subgraph = self._extract_local_subgraph(graph, nodes_gdf, node_key, max_distance)
                    
                    # Filter amenities to the local area
                    local_supermarkets = self._filter_amenities_for_subgraph(local_subgraph, nodes_gdf, data.get('supermarkets_gdf'))
                    local_schools = self._filter_amenities_for_subgraph(local_subgraph, nodes_gdf, data.get('schools_gdf'))
                    local_playgrounds = self._filter_amenities_for_subgraph(local_subgraph, nodes_gdf, data.get('playgrounds_gdf'))
                    local_cafes_bars = self._filter_amenities_for_subgraph(local_subgraph, nodes_gdf, data.get('cafes_bars_gdf'))
                    local_transit = self._filter_amenities_for_subgraph(local_subgraph, nodes_gdf, data.get('transit_gdf'))
                    
                    # Create simple NetworkX adapter for the local subgraph (no NetworKit conversion needed)
                    local_adapter = NetworkXAdapter(local_subgraph)
                    
                    # Calculate WITH park using local subgraph
                    with_park = self._calculate_isochrone(local_adapter, nodes_gdf, data['edges_gdf'], 
                                                        node_key, max_distance, viz_method,
                                                        supermarkets_gdf=local_supermarkets,
                                                        schools_gdf=local_schools,
                                                        playgrounds_gdf=local_playgrounds,
                                                        cafes_bars_gdf=local_cafes_bars,
                                                        transit_gdf=local_transit,
                                                        debug_label="WITH_PARK")
                    
                    # Calculate WITHOUT park - remove park edges from local subgraph
                    park_edges_in_subgraph = self._get_park_edges(local_subgraph, nodes_gdf, park_geom)
                    without_park_subgraph = local_adapter.remove_edges(park_edges_in_subgraph)
                    
                    without_park = self._calculate_isochrone(without_park_subgraph, nodes_gdf, data['edges_gdf'],
                                                           node_key, max_distance, viz_method,
                                                           supermarkets_gdf=local_supermarkets,
                                                           schools_gdf=local_schools,
                                                           playgrounds_gdf=local_playgrounds,
                                                           cafes_bars_gdf=local_cafes_bars,
                                                           transit_gdf=local_transit,
                                                           debug_label="WITHOUT_PARK")
                    
                    # Calculate metrics (same as single node analysis)
                    area_difference = without_park['area_km2'] - with_park['area_km2']
                    street_network_difference = without_park['street_network_stats']['total_length_km'] - with_park['street_network_stats']['total_length_km']
                    node_difference = without_park['reachable_nodes'] - with_park['reachable_nodes']
                    edge_difference = without_park['street_network_stats']['edge_count'] - with_park['street_network_stats']['edge_count']
                    
                    # Calculate percentage changes
                    area_change_pct = (area_difference / with_park['area_km2'] * 100) if with_park['area_km2'] > 0 else 0
                    street_network_change_pct = (street_network_difference / with_park['street_network_stats']['total_length_km'] * 100) if with_park['street_network_stats']['total_length_km'] > 0 else 0
                    
                    # Determine impact category
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
                        'node_id': node_id,
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
                            'without_street_density': without_park['street_network_stats']['street_density_km_per_km2'],
                            'with_supermarkets_accessible': with_park.get('accessible_supermarkets', 0),
                            'without_supermarkets_accessible': without_park.get('accessible_supermarkets', 0),
                            'with_schools_accessible': with_park.get('accessible_schools', 0),
                            'without_schools_accessible': without_park.get('accessible_schools', 0),
                            'with_playgrounds_accessible': with_park.get('accessible_playgrounds', 0),
                            'without_playgrounds_accessible': without_park.get('accessible_playgrounds', 0),
                            'with_cafes_bars_accessible': with_park.get('accessible_cafes_bars', 0),
                            'without_cafes_bars_accessible': without_park.get('accessible_cafes_bars', 0),
                            'with_transit_accessible': with_park.get('accessible_transit', 0),
                            'without_transit_accessible': without_park.get('accessible_transit', 0)
                        }
                    }
                except Exception as e:
                    print(f"Error processing node {node_id}: {e}")
                    return {
                        'node_id': node_id,
                        'error': str(e),
                        'park_id': park_id
                    }
            
            # Process nodes in parallel with enhanced threading
            # Use more workers for CPU-bound operations, but respect system limits
            import os
            cpu_count = os.cpu_count() or 4
            max_workers = min(max(cpu_count - 1, 4), 12, len(node_ids))  # Dynamic worker count
            print(f"Using {max_workers} parallel workers (CPU cores: {cpu_count})")
            with ThreadPoolExecutor(max_workers=max_workers) as parallel_executor:
                future_to_node = {parallel_executor.submit(process_single_node, node_id): node_id 
                                for node_id in node_ids}
                
                for future in as_completed(future_to_node):
                    node_id = future_to_node[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress
                        MASS_ANALYSIS_PROGRESS[progress_key]['completed_nodes'] = len(results)
                        MASS_ANALYSIS_PROGRESS[progress_key]['current_node'] = node_id
                        
                        print(f"Completed node {node_id} ({len(results)}/{len(node_ids)})")
                    except Exception as e:
                        print(f"Node {node_id} failed: {e}")
                        import traceback
                        print(f"Node {node_id} traceback:")
                        traceback.print_exc()
                        
                        # Add detailed error info for node failure debugging
                        node_error_details = {
                            'node_id': node_id,
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'graph_nodes': len(graph) if 'graph' in locals() else 'unknown',
                            'nodes_gdf_size': len(nodes_gdf) if 'nodes_gdf' in locals() else 'unknown',
                            'park_exists': park_key is not None if 'park_key' in locals() else 'unknown'
                        }
                        print(f"Node {node_id} error details: {node_error_details}")
                        
                        results.append({
                            'node_id': node_id,
                            'error': str(e),
                            'park_id': park_id,
                            'error_details': node_error_details
                        })
                        
                        # Update progress even for failed nodes
                        MASS_ANALYSIS_PROGRESS[progress_key]['completed_nodes'] = len(results)
                        MASS_ANALYSIS_PROGRESS[progress_key]['current_node'] = f"{node_id} (failed)"
            
            # Mark progress as completed
            MASS_ANALYSIS_PROGRESS[progress_key]['completed'] = True
            MASS_ANALYSIS_PROGRESS[progress_key]['current_node'] = "Complete"
            
            print(f"Mass accessibility calculation complete: {len(results)} results")
            
            # Memory cleanup and monitoring
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"Memory usage after mass analysis: {memory_mb:.1f} MB")
                
                # Clean up large temporary variables
                del future_to_node
                if 'with_park_adapter' in locals():
                    del with_park_adapter
                if 'without_park_adapter' in locals():
                    del without_park_adapter
                    
            except ImportError:
                print("psutil not available for memory monitoring")
            except Exception as e:
                print(f"Memory monitoring error: {e}")
            
            # Return just the results array
            return results
            
        except Exception as e:
            # Mark progress as failed
            if 'progress_key' in locals():
                MASS_ANALYSIS_PROGRESS[progress_key]['completed'] = True
                MASS_ANALYSIS_PROGRESS[progress_key]['error'] = str(e)
            
            print(f"Exception in _calculate_mass_accessibility_sync: {e}")
            import traceback
            traceback.print_exc()
            
            # Add detailed error information for debugging deployment issues
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'park_id': park_id,
                'node_ids_count': len(node_ids) if 'node_ids' in locals() else 'unknown',
                'available_parks': len(parks_gdf) if 'parks_gdf' in locals() else 'unknown'
            }
            print(f"Mass analysis error details: {error_details}")
            
            raise Exception(f"Failed to calculate mass accessibility: {str(e)} | Details: {error_details}")
    
    def _calculate_accessibility_sync(self, data: Dict, park_id: str, node_id: str, 
                                     walk_time: float, walk_speed: float, viz_method: str,
                                     network_backend: str = "networkit") -> Dict:
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
            print(f"Using network backend: {network_backend}")
            
            # Get park geometry for enhanced analysis
            park_geom = parks_gdf.loc[park_key].geometry
            
            # Create network adapter for the selected backend
            network_adapter = create_network_adapter(graph, network_backend)
            
            # Calculate WITH park
            with_park = self._calculate_isochrone(network_adapter, nodes_gdf, data['edges_gdf'], node_key, max_distance, viz_method,
                                                supermarkets_gdf=data.get('supermarkets_gdf'),
                                                schools_gdf=data.get('schools_gdf'),
                                                playgrounds_gdf=data.get('playgrounds_gdf'),
                                                cafes_bars_gdf=data.get('cafes_bars_gdf'),
                                                transit_gdf=data.get('transit_gdf'),
                                                amenity_indexes=data.get('amenity_indexes'))
            
            print("Calculating WITHOUT park...")
            # Calculate WITHOUT park - remove park edges and create new adapter
            edges_to_remove = self._get_park_edges(graph, nodes_gdf, park_geom)
            filtered_adapter = network_adapter.remove_edges(edges_to_remove)
            without_park = self._calculate_isochrone(filtered_adapter, nodes_gdf, data['edges_gdf'], node_key, max_distance, viz_method,
                                                   supermarkets_gdf=data.get('supermarkets_gdf'),
                                                   schools_gdf=data.get('schools_gdf'),
                                                   playgrounds_gdf=data.get('playgrounds_gdf'),
                                                   cafes_bars_gdf=data.get('cafes_bars_gdf'),
                                                   transit_gdf=data.get('transit_gdf'),
                                                   amenity_indexes=data.get('amenity_indexes'))
            
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
                    'without_street_density': without_park['street_network_stats']['street_density_km_per_km2'],
                    'with_supermarkets_accessible': with_park.get('accessible_supermarkets', 0),
                    'without_supermarkets_accessible': without_park.get('accessible_supermarkets', 0),
                    'with_schools_accessible': with_park.get('accessible_schools', 0),
                    'without_schools_accessible': without_park.get('accessible_schools', 0),
                    'with_playgrounds_accessible': with_park.get('accessible_playgrounds', 0),
                    'without_playgrounds_accessible': without_park.get('accessible_playgrounds', 0),
                    'with_cafes_bars_accessible': with_park.get('accessible_cafes_bars', 0),
                    'without_cafes_bars_accessible': without_park.get('accessible_cafes_bars', 0),
                    'with_transit_accessible': with_park.get('accessible_transit', 0),
                    'without_transit_accessible': without_park.get('accessible_transit', 0)
                }
            }
            
        except Exception as e:
            print(f"Exception in _calculate_accessibility_sync: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to calculate accessibility: {str(e)}")
    
    def _calculate_isochrone(self, network_adapter: NetworkAdapter, nodes_gdf: gpd.GeoDataFrame, 
                           edges_gdf: gpd.GeoDataFrame, start_node: int, max_distance: float, 
                           viz_method: str = "convex_hull", park_geometry=None, 
                           supermarkets_gdf=None, schools_gdf=None, playgrounds_gdf=None, 
                           cafes_bars_gdf=None, transit_gdf=None, debug_label="", amenity_indexes=None) -> Dict:
        """Calculate isochrone for a node with enhanced metrics"""
        if start_node not in network_adapter.nx_graph:
            print(f"Start node {start_node} not found in graph")
            return self._empty_isochrone_result()
        
        try:
            print(f"Finding reachable nodes from {start_node} within {max_distance}m")
            
            # Find reachable nodes using network adapter
            lengths = network_adapter.shortest_path_lengths(start_node, max_distance)
            
            if not lengths:
                print("No reachable nodes found")
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
            
            # Analyze connectivity using network adapter
            connected_components = network_adapter.connected_components(reachable_nodes)
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
                area_result = self._create_buffered_network_isochrone(network_adapter.nx_graph, edges_gdf, reachable_nodes, max_distance)
            else:
                points_array = np.array(points)
                area_result = self._create_convex_hull_isochrone(points_array, reachable_nodes)
            
            # Calculate reachable street network metrics
            street_network_stats = self._calculate_street_network_stats(network_adapter.nx_graph, edges_gdf, reachable_nodes, area_result['area_km2'])
            
            # Calculate accessible amenities within the isochrone using pre-loaded data
            print(f"{debug_label} Isochrone calculation - Received amenity data: supermarkets={len(supermarkets_gdf) if supermarkets_gdf is not None else 0} (empty={supermarkets_gdf.empty if supermarkets_gdf is not None else 'N/A'}), schools={len(schools_gdf) if schools_gdf is not None else 0} (empty={schools_gdf.empty if schools_gdf is not None else 'N/A'}), playgrounds={len(playgrounds_gdf) if playgrounds_gdf is not None else 0} (empty={playgrounds_gdf.empty if playgrounds_gdf is not None else 'N/A'}), cafes_bars={len(cafes_bars_gdf) if cafes_bars_gdf is not None else 0} (empty={cafes_bars_gdf.empty if cafes_bars_gdf is not None else 'N/A'}), transit={len(transit_gdf) if transit_gdf is not None else 0} (empty={transit_gdf.empty if transit_gdf is not None else 'N/A'})")
            
            # Use proper null checking to avoid GeoDataFrame boolean ambiguity
            # Get spatial indexes for optimization
            supermarkets_idx = amenity_indexes.get('supermarkets') if amenity_indexes else None
            schools_idx = amenity_indexes.get('schools') if amenity_indexes else None
            playgrounds_idx = amenity_indexes.get('playgrounds') if amenity_indexes else None
            cafes_bars_idx = amenity_indexes.get('cafes_bars') if amenity_indexes else None
            transit_idx = amenity_indexes.get('transit') if amenity_indexes else None
            
            accessible_supermarkets = self._calculate_accessible_supermarkets(area_result, points, supermarkets_gdf if supermarkets_gdf is not None and not supermarkets_gdf.empty else None, debug_label, supermarkets_idx)
            accessible_schools = self._calculate_accessible_schools(area_result, points, schools_gdf if schools_gdf is not None and not schools_gdf.empty else None, debug_label, schools_idx)
            accessible_playgrounds = self._calculate_accessible_playgrounds(area_result, points, playgrounds_gdf if playgrounds_gdf is not None and not playgrounds_gdf.empty else None, debug_label, playgrounds_idx)
            accessible_cafes_bars = self._calculate_accessible_cafes_bars(area_result, points, cafes_bars_gdf if cafes_bars_gdf is not None and not cafes_bars_gdf.empty else None, debug_label, cafes_bars_idx)
            accessible_transit = self._calculate_accessible_transit(area_result, points, transit_gdf if transit_gdf is not None and not transit_gdf.empty else None, debug_label, transit_idx)
            
            # Compile enhanced result
            result = {
                **area_result,
                'distance_stats': distance_stats,
                'connectivity_stats': connectivity_stats,
                'street_network_stats': street_network_stats,
                'node_coordinates': node_coords,
                'reachable_distances': dict(zip(reachable_nodes, distances)),
                'accessible_supermarkets': accessible_supermarkets,
                'accessible_schools': accessible_schools,
                'accessible_playgrounds': accessible_playgrounds,
                'accessible_cafes_bars': accessible_cafes_bars,
                'accessible_transit': accessible_transit
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
            'reachable_distances': {},
            'accessible_supermarkets': 0,
            'accessible_schools': 0,
            'accessible_playgrounds': 0,
            'accessible_cafes_bars': 0,
            'accessible_transit': 0
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
    
    def _calculate_accessible_supermarkets(self, area_result, points, supermarkets_gdf, debug_label="", spatial_index=None):
        """Calculate number of supermarkets accessible within the isochrone area using pre-loaded data"""
        return self._filter_amenities_in_polygon(area_result, points, supermarkets_gdf, "Supermarkets", debug_label, spatial_index)
    
    def _filter_amenities_in_polygon(self, area_result, points, amenities_gdf, amenity_name, debug_label="", spatial_index=None):
        """Generic method to filter amenities within the isochrone polygon with spatial index optimization"""
        try:
            # Quick validation checks
            if len(points) < 3 or amenities_gdf is None:
                return 0
            
            if hasattr(amenities_gdf, 'empty') and amenities_gdf.empty:
                return 0
            
            # Create polygon from isochrone area for containment checks (with caching)
            from shapely.geometry import Point
            polygon = self._get_or_create_polygon(area_result, points)
            if polygon is None:
                return 0
            
            # Use spatial index optimization if available
            if spatial_index is not None and isinstance(spatial_index, dict):
                try:
                    # Get polygon bounds for spatial index query
                    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                    rtree_index = spatial_index['index']
                    id_mapping = spatial_index['id_mapping']
                    candidate_int_ids = list(rtree_index.intersection(bounds))
                    
                    # Calculate polygon centroid for distance pre-filtering
                    poly_centroid = polygon.centroid
                    
                    # Estimate reasonable distance threshold based on polygon size
                    # Use polygon bounds diagonal as a rough distance threshold
                    minx, miny, maxx, maxy = bounds
                    diagonal_distance = ((maxx - minx)**2 + (maxy - miny)**2)**0.5
                    distance_threshold = diagonal_distance * 1.5  # Add some buffer
                    
                    # Now do distance pre-filtering and precise polygon containment test only on candidates
                    accessible_count = 0
                    distance_filtered = 0
                    for int_id in candidate_int_ids:
                        try:
                            # Map back to original index
                            original_idx = id_mapping[int_id]
                            if original_idx in amenities_gdf.index:
                                amenity = amenities_gdf.loc[original_idx]
                                if hasattr(amenity.geometry, 'centroid'):
                                    point = amenity.geometry.centroid
                                else:
                                    point = amenity.geometry
                                
                                # Distance pre-filtering: skip if too far from polygon centroid
                                distance_to_center = ((point.x - poly_centroid.x)**2 + (point.y - poly_centroid.y)**2)**0.5
                                if distance_to_center > distance_threshold:
                                    distance_filtered += 1
                                    continue
                                
                                amenity_point = Point(point.x, point.y)
                                if polygon.contains(amenity_point):
                                    accessible_count += 1
                        except Exception:
                            continue
                    
                    
                    return accessible_count
                    
                except Exception as e:
                    print(f"{amenity_name} spatial index failed: {e}")
                    # Fall through to brute force method
                    pass
            else:
                pass  # Fall back to brute force method
            
            # Fallback: Filter amenities within the accessible area polygon (brute force)
            accessible_count = 0
            for idx, amenity in amenities_gdf.iterrows():
                try:
                    if hasattr(amenity.geometry, 'centroid'):
                        point = amenity.geometry.centroid
                    else:
                        point = amenity.geometry
                    
                    # Check if point is within the accessible area polygon
                    amenity_point = Point(point.x, point.y)  # Point(lng, lat) to match polygon coordinate system
                    if polygon.contains(amenity_point):
                        accessible_count += 1
                except Exception:
                    continue
            
            return accessible_count
                
        except Exception as e:
            print(f"Error calculating accessible {amenity_name.lower()}: {e}")
            return 0

    def _calculate_accessible_schools(self, area_result, points, schools_gdf, debug_label="", spatial_index=None):
        """Calculate number of schools accessible within the isochrone area using pre-loaded data"""
        return self._filter_amenities_in_polygon(area_result, points, schools_gdf, "Schools", debug_label, spatial_index)
    
    def _calculate_accessible_playgrounds(self, area_result, points, playgrounds_gdf, debug_label="", spatial_index=None):
        """Calculate number of playgrounds accessible within the isochrone area using pre-loaded data"""
        return self._filter_amenities_in_polygon(area_result, points, playgrounds_gdf, "Playgrounds", debug_label, spatial_index)
    
    def _calculate_accessible_cafes_bars(self, area_result, points, cafes_bars_gdf, debug_label="", spatial_index=None):
        """Calculate number of cafes/bars accessible within the isochrone area using pre-loaded data"""
        return self._filter_amenities_in_polygon(area_result, points, cafes_bars_gdf, "Cafes/Bars", debug_label, spatial_index)
    
    def _calculate_accessible_transit(self, area_result, points, transit_gdf, debug_label="", spatial_index=None):
        """Calculate number of public transit stations accessible within the isochrone area using pre-loaded data"""
        return self._filter_amenities_in_polygon(area_result, points, transit_gdf, "Transit", debug_label, spatial_index)
    
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
            import time
            start_time = time.time()
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
            buffer_start = time.time()
            buffered_edges = edges_projected.buffer(buffer_distance, cap_style=2, join_style=2)
            print(f"Buffering {len(reachable_edges)} edges took {time.time() - buffer_start:.2f}s")
            
            # Optimize union operation for large numbers of edges
            union_start = time.time()
            if len(buffered_edges) > 1500:
                # For very large numbers, use a faster approximation
                print(f"Using fast approximation for {len(buffered_edges)} buffers...")
                # Get convex hull of all buffer centroids and expand
                centroids = [geom.centroid for geom in buffered_edges.tolist()]
                from shapely.geometry import MultiPoint
                points = MultiPoint(centroids)
                unified_buffer = points.convex_hull.buffer(buffer_distance * 2)
                print(f"Fast approximation took {time.time() - union_start:.2f}s")
            elif len(buffered_edges) > 1000:
                # For large numbers, use cascaded union which is more efficient
                try:
                    from shapely.ops import unary_union
                    unified_buffer = unary_union(buffered_edges.tolist())
                    print(f"Cascaded union of {len(buffered_edges)} buffers took {time.time() - union_start:.2f}s")
                except:
                    # Fallback to pandas unary_union
                    unified_buffer = buffered_edges.unary_union
                    print(f"Pandas unary_union of {len(buffered_edges)} buffers took {time.time() - union_start:.2f}s")
            else:
                unified_buffer = buffered_edges.unary_union
                print(f"Standard union of {len(buffered_edges)} buffers took {time.time() - union_start:.2f}s")
            
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
            
            print(f"Total buffered network isochrone creation took {time.time() - start_time:.2f}s")
            
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
    
    def _get_park_edges(self, graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                       park_geom) -> List[Tuple[int, int]]:
        """Get list of edges that intersect with park"""
        edges_to_remove = []
        
        try:
            # Get edges as GeoDataFrame
            _, edges_gdf = ox.graph_to_gdfs(graph)
            
            # Project to metric CRS for intersection
            edges_proj = edges_gdf.to_crs('EPSG:3857')
            park_proj = gpd.GeoSeries([park_geom], crs=edges_gdf.crs).to_crs('EPSG:3857').iloc[0]
            
            # Find intersecting edges
            intersects = edges_proj.geometry.intersects(park_proj)
            intersecting_edges = edges_proj[intersects]
            
            # Collect edge tuples to remove
            for edge_idx in intersecting_edges.index:
                try:
                    u, v, k = edge_idx
                    # For NetworkX multigraphs, we only need (u, v) for removal
                    edges_to_remove.append((u, v))
                except:
                    continue
                    
        except Exception as e:
            print(f"Error finding park edges: {e}")
        
        return edges_to_remove
    
    def _remove_park_edges(self, graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, 
                          park_geom) -> nx.Graph:
        """Remove edges that intersect with park (legacy method)"""
        # Get edges to remove
        edges_to_remove = self._get_park_edges(graph, nodes_gdf, park_geom)
        
        # Create new graph with edges removed
        filtered_graph = graph.copy()
        for edge in edges_to_remove:
            if filtered_graph.has_edge(edge[0], edge[1]):
                filtered_graph.remove_edge(edge[0], edge[1])
        
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
        .tab-navigation {
            display: flex;
            border-bottom: 2px solid #e9ecef;
            margin-bottom: 20px;
        }
        .tab-btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background-color: #f8f9fa;
            color: #6c757d;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            border-radius: 8px 8px 0 0;
            margin-right: 4px;
            transition: all 0.3s ease;
        }
        .tab-btn:hover {
            background-color: #e9ecef;
            color: #495057;
        }
        .tab-btn.active {
            background-color: #007bff;
            color: white;
            border-bottom: 2px solid #007bff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .comparison-table-container {
            margin: 20px 0;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-table th {
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: 600;
        }
        .comparison-table th:first-child {
            background-color: #495057;
            text-align: left;
        }
        .comparison-table td {
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
        }
        .comparison-table td:first-child {
            background-color: #f8f9fa;
            font-weight: 500;
            color: #495057;
        }
        .comparison-table td:not(:first-child) {
            text-align: center;
            font-weight: 600;
            font-size: 16px;
        }
        .comparison-table tr:last-child td {
            border-bottom: none;
        }
        .comparison-table .metric-unit {
            font-size: 14px;
            color: #6c757d;
            font-weight: normal;
        }
        .comparison-table .impact-row td:not(:first-child) {
            font-size: 14px;
        }
        .comparison-table .difference-row {
            background-color: #f1f3f4;
        }
        .comparison-table .difference-row td:first-child {
            background-color: #e9ecef;
            font-weight: 600;
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
        
        /* Significance level styling */
        .significance-no-change { color: #6c757d; }
        .significance-minimal { color: #ffc107; }
        .significance-notable { color: #17a2b8; }
        .significance-strong { color: #28a745; }
        .significance-high { color: #20c997; font-weight: bold; }
        .significance-transformative { color: #198754; font-weight: bold; }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            line-height: 1.4;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
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
            <p>Analyze how parks affect X-minute walkability</p>
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
                        <option value="buffered_network" selected>Buffered Streets (Realistic)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="networkBackend">Network Analysis:</label>
                    <select id="networkBackend">
                        <option value="networkx">NetworkX (Stable)</option>
                        <option value="networkit" selected>NetworKit (Faster)</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="step">
            <h3>Step 4: Analysis Mode</h3>
            <p>Choose your analysis approach and configure settings.</p>
            
            <!-- Tab Navigation -->
            <div class="tab-navigation">
                <button class="tab-btn active" onclick="switchTab('single')" id="singleTab">🎯 Single Node Analysis</button>
                <button class="tab-btn" onclick="switchTab('mass')" id="massTab">📊 Mass Analysis</button>
            </div>
            
            <!-- Single Node Analysis Tab -->
            <div id="singleAnalysisTab" class="tab-content active">
                <p>Run accessibility analysis for the selected park and node to see the impact.</p>
                <div class="controls">
                    <div class="control-group">
                        <label>&nbsp;</label>
                        <button class="btn" onclick="runAnalysis()" disabled id="analyzeBtn">🔍 Run Single Analysis</button>
                    </div>
                </div>
            </div>
            
            <!-- Mass Analysis Tab -->
            <div id="massAnalysisTab" class="tab-content">
                <p>Analyze multiple nodes around the selected park to get comprehensive statistics about the park's accessibility impact.</p>
                <div class="controls">
                    <div class="control-group">
                        <label for="massNodeCount">Number of nodes to analyze:</label>
                        <input type="number" id="massNodeCount" min="2" max="50" value="10" step="1" style="width: 80px;">
                        <span id="massNodeRange" style="color: #6c757d; font-size: 12px; margin-left: 10px;">Range: 2 - 50 nodes</span>
                    </div>
                    <div class="control-group">
                        <label for="massRadius">Node search radius (m):</label>
                        <input type="range" id="massRadius" min="400" max="1200" value="600" step="100" oninput="updateMassRadius()">
                        <span id="massRadiusValue">600m</span>
                    </div>
                    <div class="control-group">
                        <label>&nbsp;</label>
                        <button class="btn" onclick="runMassAnalysis()" disabled id="massAnalyzeBtn" style="background-color: #28a745;">🔍 Mass Analysis</button>
                        <button class="btn" onclick="cancelMassAnalysis()" disabled id="massCancelBtn" style="background-color: #dc3545; margin-left: 10px; display: none;">❌ Cancel</button>
                    </div>
                </div>
                <div id="massProgress" style="display: none;">
                    <div style="background-color: #e9ecef; border-radius: 10px; padding: 3px; margin: 10px 0;">
                        <div id="massProgressBar" style="background-color: #28a745; height: 20px; border-radius: 7px; width: 0%; transition: width 0.3s;"></div>
                    </div>
                    <div id="massProgressText">Analyzing node 0/10...</div>
                </div>
            </div>
        </div>

        <!-- Single Node Results -->
        <div id="results" class="results">
            <h3>📊 Single Node Results</h3>
            <div id="metrics" class="comparison-table-container"></div>
            
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
                <h4>📈 Interactive Visualizations</h4>
                <div class="chart-container" style="margin-bottom: 30px;">
                    <canvas id="massComparisonChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="impactCategoriesChart"></canvas>
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
        let poisData = [];
        let layerGroups = {
            parks: [],
            nodes: [],
            withPark: null,
            withoutPark: null,
            pois: []
        };
        let layerVisibility = {
            parks: true,
            nodes: true,
            withPark: true,
            withoutPark: true,
            pois: false
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

        function switchTab(tabName) {
            // Hide all tab contents
            document.getElementById('singleAnalysisTab').classList.remove('active');
            document.getElementById('massAnalysisTab').classList.remove('active');
            
            // Remove active class from all tab buttons
            document.getElementById('singleTab').classList.remove('active');
            document.getElementById('massTab').classList.remove('active');
            
            // Show selected tab content and activate button
            if (tabName === 'single') {
                document.getElementById('singleAnalysisTab').classList.add('active');
                document.getElementById('singleTab').classList.add('active');
                // Hide mass results when switching to single tab
                document.getElementById('massResults').style.display = 'none';
            } else if (tabName === 'mass') {
                document.getElementById('massAnalysisTab').classList.add('active');
                document.getElementById('massTab').classList.add('active');
                // Hide single results when switching to mass tab
                document.getElementById('results').style.display = 'none';
            }
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
            
            if (layerGroups.pois.length > 0) {
                layerConfigs.push({
                    key: 'pois',
                    name: 'Points of Interest',
                    color: 'purple',
                    count: layerGroups.pois.length
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

        function getRangeText(percentChange) {
            const absChange = Math.abs(percentChange);
            if (absChange === 0) return '0';
            if (absChange <= 5) return '0–5';
            if (absChange <= 15) return '5–15';
            if (absChange <= 30) return '15–30';
            if (absChange <= 50) return '30–50';
            return '>50';
        }

        function calculateMetricStats(values) {
            if (!values || values.length === 0) return null;
            
            const sorted = [...values].sort((a, b) => a - b);
            const len = sorted.length;
            
            const min = sorted[0];
            const max = sorted[len - 1];
            const avg = values.reduce((sum, val) => sum + val, 0) / len;
            
            // Calculate standard deviation
            const variance = values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / len;
            const stdDev = Math.sqrt(variance);
            
            // Calculate percentiles
            const q1Index = Math.floor(len * 0.25);
            const q3Index = Math.floor(len * 0.75);
            const q1 = sorted[q1Index];
            const q3 = sorted[q3Index];
            
            return {
                min: min,
                max: max,
                avg: avg,
                stdDev: stdDev,
                q1: q1,
                q3: q3,
                count: len
            };
        }

        function createRangeBar(stats, value, unit = '', isPercentage = false) {
            if (!stats) return '';
            
            const range = stats.max - stats.min;
            const avgPosition = range > 0 ? ((stats.avg - stats.min) / range) * 100 : 50;
            const q1Position = range > 0 ? ((stats.q1 - stats.min) / range) * 100 : 25;
            const q3Position = range > 0 ? ((stats.q3 - stats.min) / range) * 100 : 75;
            
            const formatValue = (val) => isPercentage ? `${val.toFixed(1)}%` : `${val.toFixed(3)}${unit}`;
            
            return `
                <div class="range-bar-container" style="margin: 10px 0;">
                    <div class="range-labels" style="display: flex; justify-content: space-between; font-size: 11px; color: #666; margin-bottom: 2px;">
                        <span>Min: ${formatValue(stats.min)}</span>
                        <span>Avg: ${formatValue(stats.avg)}</span>
                        <span>Max: ${formatValue(stats.max)}</span>
                    </div>
                    <div class="range-bar" style="position: relative; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden;">
                        <div class="range-iqr" style="position: absolute; left: ${q1Position}%; width: ${q3Position - q1Position}%; height: 100%; background: rgba(40, 167, 69, 0.3);"></div>
                        <div class="range-avg-marker" style="position: absolute; left: ${avgPosition}%; width: 2px; height: 100%; background: #007bff; transform: translateX(-50%);"></div>
                        <div class="range-value-text" style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); font-size: 10px; font-weight: bold; color: #495057;">
                            ${formatValue(value)}
                        </div>
                    </div>
                    <div class="range-stats" style="font-size: 10px; color: #6c757d; margin-top: 2px;">
                        Std Dev: ${formatValue(stats.stdDev)} | IQR: ${formatValue(stats.q1)} - ${formatValue(stats.q3)}
                    </div>
                </div>
            `;
        }

        function toggleMetricDetails(metricId) {
            const detailsRow = document.getElementById(`details-${metricId}`);
            const icon = document.getElementById(`icon-${metricId}`);
            
            if (detailsRow.style.display === 'none' || detailsRow.style.display === '') {
                detailsRow.style.display = 'table-row';
                icon.textContent = '▼';
            } else {
                detailsRow.style.display = 'none';
                icon.textContent = '▶';
            }
        }

        function isPointInPolygon(point, polygon) {
            // Ray casting algorithm to check if point is inside polygon
            const x = point[1]; // longitude
            const y = point[0]; // latitude
            let inside = false;
            
            for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
                const xi = polygon[i][1]; // longitude
                const yi = polygon[i][0]; // latitude
                const xj = polygon[j][1]; // longitude
                const yj = polygon[j][0]; // latitude
                
                if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                    inside = !inside;
                }
            }
            
            return inside;
        }

        function getSignificance(percentChange) {
            const absChange = Math.abs(percentChange);
            
            if (absChange === 0) {
                return {
                    level: 'No Change',
                    class: 'significance-no-change',
                    description: 'No difference in walking access; the park removal has no measurable network impact.'
                };
            } else if (absChange <= 5) {
                return {
                    level: 'Minimal',
                    class: 'significance-minimal',
                    description: 'Little change in walking access; added paths have negligible network impact.'
                };
            } else if (absChange <= 15) {
                return {
                    level: 'Notable', 
                    class: 'significance-notable',
                    description: 'Clear, measurable improvement in walking routes and connectivity.'
                };
            } else if (absChange <= 30) {
                return {
                    level: 'Strong',
                    class: 'significance-strong', 
                    description: 'Substantial gains in walking accessibility.'
                };
            } else if (absChange <= 50) {
                return {
                    level: 'High',
                    class: 'significance-high',
                    description: 'Major enhancement; walking network performs far better, enabling new trip patterns.'
                };
            } else {
                return {
                    level: 'Transformative',
                    class: 'significance-transformative',
                    description: 'Game-changing connectivity; walking opportunities expand dramatically across the area.'
                };
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
            } else if (layerKey === 'pois') {
                layerGroups.pois.forEach(layer => {
                    if (show) {
                        map.addLayer(layer);
                    } else {
                        map.removeLayer(layer);
                    }
                });
            }
        }

        async function loadData() {
            const lat = parseFloat(document.getElementById('lat').value);
            const lng = parseFloat(document.getElementById('lng').value);
            const radius = parseInt(document.getElementById('radius').value);

            showStatus('Starting data loading...', 'loading');
            
            // Poll for real progress from backend
            let progressInterval = setInterval(async () => {
                try {
                    const progressResponse = await fetch(`/progress?lat=${lat}&lng=${lng}&radius=${radius}`);
                    const progress = await progressResponse.json();
                    
                    if (progress.error) {
                        clearInterval(progressInterval);
                        showStatus(`Error: ${progress.step_name}`, 'error');
                        return;
                    }
                    
                    let statusText = `Step ${progress.current_step}/${progress.total_steps}: ${progress.step_name}`;
                    
                    // Add detailed sub-progress and operation info if available
                    if (progress.current_operation && progress.current_operation !== progress.step_name) {
                        statusText += `\n${progress.current_operation}`;
                    }
                    
                    // Add time estimate if available
                    if (progress.estimated_time_remaining && progress.estimated_time_remaining > 10) {
                        const minutes = Math.ceil(progress.estimated_time_remaining / 60);
                        statusText += ` (Est. ${minutes}min remaining)`;
                    }
                    
                    showStatus(statusText, 'loading');
                    
                    if (progress.completed) {
                        clearInterval(progressInterval);
                    }
                } catch (error) {
                    console.log('Progress polling error:', error);
                    // Continue polling even if progress request fails
                }
            }, 500); // Poll every 500ms for real-time updates
            
            try {
                const response = await fetch(`/load-data?lat=${lat}&lng=${lng}&radius=${radius}`);
                clearInterval(progressInterval); // Stop progress polling
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Failed to load data');
                }

                parksData = data.parks;
                nodesData = data.nodes;
                
                // Store POI data
                poisData = data.pois || [];

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
                
                // Add POIs to map
                poisData.forEach(poi => {
                    // Create custom marker with emoji
                    const poiMarker = L.marker([poi.lat, poi.lng], {
                        title: `${poi.emoji} ${poi.name}`
                    });
                    
                    // Only add to map if POIs are visible  
                    if (layerVisibility.pois) {
                        poiMarker.addTo(map);
                    }
                    
                    poiMarker.bindPopup(`<b>${poi.emoji} ${poi.name}</b><br>Type: ${poi.type}`);
                    
                    layerGroups.pois.push(poiMarker);
                });

                updateLayerControls();

                parkSelect.disabled = false;
                showStatus(`Loaded ${parksData.length} parks, ${nodesData.length} nodes, and ${poisData.length} POIs`, 'success');
                
                setTimeout(hideStatus, 3000);
                
            } catch (error) {
                clearInterval(progressInterval); // Stop progress polling on error
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
            layerGroups.pois = [];
            
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
            
            // Filter nodes near park (within reasonable distance) but NOT inside the park
            const nearbyNodes = nodesData.filter(node => {
                const distance = getDistance(
                    selectedPark.centroid[0], selectedPark.centroid[1],
                    node.lat, node.lng
                );
                
                // Check if node is within reasonable distance
                if (distance >= 800) return false;
                
                // Check if node is inside the park - exclude if it is
                const nodePoint = [node.lat, node.lng];
                const isInsidePark = isPointInPolygon(nodePoint, selectedPark.coords);
                
                return !isInsidePark; // Return true only if NOT inside park
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
            
            // Update mass analysis node count range based on available nodes
            const massNodeCount = document.getElementById('massNodeCount');
            const massNodeRange = document.getElementById('massNodeRange');
            const maxNodes = Math.max(2, nearbyNodes.length);
            massNodeCount.max = maxNodes;
            if (massNodeCount.value > maxNodes) {
                massNodeCount.value = maxNodes;
            }
            massNodeRange.textContent = `Range: 2 - ${maxNodes} nodes (${maxNodes} available)`;
            
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
            const networkBackend = document.getElementById('networkBackend').value;
            
            if (!parkId || !nodeId) {
                alert('Please select both a park and a node');
                return;
            }

            const methodName = vizMethod === 'buffered_network' ? 'Buffered Streets' : 'Convex Hull';
            const backendName = networkBackend === 'networkit' ? 'NetworKit' : 'NetworkX';
            showStatus(`Calculating ${walkTime}-minute accessibility (${methodName} + ${backendName})...`, 'loading');
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat, lng, radius, park_id: parkId, node_id: nodeId,
                        walk_time: walkTime, walk_speed: walkSpeed, viz_method: vizMethod,
                        network_backend: networkBackend
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
            
            // Calculate differences safely with null checks
            const withPark = results.with_park || {};
            const withoutPark = results.without_park || {};
            const enhanced = results.enhanced_metrics || {};
            
            // Calculate percentage changes from base (without park) values
            const areaWithout = withoutPark.area_km2 || 0;
            const areaWith = withPark.area_km2 || 0;
            const areaChange = areaWithout > 0 ? ((areaWith - areaWithout) / areaWithout) * 100 : 0;
            
            const streetWithout = enhanced.without_street_length_km || 0;
            const streetWith = enhanced.with_street_length_km || 0;
            const streetChange = streetWithout > 0 ? ((streetWith - streetWithout) / streetWithout) * 100 : 0;
            
            const supermarketsWithout = enhanced.without_supermarkets_accessible || 0;
            const supermarketsWith = enhanced.with_supermarkets_accessible || 0;
            const supermarketsChange = supermarketsWithout > 0 ? ((supermarketsWith - supermarketsWithout) / supermarketsWithout) * 100 : 0;
            
            const schoolsWithout = enhanced.without_schools_accessible || 0;
            const schoolsWith = enhanced.with_schools_accessible || 0;
            const schoolsChange = schoolsWithout > 0 ? ((schoolsWith - schoolsWithout) / schoolsWithout) * 100 : 0;
            
            const playgroundsWithout = enhanced.without_playgrounds_accessible || 0;
            const playgroundsWith = enhanced.with_playgrounds_accessible || 0;
            const playgroundsChange = playgroundsWithout > 0 ? ((playgroundsWith - playgroundsWithout) / playgroundsWithout) * 100 : 0;
            
            const cafesBarsWithout = enhanced.without_cafes_bars_accessible || 0;
            const cafesBarsWith = enhanced.with_cafes_bars_accessible || 0;
            const cafesBarsChange = cafesBarsWithout > 0 ? ((cafesBarsWith - cafesBarsWithout) / cafesBarsWithout) * 100 : 0;
            
            const transitWithout = enhanced.without_transit_accessible || 0;
            const transitWith = enhanced.with_transit_accessible || 0;
            const transitChange = transitWithout > 0 ? ((transitWith - transitWithout) / transitWithout) * 100 : 0;
            
            metrics.innerHTML = `
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Without Park</th>
                            <th>With Park</th>
                            <th>Difference</th>
                            <th>Significance</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Accessible Area</td>
                            <td>${areaWithout.toFixed(3)} <span class="metric-unit">km²</span></td>
                            <td>${areaWith.toFixed(3)} <span class="metric-unit">km²</span></td>
                            <td class="${areaChange > 0 ? 'impact-positive' : areaChange < 0 ? 'impact-negative' : 'impact-neutral'}">${areaChange > 0 ? '+' : ''}${areaChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(areaChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(areaChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(areaChange).level} (${getRangeText(areaChange)}%)</strong><br>
                                        ${getSignificance(areaChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>Street Network Length</td>
                            <td>${streetWithout.toFixed(3)} <span class="metric-unit">km</span></td>
                            <td>${streetWith.toFixed(3)} <span class="metric-unit">km</span></td>
                            <td class="${streetChange > 0 ? 'impact-positive' : streetChange < 0 ? 'impact-negative' : 'impact-neutral'}">${streetChange > 0 ? '+' : ''}${streetChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(streetChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(streetChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(streetChange).level} (${getRangeText(streetChange)}%)</strong><br>
                                        ${getSignificance(streetChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>🥖 Accessible Supermarkets</td>
                            <td>${supermarketsWithout} <span class="metric-unit">count</span></td>
                            <td>${supermarketsWith} <span class="metric-unit">count</span></td>
                            <td class="${supermarketsChange > 0 ? 'impact-positive' : supermarketsChange < 0 ? 'impact-negative' : 'impact-neutral'}">${supermarketsChange > 0 ? '+' : ''}${supermarketsChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(supermarketsChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(supermarketsChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(supermarketsChange).level} (${getRangeText(supermarketsChange)}%)</strong><br>
                                        ${getSignificance(supermarketsChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>🏫 Accessible Schools</td>
                            <td>${schoolsWithout} <span class="metric-unit">count</span></td>
                            <td>${schoolsWith} <span class="metric-unit">count</span></td>
                            <td class="${schoolsChange > 0 ? 'impact-positive' : schoolsChange < 0 ? 'impact-negative' : 'impact-neutral'}">${schoolsChange > 0 ? '+' : ''}${schoolsChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(schoolsChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(schoolsChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(schoolsChange).level} (${getRangeText(schoolsChange)}%)</strong><br>
                                        ${getSignificance(schoolsChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>🛝 Accessible Playgrounds</td>
                            <td>${playgroundsWithout} <span class="metric-unit">count</span></td>
                            <td>${playgroundsWith} <span class="metric-unit">count</span></td>
                            <td class="${playgroundsChange > 0 ? 'impact-positive' : playgroundsChange < 0 ? 'impact-negative' : 'impact-neutral'}">${playgroundsChange > 0 ? '+' : ''}${playgroundsChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(playgroundsChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(playgroundsChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(playgroundsChange).level} (${getRangeText(playgroundsChange)}%)</strong><br>
                                        ${getSignificance(playgroundsChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>☕ Accessible Cafes/Bars</td>
                            <td>${cafesBarsWithout} <span class="metric-unit">count</span></td>
                            <td>${cafesBarsWith} <span class="metric-unit">count</span></td>
                            <td class="${cafesBarsChange > 0 ? 'impact-positive' : cafesBarsChange < 0 ? 'impact-negative' : 'impact-neutral'}">${cafesBarsChange > 0 ? '+' : ''}${cafesBarsChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(cafesBarsChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(cafesBarsChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(cafesBarsChange).level} (${getRangeText(cafesBarsChange)}%)</strong><br>
                                        ${getSignificance(cafesBarsChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td>🚌 Accessible Transit Stops</td>
                            <td>${transitWithout} <span class="metric-unit">count</span></td>
                            <td>${transitWith} <span class="metric-unit">count</span></td>
                            <td class="${transitChange > 0 ? 'impact-positive' : transitChange < 0 ? 'impact-negative' : 'impact-neutral'}">${transitChange > 0 ? '+' : ''}${transitChange.toFixed(1)} <span class="metric-unit">%</span></td>
                            <td class="${getSignificance(transitChange).class}">
                                <div class="tooltip">
                                    ${getSignificance(transitChange).level}
                                    <span class="tooltiptext">
                                        <strong>${getSignificance(transitChange).level} (${getRangeText(transitChange)}%)</strong><br>
                                        ${getSignificance(transitChange).description}
                                    </span>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            `;

            
            // Create charts
            createSingleAnalysisCharts(results);

            // Only show results if single analysis tab is active
            if (document.getElementById('singleAnalysisTab').classList.contains('active')) {
                document.getElementById('results').style.display = 'block';
            }
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
        let analysisAborted = false;

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

        function selectOptimalNodes(nearbyNodes, nodeCount, selectedPark) {
            // If we have fewer or equal nodes than requested, return all
            if (nearbyNodes.length <= nodeCount) {
                return nearbyNodes;
            }

            // Sort nodes by distance from park centroid for initial ordering
            const sortedByDistance = nearbyNodes.map(node => ({
                ...node,
                distanceFromPark: getDistance(
                    selectedPark.centroid[0], selectedPark.centroid[1],
                    node.lat, node.lng
                )
            })).sort((a, b) => a.distanceFromPark - b.distanceFromPark);

            // Implement spatial clustering to ensure good coverage
            const selectedNodes = [];
            const minSeparationDistance = 100; // meters - minimum distance between selected nodes
            
            // Always include the closest node
            selectedNodes.push(sortedByDistance[0]);
            
            // Select additional nodes ensuring spatial separation
            for (let i = 1; i < sortedByDistance.length && selectedNodes.length < nodeCount; i++) {
                const candidate = sortedByDistance[i];
                let tooClose = false;
                
                // Check if candidate is too close to any already selected node
                for (const selected of selectedNodes) {
                    const separation = getDistance(
                        candidate.lat, candidate.lng,
                        selected.lat, selected.lng
                    );
                    if (separation < minSeparationDistance) {
                        tooClose = true;
                        break;
                    }
                }
                
                if (!tooClose) {
                    selectedNodes.push(candidate);
                }
            }
            
            // If we still need more nodes and the separation constraint is too strict,
            // fill remaining slots with closest available nodes
            if (selectedNodes.length < nodeCount) {
                const remaining = sortedByDistance.filter(node => 
                    !selectedNodes.some(selected => selected.id === node.id)
                );
                const needed = nodeCount - selectedNodes.length;
                selectedNodes.push(...remaining.slice(0, needed));
            }
            
            console.log(`Selected ${selectedNodes.length} spatially distributed nodes from ${nearbyNodes.length} available`);
            return selectedNodes;
        }

        async function runMassAnalysis() {
            const parkId = document.getElementById('parkSelect').value;
            const nodeCount = parseInt(document.getElementById('massNodeCount').value);
            const searchRadius = parseInt(document.getElementById('massRadius').value);
            const walkTime = parseFloat(document.getElementById('walkTime').value);
            const walkSpeed = parseFloat(document.getElementById('walkSpeed').value);
            const vizMethod = document.getElementById('vizMethod').value;
            const networkBackend = document.getElementById('networkBackend').value;
            
            if (!parkId) {
                alert('Please select a park first');
                return;
            }

            // Find selected park
            const selectedPark = parksData.find(p => p.id === parkId);
            if (!selectedPark) return;

            // Get nearby nodes for mass analysis (excluding nodes inside the park)
            const nearbyNodes = nodesData.filter(node => {
                const distance = getDistance(
                    selectedPark.centroid[0], selectedPark.centroid[1],
                    node.lat, node.lng
                );
                
                // Check if node is within search radius
                if (distance >= searchRadius) return false;
                
                // Check if node is inside the park - exclude if it is
                const nodePoint = [node.lat, node.lng];
                const isInsidePark = isPointInPolygon(nodePoint, selectedPark.coords);
                
                return !isInsidePark; // Return true only if NOT inside park
            });

            if (nearbyNodes.length === 0) {
                alert('No nodes found in the specified radius. Try increasing the search radius.');
                return;
            }

            // Smart node selection with spatial clustering optimization
            const selectedNodes = selectOptimalNodes(nearbyNodes, nodeCount, selectedPark);

            // Show progress with time estimation
            document.getElementById('massProgress').style.display = 'block';
            document.getElementById('massAnalyzeBtn').disabled = true;
            document.getElementById('massCancelBtn').style.display = 'inline-block';
            document.getElementById('massCancelBtn').disabled = false;
            
            // Initialize time tracking for progress estimation
            const analysisStartTime = Date.now();
            analysisAborted = false;  // Reset global flag

            massResultsData = [];
            const lat = parseFloat(document.getElementById('lat').value);
            const lng = parseFloat(document.getElementById('lng').value);
            const radius = parseInt(document.getElementById('radius').value);

            // Extract node IDs for batch processing
            const nodeIds = selectedNodes.map(node => node.id);
            
            try {
                // Show initial progress
                document.getElementById('massProgressBar').style.width = '10%';
                document.getElementById('massProgressText').textContent = 
                    `Starting batch analysis of ${nodeIds.length} nodes...`;

                // Start batch request and get progress key immediately
                const response = await fetch('/mass-analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat, lng, radius, 
                        park_id: parkId, 
                        node_ids: nodeIds,
                        walk_time: walkTime, 
                        walk_speed: walkSpeed, 
                        viz_method: vizMethod,
                        network_backend: networkBackend
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to start batch analysis');
                }
                
                const batchData = await response.json();
                const progressKey = batchData.progress_key;
                
                let results = null;
                
                // Single unified progress polling with completion handling
                await new Promise(resolve => {
                    let progressInterval = setInterval(async () => {
                        try {
                            const progressResponse = await fetch(`/mass-progress?progress_key=${progressKey}`);
                            const progress = await progressResponse.json();
                            
                            const percentage = Math.min((progress.completed_nodes / progress.total_nodes) * 100, 100);
                            document.getElementById('massProgressBar').style.width = percentage + '%';
                            
                            if (progress.current_node && progress.current_node !== 'Not found') {
                                // Calculate time estimation
                                const elapsed = Date.now() - analysisStartTime;
                                const avgTimePerNode = elapsed / Math.max(progress.completed_nodes, 1);
                                const remaining = progress.total_nodes - progress.completed_nodes;
                                const estimatedTimeRemaining = remaining * avgTimePerNode;
                                
                                let timeText = '';
                                if (estimatedTimeRemaining > 60000) {
                                    timeText = ` (~${Math.ceil(estimatedTimeRemaining/60000)} min remaining)`;
                                } else if (estimatedTimeRemaining > 1000) {
                                    timeText = ` (~${Math.ceil(estimatedTimeRemaining/1000)} sec remaining)`;
                                }
                                
                                document.getElementById('massProgressText').textContent = 
                                    `Analyzing node ${progress.completed_nodes}/${progress.total_nodes}... (${progress.current_node})${timeText}`;
                            }
                            
                            // Check for completion and resolve
                            if (progress.completed) {
                                clearInterval(progressInterval);
                                
                                // Check for server-side errors
                                if (progress.error) {
                                    console.error('Mass analysis failed on server:', progress.error);
                                    alert(`Mass analysis failed: ${progress.error}`);
                                    return;
                                }
                                
                                results = progress.results;
                                console.log('Mass analysis completed. Results:', results);
                                
                                if (!results) {
                                    console.error('Mass analysis completed but no results found');
                                    alert('Mass analysis completed but returned no results. Check server logs for errors.');
                                    return;
                                }
                                
                                document.getElementById('massProgressText').textContent = 'Processing results...';
                                resolve();
                            }
                        } catch (error) {
                            console.log('Progress polling error:', error);
                        }
                    }, 1500); // Poll every 1.5 seconds for good responsiveness without overload
                });
                
                document.getElementById('massProgressBar').style.width = '90%';
                document.getElementById('massProgressText').textContent = 'Processing batch results...';
                
                if (response.ok) {
                    // Check if results is actually an array
                    if (!Array.isArray(results)) {
                        console.error('Invalid response format:', results);
                        alert(`Mass analysis failed: Server returned invalid response format. Response: ${JSON.stringify(results)}`);
                        return;
                    }
                    
                    console.log(`Received ${results.length} results from server`);
                    
                    // Process batch results
                    for (let result of results) {
                        if (result.error) {
                            console.error(`Error for node ${result.node_id}:`, result.error);
                            if (result.error_details) {
                                console.error(`Node ${result.node_id} error details:`, result.error_details);
                            }
                            // Skip adding failed nodes to results display
                            continue;
                        }
                        
                        // Only process successful results with valid park_id
                        if (!result.park_id || !result.with_park || !result.without_park) {
                            console.error(`Invalid result structure for node ${result.node_id}:`, result);
                            continue;
                        }
                        
                        // Find the node data to get coordinates
                        const nodeData = selectedNodes.find(n => n.id === result.node_id);
                        
                        massResultsData.push({
                            node_id: result.node_id,
                            node_lat: nodeData ? nodeData.lat : 0,
                            node_lng: nodeData ? nodeData.lng : 0,
                            with_area: result.with_park?.area_km2 || 0,
                            without_area: result.without_park?.area_km2 || 0,
                            with_street_length: result.enhanced_metrics?.with_street_length_km || 0,
                            without_street_length: result.enhanced_metrics?.without_street_length_km || 0,
                            difference: result.difference_km2 || 0,
                            street_network_difference: result.street_network_difference_km || 0,
                            area_change_pct: result.area_change_pct || 0,
                            street_network_change_pct: result.street_network_change_pct || 0,
                            // Amenity data
                            with_supermarkets: result.enhanced_metrics?.with_supermarkets_accessible || 0,
                            without_supermarkets: result.enhanced_metrics?.without_supermarkets_accessible || 0,
                            with_schools: result.enhanced_metrics?.with_schools_accessible || 0,
                            without_schools: result.enhanced_metrics?.without_schools_accessible || 0,
                            with_playgrounds: result.enhanced_metrics?.with_playgrounds_accessible || 0,
                            without_playgrounds: result.enhanced_metrics?.without_playgrounds_accessible || 0,
                            with_cafes_bars: result.enhanced_metrics?.with_cafes_bars_accessible || 0,
                            without_cafes_bars: result.enhanced_metrics?.without_cafes_bars_accessible || 0,
                            with_transit: result.enhanced_metrics?.with_transit_accessible || 0,
                            without_transit: result.enhanced_metrics?.without_transit_accessible || 0,
                            with_nodes: result.with_park?.reachable_nodes || 0,
                            without_nodes: result.without_park?.reachable_nodes || 0,
                            node_difference: result.node_difference || 0,
                            with_edges: result.with_park?.street_network_stats?.edge_count || 0,
                            without_edges: result.without_park?.street_network_stats?.edge_count || 0,
                            edge_difference: result.edge_difference || 0,
                            impact_category: result.impact_category || 'neutral',
                            impact_label: result.impact_label || 'Neutral',
                            connectivity_with: result.enhanced_metrics?.with_connectivity || 0,
                            connectivity_without: result.enhanced_metrics?.without_connectivity || 0,
                            connectivity_change: result.enhanced_metrics?.connectivity_change || 0,
                            distance_with: result.enhanced_metrics?.with_avg_distance || 0,
                            distance_without: result.enhanced_metrics?.without_avg_distance || 0,
                            street_density_with: result.enhanced_metrics?.with_street_density || 0,
                            street_density_without: result.enhanced_metrics?.without_street_density || 0
                        });
                    }
                } else {
                    // Server returned an error response
                    const errorMessage = results?.detail || results?.message || `Server error ${response.status}`;
                    console.error('Server error response:', results);
                    throw new Error(`Batch analysis failed: ${errorMessage}`);
                }
            } catch (error) {
                console.error('Error in mass analysis:', error);
                alert(`Mass analysis failed: ${error.message}`);
            }

            // Complete progress and hide
            document.getElementById('massProgressBar').style.width = '100%';
            document.getElementById('massProgressText').textContent = `Completed analysis of ${massResultsData.length} nodes`;
            
            // Reset UI state
            document.getElementById('massAnalyzeBtn').disabled = false;
            document.getElementById('massCancelBtn').style.display = 'none';
            
            setTimeout(() => {
                document.getElementById('massProgress').style.display = 'none';
            }, 1000);
            
            // Check if we have any successful results
            const failedNodes = selectedNodes.length - massResultsData.length;
            if (failedNodes > 0 && massResultsData.length > 0) {
                console.warn(`${failedNodes} nodes failed during analysis`);
            } else if (massResultsData.length === 0) {
                alert(`Analysis failed: No successful results. All ${failedNodes} nodes failed. This is likely due to memory constraints or missing dependencies on the server. Check browser console and server logs for details.`);
                return;
            }
            
            displayMassResults();
        }

        function cancelMassAnalysis() {
            // Set global cancellation flag
            analysisAborted = true;
            
            // Reset UI immediately
            document.getElementById('massProgressBar').style.width = '0%';
            document.getElementById('massProgressText').textContent = 'Analysis cancelled by user';
            document.getElementById('massAnalyzeBtn').disabled = false;
            document.getElementById('massCancelBtn').style.display = 'none';
            
            // Hide progress after short delay
            setTimeout(() => {
                document.getElementById('massProgress').style.display = 'none';
            }, 1500);
            
            console.log('Mass analysis cancelled by user');
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
            const maxDifference = Math.max(...differences);
            const minDifference = Math.min(...differences);
            const avgWithArea = withAreas.reduce((a, b) => a + b, 0) / withAreas.length;
            const avgWithoutArea = withoutAreas.reduce((a, b) => a + b, 0) / withoutAreas.length;
            const avgWithStreetLength = withStreetLengths.reduce((a, b) => a + b, 0) / withStreetLengths.length;
            const avgWithoutStreetLength = withoutStreetLengths.reduce((a, b) => a + b, 0) / withoutStreetLengths.length;
            
            // Fix percentage calculation: should be (with - without) / without * 100
            const avgAreaChangePct = avgWithoutArea > 0 ? ((avgWithArea - avgWithoutArea) / avgWithoutArea * 100) : 0;
            const avgStreetNetworkChangePct = avgWithoutStreetLength > 0 ? ((avgWithStreetLength - avgWithoutStreetLength) / avgWithoutStreetLength * 100) : 0;

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
            // Clear the overview - we'll only show the averaged comparison table
            document.getElementById('massOverview').innerHTML = ``;

            // Calculate averaged amenity data
            const avgWithSupermarkets = massResultsData.reduce((sum, r) => sum + r.with_supermarkets, 0) / massResultsData.length;
            const avgWithoutSupermarkets = massResultsData.reduce((sum, r) => sum + r.without_supermarkets, 0) / massResultsData.length;
            const avgWithSchools = massResultsData.reduce((sum, r) => sum + r.with_schools, 0) / massResultsData.length;
            const avgWithoutSchools = massResultsData.reduce((sum, r) => sum + r.without_schools, 0) / massResultsData.length;
            const avgWithPlaygrounds = massResultsData.reduce((sum, r) => sum + r.with_playgrounds, 0) / massResultsData.length;
            const avgWithoutPlaygrounds = massResultsData.reduce((sum, r) => sum + r.without_playgrounds, 0) / massResultsData.length;
            const avgWithCafesBars = massResultsData.reduce((sum, r) => sum + r.with_cafes_bars, 0) / massResultsData.length;
            const avgWithoutCafesBars = massResultsData.reduce((sum, r) => sum + r.without_cafes_bars, 0) / massResultsData.length;
            const avgWithTransit = massResultsData.reduce((sum, r) => sum + r.with_transit, 0) / massResultsData.length;
            const avgWithoutTransit = massResultsData.reduce((sum, r) => sum + r.without_transit, 0) / massResultsData.length;

            // Calculate percentage changes for amenities
            const supermarketsChangePct = avgWithoutSupermarkets > 0 ? ((avgWithSupermarkets - avgWithoutSupermarkets) / avgWithoutSupermarkets * 100) : 0;
            const schoolsChangePct = avgWithoutSchools > 0 ? ((avgWithSchools - avgWithoutSchools) / avgWithoutSchools * 100) : 0;
            const playgroundsChangePct = avgWithoutPlaygrounds > 0 ? ((avgWithPlaygrounds - avgWithoutPlaygrounds) / avgWithoutPlaygrounds * 100) : 0;
            const cafesBarsChangePct = avgWithoutCafesBars > 0 ? ((avgWithCafesBars - avgWithoutCafesBars) / avgWithoutCafesBars * 100) : 0;
            const transitChangePct = avgWithoutTransit > 0 ? ((avgWithTransit - avgWithoutTransit) / avgWithoutTransit * 100) : 0;

            // Calculate statistics for all metrics
            const withAreaStats = calculateMetricStats(massResultsData.map(r => r.with_area));
            const withoutAreaStats = calculateMetricStats(massResultsData.map(r => r.without_area));
            const areaDiffStats = calculateMetricStats(massResultsData.map(r => r.difference));
            const areaChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_area;
                const with_ = r.with_area;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withStreetStats = calculateMetricStats(massResultsData.map(r => r.with_street_length));
            const withoutStreetStats = calculateMetricStats(massResultsData.map(r => r.without_street_length));
            const streetDiffStats = calculateMetricStats(massResultsData.map(r => r.street_network_difference));
            const streetChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_street_length;
                const with_ = r.with_street_length;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withSupermarketsStats = calculateMetricStats(massResultsData.map(r => r.with_supermarkets));
            const withoutSupermarketsStats = calculateMetricStats(massResultsData.map(r => r.without_supermarkets));
            const supermarketsChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_supermarkets;
                const with_ = r.with_supermarkets;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withSchoolsStats = calculateMetricStats(massResultsData.map(r => r.with_schools));
            const withoutSchoolsStats = calculateMetricStats(massResultsData.map(r => r.without_schools));
            const schoolsChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_schools;
                const with_ = r.with_schools;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withPlaygroundsStats = calculateMetricStats(massResultsData.map(r => r.with_playgrounds));
            const withoutPlaygroundsStats = calculateMetricStats(massResultsData.map(r => r.without_playgrounds));
            const playgroundsChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_playgrounds;
                const with_ = r.with_playgrounds;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withCafesBarsStats = calculateMetricStats(massResultsData.map(r => r.with_cafes_bars));
            const withoutCafesBarsStats = calculateMetricStats(massResultsData.map(r => r.without_cafes_bars));
            const cafesBarsChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_cafes_bars;
                const with_ = r.with_cafes_bars;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));
            
            const withTransitStats = calculateMetricStats(massResultsData.map(r => r.with_transit));
            const withoutTransitStats = calculateMetricStats(massResultsData.map(r => r.without_transit));
            const transitChangePctStats = calculateMetricStats(massResultsData.map(r => {
                const without = r.without_transit;
                const with_ = r.with_transit;
                return without > 0 ? ((with_ - without) / without * 100) : 0;
            }));

            // Pre-calculate all significance data to avoid function calls in template
            const areaSignificance = getSignificance(avgAreaChangePct);
            const streetSignificance = getSignificance(avgStreetNetworkChangePct);
            const supermarketsSignificance = getSignificance(supermarketsChangePct);
            const schoolsSignificance = getSignificance(schoolsChangePct);
            const playgroundsSignificance = getSignificance(playgroundsChangePct);
            const cafesBarsSignificance = getSignificance(cafesBarsChangePct);
            const transitSignificance = getSignificance(transitChangePct);
            
            const areaRange = getRangeText(avgAreaChangePct);
            const streetRange = getRangeText(avgStreetNetworkChangePct);
            const supermarketsRange = getRangeText(supermarketsChangePct);
            const schoolsRange = getRangeText(schoolsChangePct);
            const playgroundsRange = getRangeText(playgroundsChangePct);
            const cafesBarsRange = getRangeText(cafesBarsChangePct);
            const transitRange = getRangeText(transitChangePct);

            // Add averaged comparison table after overview
            const massOverviewElement = document.getElementById('massOverview');
            const averagedTableHTML = `
                <div style="margin-top: 30px;">
                    <h4>📊 Averaged Comparison (${massResultsData.length} nodes)</h4>
                    <div class="comparison-table-container">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Without Park</th>
                                    <th>With Park</th>
                                    <th>Difference</th>
                                    <th>Significance</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('area')">
                                    <td><span id="icon-area">▶</span> Accessible Area</td>
                                    <td>${avgWithoutArea.toFixed(3)} <span class="metric-unit">km²</span></td>
                                    <td>${avgWithArea.toFixed(3)} <span class="metric-unit">km²</span></td>
                                    <td class="${avgAreaChangePct > 0 ? 'impact-positive' : avgAreaChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${avgAreaChangePct > 0 ? '+' : ''}${avgAreaChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${areaSignificance.class}">
                                        <div class="tooltip">
                                            ${areaSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${areaSignificance.level} (${areaRange}%)</strong><br>
                                                ${areaSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-area" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutAreaStats, avgWithoutArea, ' km²')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withAreaStats, avgWithArea, ' km²')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(areaChangePctStats, avgAreaChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('street')">
                                    <td><span id="icon-street">▶</span> Street Network Length</td>
                                    <td>${avgWithoutStreetLength.toFixed(3)} <span class="metric-unit">km</span></td>
                                    <td>${avgWithStreetLength.toFixed(3)} <span class="metric-unit">km</span></td>
                                    <td class="${avgStreetNetworkChangePct > 0 ? 'impact-positive' : avgStreetNetworkChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${avgStreetNetworkChangePct > 0 ? '+' : ''}${avgStreetNetworkChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${streetSignificance.class}">
                                        <div class="tooltip">
                                            ${streetSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${streetSignificance.level} (${streetRange}%)</strong><br>
                                                ${streetSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-street" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutStreetStats, avgWithoutStreetLength, ' km')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withStreetStats, avgWithStreetLength, ' km')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(streetChangePctStats, avgStreetNetworkChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('supermarkets')">
                                    <td><span id="icon-supermarkets">▶</span> 🥖 Accessible Supermarkets</td>
                                    <td>${avgWithoutSupermarkets.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td>${avgWithSupermarkets.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td class="${supermarketsChangePct > 0 ? 'impact-positive' : supermarketsChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${supermarketsChangePct > 0 ? '+' : ''}${supermarketsChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${supermarketsSignificance.class}">
                                        <div class="tooltip">
                                            ${supermarketsSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${supermarketsSignificance.level} (${supermarketsRange}%)</strong><br>
                                                ${supermarketsSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-supermarkets" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutSupermarketsStats, avgWithoutSupermarkets, ' count')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withSupermarketsStats, avgWithSupermarkets, ' count')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(supermarketsChangePctStats, supermarketsChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('schools')">
                                    <td><span id="icon-schools">▶</span> 🏫 Accessible Schools</td>
                                    <td>${avgWithoutSchools.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td>${avgWithSchools.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td class="${schoolsChangePct > 0 ? 'impact-positive' : schoolsChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${schoolsChangePct > 0 ? '+' : ''}${schoolsChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${schoolsSignificance.class}">
                                        <div class="tooltip">
                                            ${schoolsSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${schoolsSignificance.level} (${schoolsRange}%)</strong><br>
                                                ${schoolsSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-schools" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutSchoolsStats, avgWithoutSchools, ' count')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withSchoolsStats, avgWithSchools, ' count')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(schoolsChangePctStats, schoolsChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('playgrounds')">
                                    <td><span id="icon-playgrounds">▶</span> 🛝 Accessible Playgrounds</td>
                                    <td>${avgWithoutPlaygrounds.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td>${avgWithPlaygrounds.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td class="${playgroundsChangePct > 0 ? 'impact-positive' : playgroundsChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${playgroundsChangePct > 0 ? '+' : ''}${playgroundsChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${playgroundsSignificance.class}">
                                        <div class="tooltip">
                                            ${playgroundsSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${playgroundsSignificance.level} (${playgroundsRange}%)</strong><br>
                                                ${playgroundsSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-playgrounds" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutPlaygroundsStats, avgWithoutPlaygrounds, ' count')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withPlaygroundsStats, avgWithPlaygrounds, ' count')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(playgroundsChangePctStats, playgroundsChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('cafesbars')">
                                    <td><span id="icon-cafesbars">▶</span> ☕ Accessible Cafes/Bars</td>
                                    <td>${avgWithoutCafesBars.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td>${avgWithCafesBars.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td class="${cafesBarsChangePct > 0 ? 'impact-positive' : cafesBarsChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${cafesBarsChangePct > 0 ? '+' : ''}${cafesBarsChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${cafesBarsSignificance.class}">
                                        <div class="tooltip">
                                            ${cafesBarsSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${cafesBarsSignificance.level} (${cafesBarsRange}%)</strong><br>
                                                ${cafesBarsSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-cafesbars" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutCafesBarsStats, avgWithoutCafesBars, ' count')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withCafesBarsStats, avgWithCafesBars, ' count')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(cafesBarsChangePctStats, cafesBarsChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                <tr style="cursor: pointer;" onclick="toggleMetricDetails('transit')">
                                    <td><span id="icon-transit">▶</span> 🚌 Accessible Transit Stops</td>
                                    <td>${avgWithoutTransit.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td>${avgWithTransit.toFixed(1)} <span class="metric-unit">avg</span></td>
                                    <td class="${transitChangePct > 0 ? 'impact-positive' : transitChangePct < 0 ? 'impact-negative' : 'impact-neutral'}">${transitChangePct > 0 ? '+' : ''}${transitChangePct.toFixed(1)} <span class="metric-unit">%</span></td>
                                    <td class="${transitSignificance.class}">
                                        <div class="tooltip">
                                            ${transitSignificance.level}
                                            <span class="tooltiptext">
                                                <strong>${transitSignificance.level} (${transitRange}%)</strong><br>
                                                ${transitSignificance.description}
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                                <tr id="details-transit" style="display: none;">
                                    <td colspan="5" style="padding: 15px; background: #f8f9fa; border-left: 3px solid #007bff;">
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                                            <div>
                                                <strong>Without Park Distribution</strong>
                                                ${createRangeBar(withoutTransitStats, avgWithoutTransit, ' count')}
                                            </div>
                                            <div>
                                                <strong>With Park Distribution</strong>
                                                ${createRangeBar(withTransitStats, avgWithTransit, ' count')}
                                            </div>
                                            <div>
                                                <strong>Change Distribution</strong>
                                                ${createRangeBar(transitChangePctStats, transitChangePct, '', true)}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            massOverviewElement.insertAdjacentHTML('afterend', averagedTableHTML);


            // Create interactive charts
            createMassAnalysisCharts(massResultsData, impactCounts);

            // Determine overall park impact based on street network change using significance levels
            const streetNetworkSignificance = getSignificance(avgStreetNetworkChangePct);
            let parkImpactText = '';
            let parkImpactClass = '';
            
            // Map significance levels to park impact descriptions
            switch(streetNetworkSignificance.level) {
                case 'Transformative':
                    if (avgStreetNetworkChangePct > 0) {
                        parkImpactText = '⚠️ <strong>HIGHLY PROBLEMATIC</strong> - This park creates major barriers to pedestrian movement, requiring significant detours.';
                        parkImpactClass = 'error';
                    } else {
                        parkImpactText = '🌟 <strong>TRANSFORMATIVE</strong> - This park dramatically improves street network connectivity with crucial pedestrian routes.';
                        parkImpactClass = 'success';
                    }
                    break;
                case 'High':
                    if (avgStreetNetworkChangePct > 0) {
                        parkImpactText = '🔴 <strong>PROBLEMATIC</strong> - This park significantly blocks optimal pedestrian routes and reduces accessibility.';
                        parkImpactClass = 'error';
                    } else {
                        parkImpactText = '🌟 <strong>HIGHLY BENEFICIAL</strong> - This park significantly improves street network accessibility with important connections.';
                        parkImpactClass = 'success';
                    }
                    break;
                case 'Strong':
                    if (avgStreetNetworkChangePct > 0) {
                        parkImpactText = '🔶 <strong>NEGATIVE IMPACT</strong> - This park noticeably reduces street network accessibility. Consider connectivity improvements.';
                        parkImpactClass = 'loading';
                    } else {
                        parkImpactText = '✅ <strong>BENEFICIAL</strong> - This park provides valuable pedestrian connections that improve accessibility.';
                        parkImpactClass = 'success';
                    }
                    break;
                case 'Notable':
                    if (avgStreetNetworkChangePct > 0) {
                        parkImpactText = '🔶 <strong>MINOR NEGATIVE</strong> - This park has some negative impact on pedestrian accessibility.';
                        parkImpactClass = 'loading';
                    } else {
                        parkImpactText = '✅ <strong>POSITIVE</strong> - This park modestly improves street network accessibility.';
                        parkImpactClass = 'success';
                    }
                    break;
                case 'Minimal':
                    parkImpactText = '🔸 <strong>MINIMAL IMPACT</strong> - This park has very small effects on street network accessibility.';
                    parkImpactClass = 'loading';
                    break;
                case 'No Change':
                default:
                    parkImpactText = '➖ <strong>NEUTRAL</strong> - This park has no measurable impact on street network accessibility.';
                    parkImpactClass = 'loading';
                    break;
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
            // Only show results if mass analysis tab is active
            if (document.getElementById('massAnalysisTab').classList.contains('active')) {
                document.getElementById('massResults').style.display = 'block';
            }
            
            // Scroll to results
            document.getElementById('massResults').scrollIntoView({ behavior: 'smooth' });
        }

        function createMassAnalysisCharts(data, impactCounts) {
            // Destroy existing charts
            Object.values(massCharts).forEach(chart => {
                if (chart) chart.destroy();
            });

            // 1. Mass Comparison Chart - WITH vs WITHOUT areas across all nodes
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
                        borderWidth: 1
                    }, {
                        label: 'WITHOUT Park (Area km²)',
                        data: data.map(d => d.without_area),
                        backgroundColor: 'rgba(220, 53, 69, 0.6)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1
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
                        title: { display: true, text: 'Area Comparison Across All Nodes' },
                        legend: { position: 'top' }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Analysis Points' } },
                        y: {
                            title: { display: true, text: 'Area (km²)' },
                            beginAtZero: true
                        }
                    }
                }
            });

            // 2. Impact Categories Pie Chart
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
        import numpy as np
        data = await analyzer.load_area_data(lat, lng, radius)
        # Format POI data for frontend
        pois = []
        poi_types = [
            ('supermarkets_gdf', '🥖', 'Supermarket'),
            ('schools_gdf', '🏫', 'School'), 
            ('playgrounds_gdf', '🛝', 'Playground'),
            ('cafes_bars_gdf', '☕', 'Cafe/Bar'),
            ('transit_gdf', '🚌', 'Transit')
        ]
        
        for key, emoji, type_name in poi_types:
            if key in data and data[key] is not None and not data[key].empty:
                for idx, poi in data[key].iterrows():
                    try:
                        if hasattr(poi.geometry, 'centroid'):
                            point = poi.geometry.centroid
                        else:
                            point = poi.geometry
                        
                        lat = float(point.y)
                        lng = float(point.x)
                        
                        # Validate coordinates are finite
                        if not (np.isfinite(lat) and np.isfinite(lng)):
                            continue
                            
                        name = poi.get('name', f'Unnamed {type_name}')
                        pois.append({
                            'id': f'{key}_{idx}',
                            'name': str(name),
                            'type': type_name,
                            'emoji': emoji,
                            'lat': lat,
                            'lng': lng
                        })
                    except Exception as e:
                        continue
        
        return {
            'parks': data['parks'],
            'nodes': data['nodes'],
            'center': data['center'],
            'pois': pois
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress")
async def get_progress(lat: float, lng: float, radius: int = 2000):
    """Get loading progress for a specific area"""
    progress_key = f"{lat:.4f}_{lng:.4f}_{radius}"
    progress = LOADING_PROGRESS.get(progress_key, {
        'current_step': 0,
        'total_steps': 15,
        'step_name': 'Waiting to start...',
        'sub_progress': 0,
        'current_operation': 'Initializing...',
        'estimated_time_remaining': None,
        'completed': False
    })
    return progress

@app.get("/mass-progress")
async def get_mass_progress(progress_key: str):
    """Get mass analysis progress for a specific batch"""
    progress = MASS_ANALYSIS_PROGRESS.get(progress_key, {
        'total_nodes': 0,
        'completed_nodes': 0,
        'current_node': 'Not found',
        'completed': True
    })
    
    # Include results if available
    return progress

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
        network_backend = request.get('network_backend', 'networkit')  # Default NetworkX
        
        result = await analyzer.calculate_accessibility(
            lat, lng, radius, park_id, node_id, walk_time, walk_speed, viz_method, network_backend
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mass-analyze")
async def mass_analyze_accessibility(request: dict):
    """Start batch analysis and return progress key immediately"""
    try:
        lat = request['lat']
        lng = request['lng'] 
        radius = request['radius']
        park_id = request['park_id']
        node_ids = request['node_ids']  # Array of node IDs
        walk_time = request.get('walk_time', 10.0)
        walk_speed = request.get('walk_speed', 4.5)
        viz_method = request.get('viz_method', 'convex_hull')
        network_backend = request.get('network_backend', 'networkit')
        
        # Generate progress key immediately
        import time
        progress_key = f"mass_{park_id}_{int(time.time())}"
        
        # Initialize progress tracking
        MASS_ANALYSIS_PROGRESS[progress_key] = {
            'total_nodes': len(node_ids),
            'completed_nodes': 0,
            'current_node': None,
            'completed': False,
            'start_time': time.time(),
            'results': None
        }
        
        # Start batch processing in background (non-blocking)
        asyncio.create_task(analyzer.calculate_mass_accessibility_background(
            lat, lng, radius, park_id, node_ids, walk_time, walk_speed, viz_method, network_backend, progress_key
        ))
        
        # Return progress key immediately
        return {"progress_key": progress_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)