"""
Advanced network analysis utilities for park impact assessment
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

try:
    import networkit as nk
    NETWORKIT_AVAILABLE = True
except ImportError:
    NETWORKIT_AVAILABLE = False
    logging.warning("NetworkIt not available. Falling back to NetworkX for centrality calculations.")

class NetworkAnalyzer:
    """Advanced network analysis for park accessibility impact"""
    
    def __init__(self, graph: nx.MultiDiGraph, parks: gpd.GeoDataFrame, config: Dict = None):
        self.original_graph = graph
        self.parks = parks
        self.config = config or {}
        
        # Convert to undirected for analysis
        self.graph_undirected = ox.convert.to_undirected(graph)
        self.nodes, self.edges = ox.graph_to_gdfs(graph)
        
        # Analysis results storage
        self.intersecting_edges = None
        self.filtered_graph = None
        self.centrality_results = {}
        
    def find_park_intersecting_edges(self) -> List[Tuple]:
        """Find edges that intersect with park areas"""
        print("Finding edges that intersect with parks...")
        
        # Ensure same CRS
        edges_projected = self.edges.to_crs('EPSG:3857')
        parks_projected = self.parks.to_crs('EPSG:3857')
        
        # Spatial join to find intersections
        intersecting = gpd.sjoin(edges_projected, parks_projected, 
                               how='inner', predicate='intersects')
        
        intersecting_edge_ids = intersecting.index.unique()
        self.intersecting_edges = intersecting_edge_ids
        
        print(f"Found {len(intersecting_edge_ids)} edges intersecting with parks")
        print(f"Intersection rate: {len(intersecting_edge_ids)/len(self.edges)*100:.2f}%")
        
        return intersecting_edge_ids
    
    def create_filtered_network(self) -> nx.Graph:
        """Create network with park-intersecting edges removed"""
        if self.intersecting_edges is None:
            self.find_park_intersecting_edges()
        
        print("Creating filtered network without park edges...")
        
        # Create copy of original graph
        filtered_graph = self.graph_undirected.copy()
        
        # Remove intersecting edges
        edges_to_remove = []
        for u, v, k in self.original_graph.edges(keys=True):
            edge_id = (u, v, k)
            if edge_id in self.intersecting_edges:
                # For undirected graph, we just need u, v
                edges_to_remove.append((u, v))
        
        # Remove duplicate tuples (since undirected)
        edges_to_remove = list(set(edges_to_remove))
        filtered_graph.remove_edges_from(edges_to_remove)
        
        self.filtered_graph = filtered_graph
        
        print(f"Original edges: {len(self.graph_undirected.edges())}")
        print(f"Filtered edges: {len(filtered_graph.edges())}")
        print(f"Removed: {len(edges_to_remove)} edge connections")
        
        return filtered_graph
    
    def calculate_centralities_networkx(self, graph: nx.Graph, sample_size: int = 1000) -> Dict:
        """Calculate centrality measures using NetworkX (fallback method)"""
        print(f"Calculating centralities with NetworkX (sampling {sample_size} nodes)...")
        
        # For large networks, use sampling for betweenness
        betweenness = nx.betweenness_centrality(graph, k=min(sample_size, len(graph.nodes())), 
                                              seed=42, normalized=True)
        
        # Closeness and degree can handle full network
        closeness = nx.closeness_centrality(graph, wf_improved=True)
        degree = nx.degree_centrality(graph)
        
        # PageRank as alternative to eigenvector centrality
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=1000)
        
        return {
            'betweenness': betweenness,
            'closeness': closeness, 
            'degree': degree,
            'pagerank': pagerank
        }
    
    def calculate_centralities_networkit(self, graph: nx.Graph) -> Dict:
        """Calculate centrality measures using NetworkIt (faster)"""
        print("Converting to NetworkIt format for fast centrality calculation...")
        
        # Create node mapping
        nodes = list(graph.nodes())
        node_map = {node: idx for idx, node in enumerate(nodes)}
        
        # Create NetworkIt graph
        nk_graph = nk.Graph(n=len(nodes), weighted=True, directed=False)
        
        # Add edges
        for u, v, data in graph.edges(data=True):
            u_idx = node_map[u]
            v_idx = node_map[v]
            weight = data.get('length', 100.0)
            nk_graph.addEdge(u_idx, v_idx, weight)
        
        # Calculate centralities
        print("Computing betweenness centrality...")
        bc = nk.centrality.Betweenness(nk_graph, normalized=True)
        bc.run()
        betweenness_scores = bc.scores()
        
        print("Computing closeness centrality...")
        cc = nk.centrality.Closeness(nk_graph, normalized=True)
        cc.run()
        closeness_scores = cc.scores()
        
        print("Computing PageRank...")
        pr = nk.centrality.PageRank(nk_graph)
        pr.run()
        pagerank_scores = pr.scores()
        
        # Degree centrality
        degree_scores = [nk_graph.degree(v) / (nk_graph.numberOfNodes() - 1) 
                        for v in nk_graph.iterNodes()]
        
        # Convert back to node IDs
        betweenness = {nodes[i]: score for i, score in enumerate(betweenness_scores)}
        closeness = {nodes[i]: score for i, score in enumerate(closeness_scores)}
        degree = {nodes[i]: score for i, score in enumerate(degree_scores)}
        pagerank = {nodes[i]: score for i, score in enumerate(pagerank_scores)}
        
        return {
            'betweenness': betweenness,
            'closeness': closeness,
            'degree': degree, 
            'pagerank': pagerank
        }
    
    def analyze_network_centralities(self) -> Dict:
        """Analyze centrality changes between original and filtered networks"""
        print("\\n" + "="*60)
        print("NETWORK CENTRALITY ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        # Create filtered network if not exists
        if self.filtered_graph is None:
            self.create_filtered_network()
        
        # Choose centrality calculation method
        if NETWORKIT_AVAILABLE and len(self.graph_undirected.nodes()) > 1000:
            print("Using NetworkIt for fast centrality calculations...")
            original_centralities = self.calculate_centralities_networkit(self.graph_undirected)
            filtered_centralities = self.calculate_centralities_networkit(self.filtered_graph)
        else:
            print("Using NetworkX for centrality calculations...")
            sample_size = self.config.get('centrality_k', 1000)
            original_centralities = self.calculate_centralities_networkx(self.graph_undirected, sample_size)
            filtered_centralities = self.calculate_centralities_networkx(self.filtered_graph, sample_size)
        
        # Store results
        self.centrality_results = {
            'original': original_centralities,
            'filtered': filtered_centralities
        }
        
        # Calculate comparison metrics
        comparison_data = self.compare_centralities(original_centralities, filtered_centralities)
        
        elapsed = time.time() - start_time
        print(f"\\nCentrality analysis completed in {elapsed:.2f} seconds")
        
        return comparison_data
    
    def compare_centralities(self, original: Dict, filtered: Dict) -> pd.DataFrame:
        """Compare centrality measures between networks"""
        print("Comparing centrality measures...")
        
        # Find common nodes
        common_nodes = set(original['betweenness'].keys()) & set(filtered['betweenness'].keys())
        
        comparison_data = []
        for node in common_nodes:
            comparison_data.append({
                'node': node,
                'orig_betweenness': original['betweenness'].get(node, 0),
                'filt_betweenness': filtered['betweenness'].get(node, 0), 
                'orig_closeness': original['closeness'].get(node, 0),
                'filt_closeness': filtered['closeness'].get(node, 0),
                'orig_degree': original['degree'].get(node, 0),
                'filt_degree': filtered['degree'].get(node, 0),
                'orig_pagerank': original['pagerank'].get(node, 0),
                'filt_pagerank': filtered['pagerank'].get(node, 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate differences
        df['betweenness_diff'] = df['filt_betweenness'] - df['orig_betweenness']
        df['closeness_diff'] = df['filt_closeness'] - df['orig_closeness'] 
        df['degree_diff'] = df['filt_degree'] - df['orig_degree']
        df['pagerank_diff'] = df['filt_pagerank'] - df['orig_pagerank']
        
        # Summary statistics
        print(f"\\nCentrality Change Summary:")
        print(f"Nodes compared: {len(df)}")
        print(f"Mean betweenness change: {df['betweenness_diff'].mean():.8f}")
        print(f"Mean closeness change: {df['closeness_diff'].mean():.8f}")
        print(f"Nodes with increased betweenness: {len(df[df['betweenness_diff'] > 0])}")
        print(f"Nodes with increased closeness: {len(df[df['closeness_diff'] > 0])}")
        
        return df
    
    def analyze_edge_importance(self, max_edges: int = 50) -> List[Dict]:
        """Analyze importance of individual park-intersecting edges"""
        print(f"\\nAnalyzing importance of up to {max_edges} park edges...")
        
        if self.intersecting_edges is None:
            self.find_park_intersecting_edges()
        
        # Limit analysis for performance
        edges_to_analyze = list(self.intersecting_edges)[:max_edges]
        edge_importance = []
        
        # Original network metrics
        original_components = nx.number_connected_components(self.graph_undirected)
        original_largest = len(max(nx.connected_components(self.graph_undirected), key=len))
        
        print(f"Original network: {original_components} components, largest: {original_largest} nodes")
        
        for i, edge_id in enumerate(edges_to_analyze):
            if i % 10 == 0:
                print(f"  Analyzing edge {i+1}/{len(edges_to_analyze)}")
            
            # Convert edge format
            u, v = edge_id[0], edge_id[1]
            
            if self.graph_undirected.has_edge(u, v):
                # Store and remove edge
                edge_data = dict(self.graph_undirected[u][v])
                self.graph_undirected.remove_edge(u, v)
                
                # Calculate impact
                new_components = nx.number_connected_components(self.graph_undirected)
                new_largest = len(max(nx.connected_components(self.graph_undirected), key=len))
                
                # Impact metrics
                component_impact = new_components - original_components
                connectivity_impact = original_largest - new_largest
                
                edge_importance.append({
                    'edge': (u, v),
                    'component_impact': component_impact,
                    'connectivity_impact': connectivity_impact,
                    'is_bridge': component_impact > 0,
                    'importance_score': component_impact * 100 + connectivity_impact
                })
                
                # Restore edge
                self.graph_undirected.add_edge(u, v, **{k: v for k, v in edge_data.items() 
                                                       if isinstance(k, str)})
        
        # Sort by importance
        edge_importance.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Summary
        bridge_edges = [e for e in edge_importance if e['is_bridge']]
        print(f"\\nEdge Importance Analysis:")
        print(f"Total edges analyzed: {len(edge_importance)}")
        print(f"Bridge edges (critical): {len(bridge_edges)}")
        
        return edge_importance
    
    def calculate_accessibility_metrics(self, walking_time_minutes: int = 15) -> Dict:
        """Calculate accessibility metrics for walking catchments"""
        print(f"\\nCalculating {walking_time_minutes}-minute walking accessibility...")
        
        # Convert time to meters (rough approximation: 4.5 km/h walking speed)
        walking_speed_ms = (4.5 * 1000) / 60  # meters per minute
        max_distance = walking_time_minutes * walking_speed_ms
        
        # Sample center nodes for analysis
        center_nodes = list(self.graph_undirected.nodes())[:100]  # Sample for performance
        
        accessibility_original = []
        accessibility_filtered = []
        
        for node in center_nodes:
            # Original network accessibility
            try:
                ego_orig = nx.ego_graph(self.graph_undirected, node, radius=max_distance, 
                                      distance='length')
                accessible_nodes_orig = len(ego_orig.nodes())
            except:
                accessible_nodes_orig = 1
            
            # Filtered network accessibility
            if self.filtered_graph and node in self.filtered_graph:
                try:
                    ego_filt = nx.ego_graph(self.filtered_graph, node, radius=max_distance,
                                          distance='length')
                    accessible_nodes_filt = len(ego_filt.nodes())
                except:
                    accessible_nodes_filt = 1
            else:
                accessible_nodes_filt = 0
            
            accessibility_original.append(accessible_nodes_orig)
            accessibility_filtered.append(accessible_nodes_filt)
        
        # Calculate metrics
        mean_access_orig = np.mean(accessibility_original)
        mean_access_filt = np.mean(accessibility_filtered)
        access_reduction = ((mean_access_orig - mean_access_filt) / mean_access_orig) * 100
        
        print(f"Mean accessibility (original): {mean_access_orig:.1f} nodes")
        print(f"Mean accessibility (filtered): {mean_access_filt:.1f} nodes")
        print(f"Accessibility reduction: {access_reduction:.1f}%")
        
        return {
            'walking_time_minutes': walking_time_minutes,
            'mean_accessibility_original': mean_access_orig,
            'mean_accessibility_filtered': mean_access_filt,
            'accessibility_reduction_percent': access_reduction,
            'sample_nodes': len(center_nodes)
        }
    
    def generate_analysis_summary(self) -> Dict:
        """Generate comprehensive analysis summary"""
        print("\\n" + "="*60)
        print("GENERATING ANALYSIS SUMMARY")
        print("="*60)
        
        summary = {
            'network_stats': {
                'total_nodes': len(self.graph_undirected.nodes()),
                'total_edges': len(self.graph_undirected.edges()),
                'total_parks': len(self.parks),
                'intersecting_edges': len(self.intersecting_edges) if self.intersecting_edges else 0,
                'intersection_rate': (len(self.intersecting_edges) / len(self.graph_undirected.edges()) * 100) 
                                   if self.intersecting_edges else 0
            },
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tools_used': ['OSMnx', 'NetworkX'] + (['NetworkIt'] if NETWORKIT_AVAILABLE else [])
        }
        
        if self.centrality_results:
            summary['centrality_analysis'] = True
        
        return summary