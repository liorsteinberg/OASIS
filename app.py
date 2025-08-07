import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json

# Try to import NetworkKit for performance optimization
try:
    import networkit as nk
    NETWORKIT_AVAILABLE = True
except ImportError:
    NETWORKIT_AVAILABLE = False
    nk = None

st.set_page_config(
    page_title="OASIS - Park Accessibility Impact System",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OasisApp:
    def __init__(self):
        self.initialize_session_state()
        self._performance_metrics = {}
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """Log performance metrics for analysis"""
        timestamp = time.time()
        self._performance_metrics[operation] = {
            'duration': duration,
            'timestamp': timestamp,
            'details': details or {}
        }
        
        # Keep only last 10 operations to avoid memory bloat
        if len(self._performance_metrics) > 10:
            oldest_key = min(self._performance_metrics.keys(), 
                           key=lambda k: self._performance_metrics[k]['timestamp'])
            del self._performance_metrics[oldest_key]
    
    def initialize_session_state(self):
        if 'selected_location' not in st.session_state:
            st.session_state.selected_location = None
        if 'park_data' not in st.session_state:
            st.session_state.park_data = None
        if 'selected_parks' not in st.session_state:
            st.session_state.selected_parks = set()
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'parks_loaded' not in st.session_state:
            st.session_state.parks_loaded = False
        # Initialize NetworkKit cache for session
        if 'nk_graph_cache' not in st.session_state:
            st.session_state.nk_graph_cache = {}
        if 'accessibility_cache' not in st.session_state:
            st.session_state.accessibility_cache = {}
    
    def render_header(self):
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🌳 OASIS</h1>
            <h3 style='color: #666; margin-bottom: 2rem;'>Open Accessibility Spatial Impact System</h3>
            <p style='font-size: 1.2rem; color: #888;'>
                Discover the hidden impact of parks on urban connectivity
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_progress_bar(self):
        steps = ["Choose Park", "Set Parameters", "Analyze Impact", "View Results", "Share Report"]
        current = st.session_state.current_step
        
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                step_num = i + 1
                if step_num < current:
                    # Completed steps - clickable
                    if st.button(f"✅ {step}", key=f"step_{step_num}", help="Click to go back to this step"):
                        st.session_state.current_step = step_num
                        st.rerun()
                elif step_num == current:
                    st.info(f"🔄 {step}")
                else:
                    # Future steps - not clickable yet
                    if step_num == current + 1 and self.can_proceed_to_next_step():
                        if st.button(f"⏭️ {step}", key=f"step_{step_num}", help="Click to proceed"):
                            st.session_state.current_step = step_num
                            st.rerun()
                    else:
                        st.write(f"⏳ {step}")
    
    def can_proceed_to_next_step(self):
        """Check if user can proceed to next step"""
        current = st.session_state.current_step
        if current == 1:
            return len(st.session_state.selected_parks) > 0
        elif current == 2:
            return 'analysis_params' in st.session_state
        elif current == 3:
            return st.session_state.analysis_complete
        elif current == 4:
            return st.session_state.analysis_results is not None
        return True
    
    def render_map_selection(self):
        st.subheader("🗺️ Step 1: Choose Your Parks")
        
        # Auto-load parks for default location on first visit
        if not st.session_state.parks_loaded:
            st.write("Loading parks in the area...")
            with st.spinner("Finding parks and green spaces..."):
                default_lat, default_lon = 51.9225, 4.47917
                parks = self.find_nearby_parks(default_lat, default_lon)
                if parks:
                    st.session_state.park_data = parks
                    st.session_state.selected_location = (default_lat, default_lon)
                    st.session_state.parks_loaded = True
                    st.rerun()
        
        if not st.session_state.park_data:
            st.error("Could not load parks. Please try refreshing the page.")
            return
            
        # Show instructions
        st.write("**Click on park polygons to select them for analysis.** Selected parks will turn orange.")
        
        # Create map with parks
        lat, lng = st.session_state.selected_location
        m = folium.Map(location=[lat, lng], zoom_start=13)
        
        # Add tile layer options
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        parks = st.session_state.park_data['parks']
        
        # Create a mapping of simple IDs to complex park indices
        if 'park_id_mapping' not in st.session_state:
            st.session_state.park_id_mapping = {}
            simple_id = 0
            for idx in parks.index:
                st.session_state.park_id_mapping[simple_id] = idx
                simple_id += 1
        
        # Reverse mapping for quick lookup
        reverse_mapping = {v: k for k, v in st.session_state.park_id_mapping.items()}
        
        # Add parks as clickable polygons
        for idx, row in parks.iterrows():
            if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                # Get simple ID for this park
                simple_id = reverse_mapping[idx]
                
                # Determine color based on selection
                is_selected = idx in st.session_state.selected_parks
                color = 'orange' if is_selected else 'green'
                fill_color = 'orange' if is_selected else 'green'
                fill_opacity = 0.7 if is_selected else 0.4
                
                # Handle both Polygon and MultiPolygon
                if row.geometry.geom_type == 'Polygon':
                    coords = [[point[1], point[0]] for point in row.geometry.exterior.coords]
                    folium.Polygon(
                        coords, 
                        color=color, 
                        fillColor=fill_color,
                        fillOpacity=fill_opacity,
                        weight=3 if is_selected else 2,
                        popup=f"PARK_{simple_id}",
                        tooltip=f"Park {simple_id} {'(Selected)' if is_selected else '(Click to select)'}"
                    ).add_to(m)
                else:  # MultiPolygon
                    for polygon in row.geometry.geoms:
                        coords = [[point[1], point[0]] for point in polygon.exterior.coords]
                        folium.Polygon(
                            coords, 
                            color=color, 
                            fillColor=fill_color,
                            fillOpacity=fill_opacity,
                            weight=3 if is_selected else 2,
                            popup=f"PARK_{simple_id}",
                            tooltip=f"Park {simple_id} {'(Selected)' if is_selected else '(Click to select)'}"
                        ).add_to(m)
        
        # Render map and capture clicks
        map_data = st_folium(m, width=700, height=500, key="park_selection_map")
        
        # Handle polygon clicks
        if map_data['last_object_clicked_popup']:
            clicked_popup = map_data['last_object_clicked_popup']
            # Extract park index from popup text (format: PARK_{simple_id})
            if 'PARK_' in clicked_popup:
                try:
                    # Extract simple ID
                    simple_id = int(clicked_popup.split('PARK_')[1])
                    
                    # Get the actual park index from mapping
                    actual_park_idx = st.session_state.park_id_mapping[simple_id]
                    
                    # Toggle selection
                    if actual_park_idx in st.session_state.selected_parks:
                        st.session_state.selected_parks.remove(actual_park_idx)
                        st.success(f"Park {simple_id} deselected")
                    else:
                        st.session_state.selected_parks.add(actual_park_idx)
                        st.success(f"Park {simple_id} selected")
                    
                    time.sleep(0.5)  # Brief pause to show message
                    st.rerun()
                except (ValueError, KeyError, IndexError):
                    st.error("Could not parse park selection. Please try again.")
        
        # Show selection status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Parks Found", len(parks))
        with col2:
            st.metric("Parks Selected", len(st.session_state.selected_parks))
        with col3:
            if len(st.session_state.selected_parks) > 0:
                if st.button("Start Analysis", type="primary"):
                    st.session_state.current_step = 2
                    st.rerun()
            else:
                st.info("Select at least one park to continue")
        
        # Show selected parks list
        if st.session_state.selected_parks:
            st.write("**Selected Parks:**")
            # Convert actual indices back to simple IDs for display
            reverse_mapping = {v: k for k, v in st.session_state.park_id_mapping.items()}
            selected_simple_ids = [reverse_mapping[idx] for idx in st.session_state.selected_parks if idx in reverse_mapping]
            selected_list = ", ".join([f"Park {simple_id}" for simple_id in sorted(selected_simple_ids)])
            st.write(selected_list)
            
            # Option to clear selection
            if st.button("Clear Selection"):
                st.session_state.selected_parks.clear()
                st.rerun()
                
        # Option to change location
        st.markdown("---")
        
        # Performance testing section
        with st.expander("🔧 Advanced Options & Performance Testing"):
            st.markdown("**NetworkKit Performance Test**")
            st.write("Test NetworkKit optimization on your current network:")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Quick Benchmark", help="Test NetworkKit vs NetworkX performance"):
                    if st.session_state.park_data:
                        self.run_quick_benchmark()
                    else:
                        st.error("Please load parks first")
            
            with col2:
                if st.button("📊 Show Performance History"):
                    if hasattr(self, '_performance_metrics') and self._performance_metrics:
                        st.write("Recent performance metrics:")
                        for op, metrics in self._performance_metrics.items():
                            st.write(f"• {op}: {metrics['duration']:.2f}s")
                    else:
                        st.info("No performance metrics available yet")
        
        st.markdown("---")
        if st.button("🌍 Change Location"):
            self.show_location_selector()
    
    def show_location_selector(self):
        """Show location selection interface"""
        st.subheader("🌍 Select New Location")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lat = st.number_input("Latitude", value=51.9225, format="%.6f")
        with col2:
            lng = st.number_input("Longitude", value=4.47917, format="%.6f")
        
        # Sample of popular cities
        st.write("**Or choose a city:**")
        cities = {
            "Rotterdam, Netherlands": (51.9225, 4.47917),
            "Amsterdam, Netherlands": (52.3676, 4.9041),
            "New York, USA": (40.7128, -74.0060),
            "London, UK": (51.5074, -0.1278),
            "Paris, France": (48.8566, 2.3522),
            "Berlin, Germany": (52.5200, 13.4050)
        }
        
        selected_city = st.selectbox("Select a city", ["Custom coordinates"] + list(cities.keys()))
        
        if selected_city != "Custom coordinates":
            lat, lng = cities[selected_city]
            st.write(f"Selected: {selected_city} ({lat:.4f}, {lng:.4f})")
        
        if st.button("Load Parks for This Location", type="primary"):
            with st.spinner("Loading parks for new location..."):
                parks = self.find_nearby_parks(lat, lng)
                if parks:
                    st.session_state.park_data = parks
                    st.session_state.selected_location = (lat, lng)
                    st.session_state.selected_parks.clear()  # Clear previous selections
                    st.session_state.parks_loaded = True
                    st.success(f"Loaded parks for location: {lat:.4f}, {lng:.4f}")
                    st.rerun()
                else:
                    st.error("Could not find parks at this location. Try a different area.")
    
    def find_nearby_parks(self, lat: float, lng: float, distance: int = 2000) -> Dict:
        """Find parks near the selected location"""
        try:
            # Get parks and green spaces first
            tags = {"leisure": ["park", "recreation_ground"], "landuse": ["grass", "recreation_ground"]}
            parks = ox.features_from_point((lat, lng), tags=tags, dist=distance)
            
            return {
                'parks': parks,
                'center': (lat, lng),
                'distance': distance
            }
        except Exception as e:
            st.error(f"Error finding parks: {str(e)}")
            return None
    
    def run_quick_benchmark(self):
        """Run a quick benchmark test of NetworkKit vs NetworkX"""
        st.write("🔬 **Running NetworkKit Performance Benchmark**")
        
        try:
            # Get a small sample network for testing
            network_data = self.get_network_around_selected_parks()
            if not network_data:
                st.error("Could not load network for benchmarking")
                return
            
            graph = network_data['graph']
            graph_undirected = ox.convert.to_undirected(graph)
            nodes, edges = ox.graph_to_gdfs(graph)
            
            # Use a small sample for quick test
            all_nodes = list(graph_undirected.nodes())
            sample_size = min(20, len(all_nodes))
            test_nodes = all_nodes[:sample_size]
            
            max_distance = 15 * ((4.5 * 1000) / 60)  # 15 min walk
            
            st.write(f"Testing with {sample_size} nodes from a network of {len(all_nodes)} nodes")
            
            # NetworkX benchmark
            st.write("Testing NetworkX...")
            nx_start = time.time()
            nx_results = {}
            
            for node in test_nodes:
                try:
                    ego_graph = nx.ego_graph(graph_undirected, node, radius=max_distance, distance='length')
                    reachable_count = len([n for n in ego_graph.nodes() if n != node])
                    nx_results[node] = reachable_count
                except:
                    nx_results[node] = 0
            
            nx_time = time.time() - nx_start
            
            # NetworkKit benchmark (if available)
            if NETWORKIT_AVAILABLE:
                st.write("Testing NetworkKit...")
                nk_start = time.time()
                
                # Convert and test
                nk_graph, node_to_nk, nk_to_node = self.convert_networkx_to_networkit(graph_undirected)
                nk_results = {}
                
                for node in test_nodes:
                    try:
                        nk_node = node_to_nk[node]
                        dijkstra = nk.distance.Dijkstra(nk_graph, nk_node)
                        dijkstra.run()
                        
                        reachable_count = 0
                        for target_nk in range(nk_graph.numberOfNodes()):
                            if target_nk != nk_node:
                                try:
                                    if dijkstra.distance(target_nk) <= max_distance:
                                        reachable_count += 1
                                except:
                                    continue
                        
                        nk_results[node] = reachable_count
                    except:
                        nk_results[node] = 0
                
                nk_time = time.time() - nk_start
                
                # Show results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("NetworkX Time", f"{nx_time:.2f}s")
                    
                with col2:
                    st.metric("NetworkKit Time", f"{nk_time:.2f}s")
                    
                with col3:
                    speedup = nx_time / nk_time if nk_time > 0 else float('inf')
                    st.metric("Speedup", f"{speedup:.1f}x")
                
                # Accuracy check
                accuracy_check = []
                for node in test_nodes:
                    if node in nx_results and node in nk_results:
                        diff = abs(nx_results[node] - nk_results[node])
                        accuracy_check.append(diff)
                
                avg_diff = np.mean(accuracy_check) if accuracy_check else 0
                
                if speedup > 1.2:
                    st.success(f"🚀 NetworkKit is {speedup:.1f}x faster! Average difference: {avg_diff:.1f} nodes")
                elif speedup > 0.8:
                    st.info(f"⚖️ Performance is similar. Average difference: {avg_diff:.1f} nodes")
                else:
                    st.warning(f"🐌 NetworkX was faster for this small test. Difference: {avg_diff:.1f} nodes")
                
                # Log performance
                self.log_performance(f"Benchmark_Sample_{sample_size}", nk_time, {
                    'nx_time': nx_time,
                    'speedup': speedup,
                    'sample_size': sample_size,
                    'total_network_size': len(all_nodes),
                    'avg_accuracy_diff': avg_diff
                })
                
            else:
                st.error("NetworkKit not available for benchmarking")
                st.metric("NetworkX Time", f"{nx_time:.2f}s")
                
        except Exception as e:
            st.error(f"Benchmark failed: {str(e)}")
    
    def get_network_around_selected_parks(self) -> Dict:
        """Get street network around selected parks only"""
        if not st.session_state.selected_parks or not st.session_state.park_data:
            return None
        
        try:
            # Get selected parks
            all_parks = st.session_state.park_data['parks']
            selected_parks = all_parks.loc[list(st.session_state.selected_parks)]
            
            # Calculate bounding box of selected parks
            bounds = selected_parks.total_bounds  # [minx, miny, maxx, maxy]
            
            # Add buffer around selected parks (500m)
            buffer = 0.02  # approximately 2km in degrees
            north = bounds[3] + buffer
            south = bounds[1] - buffer
            east = bounds[2] + buffer
            west = bounds[0] - buffer
            
            # Get street network for the bounding box of selected parks
            # OSMnx expects bbox as (left, bottom, right, top)
            bbox = (west, south, east, north)
            graph = ox.graph_from_bbox(bbox, network_type='walk')
            
            return {
                'graph': graph,
                'selected_parks': selected_parks,
                'bounds': bounds
            }
        except Exception as e:
            st.error(f"Error getting network around selected parks: {str(e)}")
            return None
    
    def render_parameter_selection(self):
        st.subheader("⚙️ Step 2: Set Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Walking Parameters**")
            walking_speed = st.slider("Walking Speed (km/h)", 2.0, 8.0, 4.5, 0.5)
            max_time = st.slider("Maximum Walking Time (minutes)", 5, 30, 15, 5)
            
        with col2:
            st.write("**Analysis Options**")
            analysis_depth = st.selectbox("Analysis Depth", 
                ["Quick Preview", "Standard Analysis", "Deep Analysis"])
            include_demographics = st.checkbox("Include Demographics", value=True)
            
            # Node sampling option for performance
            st.markdown("**🚀 Performance Settings**")
            use_node_sampling = st.checkbox("Use Node Sampling", value=True, 
                                           help="Sample nodes for faster analysis. Uncheck to analyze all nodes (slower but complete).")
            
            if use_node_sampling:
                sample_size = st.slider("Sample Size", 50, 500, 200, 25,
                                       help="Number of nodes to analyze. More nodes = more accurate but slower.")
                st.info(f"Will analyze ~{sample_size} sampled nodes for faster performance")
            else:
                st.warning("⚠️ Analyzing all nodes will take significantly longer (5-10x slower)")
                sample_size = None
        
        # Show selected parks preview
        if st.session_state.park_data and st.session_state.selected_parks:
            st.write("**Selected Parks Preview:**")
            parks = st.session_state.park_data['parks']
            selected_parks = parks.loc[list(st.session_state.selected_parks)]
            
            st.write(f"Analyzing {len(st.session_state.selected_parks)} selected parks")
            
            # Create preview map
            center = st.session_state.park_data['center']
            preview_map = folium.Map(location=center, zoom_start=13)
            
            # Add only selected parks
            for idx, row in selected_parks.iterrows():
                if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    if row.geometry.geom_type == 'Polygon':
                        coords = [[point[1], point[0]] for point in row.geometry.exterior.coords]
                        folium.Polygon(coords, color='orange', fillColor='orange', 
                                     fillOpacity=0.6, popup=f"Selected Park {idx}").add_to(preview_map)
                    else:  # MultiPolygon
                        for polygon in row.geometry.geoms:
                            coords = [[point[1], point[0]] for point in polygon.exterior.coords]
                            folium.Polygon(coords, color='orange', fillColor='orange', 
                                         fillOpacity=0.6, popup=f"Selected Park {idx}").add_to(preview_map)
            
            st_folium(preview_map, width=700, height=300, key="park_preview")
        else:
            st.warning("No parks selected for analysis.")
        
        if st.button("Start Analysis", type="primary"):
            st.session_state.current_step = 3
            # Store parameters
            st.session_state.analysis_params = {
                'walking_speed': walking_speed,
                'max_time': max_time,
                'analysis_depth': analysis_depth,
                'include_demographics': include_demographics,
                'use_node_sampling': use_node_sampling,
                'sample_size': sample_size
            }
            st.rerun()
    
    def render_analysis_progress(self):
        st.subheader("🔍 Step 3: Analyzing Park Impact")
        st.write("Watch the magic happen as we analyze how parks affect urban connectivity...")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis steps
        steps = [
            ("Loading street network", 20),
            ("Identifying park connections", 40),
            ("Calculating original accessibility", 60),
            ("Simulating park removal", 80),
            ("Computing impact metrics", 90),
            ("Generating visualizations", 100)
        ]
        
        for i, (step_name, progress) in enumerate(steps):
            status_text.text(f"{step_name}...")
            progress_bar.progress(progress)
            time.sleep(1)  # Simulate processing time
        
        # Get network around selected parks
        with st.spinner("Loading street network around selected parks..."):
            network_data = self.get_network_around_selected_parks()
            if not network_data:
                st.error("Could not load street network. Please try again.")
                return
        
        # Run actual analysis
        with st.spinner("Running detailed analysis..."):
            results = self.run_park_analysis(network_data)
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True
            st.session_state.current_step = 4
        
        st.success("Analysis complete! 🎉")
        st.rerun()
    
    def run_park_analysis(self, network_data: Dict) -> Dict:
        """Run the park impact analysis on selected parks only"""
        selected_park_indices = st.session_state.selected_parks
        
        if not network_data or not selected_park_indices:
            return None
        
        graph = network_data['graph']
        selected_parks = network_data['selected_parks']
        
        # Convert to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(graph)
        
        # Find intersecting edges with selected parks only
        edges_projected = edges.to_crs('EPSG:3857')
        parks_projected = selected_parks.to_crs('EPSG:3857')
        
        intersecting_edges = gpd.sjoin(edges_projected, parks_projected, 
                                     how='inner', predicate='intersects')
        
        # Basic analysis
        total_edges = len(edges)
        intersecting_count = len(intersecting_edges)
        impact_percentage = (intersecting_count / total_edges) * 100 if total_edges > 0 else 0
        
        return {
            'total_edges': total_edges,
            'intersecting_edges': intersecting_count,
            'impact_percentage': impact_percentage,
            'graph': graph,
            'nodes': nodes,
            'edges': edges,
            'parks': selected_parks,  # Use only selected parks
            'selected_park_indices': selected_park_indices,
            'intersecting_edge_ids': intersecting_edges.index.unique(),
            'network_bounds': network_data['bounds']
        }
    
    def render_results(self):
        st.subheader("📊 Step 4: Discover the Impact")
        
        if not st.session_state.analysis_results:
            st.error("No analysis results available.")
            return
        
        results = st.session_state.analysis_results
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Street Segments", f"{results['total_edges']:,}")
        
        with col2:
            st.metric("Segments Through Parks", f"{results['intersecting_edges']:,}")
        
        with col3:
            st.metric("Impact Percentage", f"{results['impact_percentage']:.1f}%")
        
        with col4:
            st.metric("Parks Analyzed", len(results['selected_park_indices']))
        
        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Impact Map", "🚶‍♀️ Accessibility Analysis", "📈 Network Analysis", "📋 Detailed Results"])
        
        with tab1:
            self.render_impact_map(results)
        
        with tab2:
            self.render_accessibility_comparison(results)
        
        with tab3:
            self.render_network_analysis(results)
        
        with tab4:
            self.render_detailed_results(results)
        
        if st.button("Generate Report", type="primary"):
            st.session_state.current_step = 5
            st.rerun()
    
    def render_impact_map(self, results: Dict):
        st.write("**Interactive Impact Visualization**")
        
        # Create map
        center = st.session_state.park_data['center']
        impact_map = folium.Map(location=center, zoom_start=13)
        
        # Add remaining edges (blue)
        remaining_edges = results['edges'].drop(results['intersecting_edge_ids'], errors='ignore')
        for idx, row in remaining_edges.iterrows():
            coords = [[point[1], point[0]] for point in row.geometry.coords]
            folium.PolyLine(coords, color='blue', weight=2, opacity=0.7).add_to(impact_map)
        
        # Add removed edges (red)
        removed_edges = results['edges'].loc[results['intersecting_edge_ids']]
        for idx, row in removed_edges.iterrows():
            coords = [[point[1], point[0]] for point in row.geometry.coords]
            folium.PolyLine(coords, color='red', weight=3, opacity=0.8).add_to(impact_map)
        
        # Add parks (green)
        for idx, row in results['parks'].iterrows():
            if row.geometry.geom_type == 'Polygon':
                coords = [[point[1], point[0]] for point in row.geometry.exterior.coords]
                folium.Polygon(coords, color='green', fillColor='green', 
                             fillOpacity=0.3, popup=f"Park {idx}").add_to(impact_map)
        
        st_folium(impact_map, width=700, height=400, key="impact_visualization")
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🔵 **Blue lines**: Remaining street network
        - 🔴 **Red lines**: Streets passing through parks
        - 🟢 **Green areas**: Parks and green spaces
        """)
    
    def render_accessibility_comparison(self, results: Dict):
        st.write("**15-Minute Walking Accessibility Comparison**")
        st.write("Compare how many destinations each location can reach within 15 minutes of walking, before and after removing park connections.")
        
        # Show current analysis settings
        if 'analysis_params' in st.session_state:
            params = st.session_state.analysis_params
            if params.get('use_node_sampling', True):
                sample_size = params.get('sample_size', 200)
                st.info(f"🎯 **Performance Mode**: Analyzing ~{sample_size} sampled nodes for faster results")
            else:
                st.warning(f"🔍 **Complete Analysis Mode**: Analyzing ALL nodes - this will take longer but be more comprehensive")
        
        # Create before/after toggle
        view_mode = st.radio("Select view:", ["Before (with parks)", "After (without parks)", "Difference"], horizontal=True)
        
        # Calculate accessibility if not already done
        if 'accessibility_data' not in st.session_state:
            with st.spinner("Calculating 15-minute accessibility for all nodes..."):
                accessibility_data = self.calculate_accessibility_comparison(results)
                st.session_state.accessibility_data = accessibility_data
        
        accessibility_data = st.session_state.accessibility_data
        
        if not accessibility_data:
            st.error("Could not calculate accessibility data.")
            return
        
        # Create the accessibility map based on selected view
        self.render_accessibility_map(accessibility_data, view_mode, results)
        
        # Show summary statistics
        self.render_accessibility_stats(accessibility_data)
    
    def calculate_accessibility_comparison(self, results: Dict) -> Dict:
        """Calculate 15-minute walking accessibility using OSMnx"""
        try:
            graph = results['graph']
            intersecting_edge_ids = results['intersecting_edge_ids']
            
            # Convert to undirected for accessibility calculations
            graph_undirected = ox.convert.to_undirected(graph)
            
            # Calculate walking distance limit (15 minutes at 4.5 km/h)
            walking_speed_ms = (4.5 * 1000) / 60  # meters per minute
            max_distance = 15 * walking_speed_ms  # ~1125 meters
            
            st.write(f"🎯 Distance threshold: {max_distance:.1f} meters ({max_distance/1000:.2f} km)")
            
            # Get all nodes and exclude nodes within selected parks
            all_nodes = list(graph_undirected.nodes())
            nodes_gdf = results['nodes']
            parks_gdf = results['parks']
            
            # Find nodes that fall within park boundaries
            nodes_within_parks = set()
            if len(parks_gdf) > 0:
                # Convert parks to same CRS as nodes for spatial operations
                parks_projected = parks_gdf.to_crs(nodes_gdf.crs)
                
                for idx, park_row in parks_projected.iterrows():
                    park_geom = park_row.geometry
                    # Check which nodes fall within this park
                    for node in all_nodes:
                        if node in nodes_gdf.index:
                            node_point = nodes_gdf.loc[node].geometry
                            if park_geom.contains(node_point):
                                nodes_within_parks.add(node)
            
            # Exclude park nodes from analysis
            analysis_nodes = [node for node in all_nodes if node not in nodes_within_parks]
            
            # Apply node sampling if specified in parameters
            if 'analysis_params' in st.session_state:
                params = st.session_state.analysis_params
                if params.get('use_node_sampling', True) and params.get('sample_size'):
                    sample_size = params['sample_size']
                    if len(analysis_nodes) > sample_size:
                        import random
                        random.seed(42)  # For reproducible results
                        analysis_nodes = random.sample(analysis_nodes, sample_size)
                        st.info(f"🎯 Sampled {len(analysis_nodes)} nodes from {len([node for node in all_nodes if node not in nodes_within_parks])} total nodes for performance")
                    else:
                        st.info(f"📊 Using all {len(analysis_nodes)} available nodes (less than sample size)")
                else:
                    st.info(f"🔍 Analyzing all {len(analysis_nodes)} nodes (complete analysis - this may take longer)")
            else:
                # Default to sampling for performance
                if len(analysis_nodes) > 200:
                    import random
                    random.seed(42)
                    analysis_nodes = random.sample(analysis_nodes, 200)
                    st.info(f"🎯 Auto-sampled {len(analysis_nodes)} nodes for performance (use parameter settings to control)")
            
            # Choose calculation method based on NetworkKit availability and network size
            network_size = len(analysis_nodes)
            
            if NETWORKIT_AVAILABLE and network_size > 50:  # Lower threshold for NetworkKit
                st.write(f"🚀 Using NetworkKit for high-performance calculation...")
                st.write(f"Analyzing {len(analysis_nodes):,} nodes (excluding {len(nodes_within_parks)} nodes within parks)")
                st.write("⚡ NetworkKit optimization enabled - this will be faster!")
                
                return self.calculate_accessibility_networkit(
                    graph_undirected, analysis_nodes, nodes_within_parks,
                    intersecting_edge_ids, nodes_gdf, max_distance, graph
                )
            else:
                if not NETWORKIT_AVAILABLE:
                    st.write(f"🔄 Using NetworkX for accessibility calculation...")
                    st.info("💡 Install NetworkKit for faster performance: `pip install networkit`")
                else:
                    st.write(f"🔄 Using NetworkX for small network calculation...")
                    st.info(f"Network size ({network_size} nodes) is small - NetworkX is sufficient")
                
                st.write(f"Analyzing {len(analysis_nodes):,} nodes (excluding {len(nodes_within_parks)} nodes within parks)")
                st.write("⏳ This will take some time - calculating shortest paths for all nodes...")
                
                return self.calculate_accessibility_osmnx(
                    graph_undirected, analysis_nodes, nodes_within_parks,
                    intersecting_edge_ids, nodes_gdf, max_distance, graph
                )
            
        except Exception as e:
            st.error(f"Error calculating accessibility: {str(e)}")
            return None
    
    def convert_networkx_to_networkit(self, nx_graph, cache_key=None):
        """Convert NetworkX graph to NetworkKit with proper node mapping and caching"""
        
        # Check cache first
        if cache_key and hasattr(self, '_nk_cache') and cache_key in self._nk_cache:
            return self._nk_cache[cache_key]
        
        # Create node mapping (NetworkKit uses integer IDs starting from 0)
        nx_nodes = list(nx_graph.nodes())
        node_to_nk = {node: i for i, node in enumerate(nx_nodes)}
        nk_to_node = {i: node for node, i in node_to_nk.items()}
        
        # Create NetworkKit graph
        nk_graph = nk.Graph(n=len(nx_nodes), weighted=True, directed=False)
        
        # Add edges with weights (optimized batch processing)
        edges_added = 0
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            weight = data.get('length', 100.0)
            edge_data.append((node_to_nk[u], node_to_nk[v], weight))
            edges_added += 1
        
        # Batch add edges for better performance
        for u_nk, v_nk, weight in edge_data:
            nk_graph.addEdge(u_nk, v_nk, weight)
        
        result = (nk_graph, node_to_nk, nk_to_node)
        
        # Cache result if cache_key provided
        if cache_key:
            if not hasattr(self, '_nk_cache'):
                self._nk_cache = {}
            self._nk_cache[cache_key] = result
        
        return result
    
    def get_reachable_nodes_optimized(self, dijkstra, nk_graph, source_nk, max_distance, nk_to_node, nodes_within_parks):
        """Optimized function to get reachable nodes using NetworkKit distance queries"""
        reachable_nodes = []
        
        # Use NetworkKit's distance vector for faster access
        try:
            # Get distances to all nodes at once (much faster than individual calls)
            distances = [dijkstra.distance(target) for target in range(nk_graph.numberOfNodes())]
            
            for target_nk, distance in enumerate(distances):
                if (target_nk != source_nk and 
                    distance <= max_distance and 
                    distance < float('inf')):
                    
                    target_node = nk_to_node[target_nk]
                    if target_node not in nodes_within_parks:
                        reachable_nodes.append(target_node)
                        
        except Exception:
            # Fallback to individual distance calls if batch fails
            for target_nk in range(nk_graph.numberOfNodes()):
                if target_nk != source_nk:
                    try:
                        distance = dijkstra.distance(target_nk)
                        if distance <= max_distance and distance < float('inf'):
                            target_node = nk_to_node[target_nk]
                            if target_node not in nodes_within_parks:
                                reachable_nodes.append(target_node)
                    except:
                        continue
        
        return reachable_nodes
    
    def calculate_accessibility_networkit(self, graph_undirected, analysis_nodes, nodes_within_parks,
                                         intersecting_edge_ids, nodes_gdf, max_distance, original_graph):
        """High-performance accessibility calculation using NetworkKit with optimizations"""
        
        try:
            # Convert NetworkX to NetworkKit with caching
            st.write("🔄 Converting network to NetworkKit format...")
            conversion_start = time.time()
            
            # Create cache key based on graph structure
            graph_hash = str(hash(tuple(sorted(graph_undirected.edges()))))
            cache_key = f"original_{graph_hash}"
            
            nk_graph, node_to_nk, nk_to_node = self.convert_networkx_to_networkit(
                graph_undirected, cache_key=cache_key
            )
            conversion_time = time.time() - conversion_start
            
            st.write(f"✅ Conversion completed in {conversion_time:.2f}s")
            st.write(f"NetworkKit graph: {nk_graph.numberOfNodes():,} nodes, {nk_graph.numberOfEdges():,} edges")
            
            # BEFORE: Calculate accessibility with original network
            st.write("🚀 Calculating BEFORE accessibility with NetworkKit...")
            before_accessibility = {}
            progress_bar = st.progress(0)
            total_nodes = len(analysis_nodes)
            
            # Batch process nodes for better performance
            batch_size = max(1, min(50, total_nodes // 10))  # Dynamic batch sizing
            
            for i, node in enumerate(analysis_nodes):
                try:
                    nk_node = node_to_nk[node]
                    
                    # Use Dijkstra for single-source shortest path
                    dijkstra = nk.distance.Dijkstra(nk_graph, nk_node)
                    dijkstra.run()
                    
                    # Get reachable nodes using optimized function
                    reachable_nodes = self.get_reachable_nodes_optimized(
                        dijkstra, nk_graph, nk_node, max_distance, nk_to_node, nodes_within_parks
                    )
                    
                    # Calculate area of reachable region
                    if len(reachable_nodes) >= 3:
                        accessible_area = self.calculate_area_from_nodes(reachable_nodes, nodes_gdf)
                        before_accessibility[node] = accessible_area
                    else:
                        before_accessibility[node] = 0
                        
                except Exception as e:
                    before_accessibility[node] = 0
                    if i < 3:  # Show errors for first few nodes only
                        st.warning(f"Error calculating accessibility for node {node}: {str(e)}")
                
                # Update progress less frequently for better performance
                if i % max(1, total_nodes // 50) == 0 or i == total_nodes - 1:
                    progress_bar.progress((i + 1) / total_nodes * 0.5)
            
            # AFTER: Create filtered NetworkKit graph (optimized)
            st.write("🔄 Creating filtered network...")
            
            # Pre-filter edges more efficiently
            edges_to_remove = set()
            for u, v, k in original_graph.edges(keys=True):
                edge_id = (u, v, k)
                if edge_id in intersecting_edge_ids:
                    edges_to_remove.add((u, v))
            
            # Create filtered graph by copying and removing edges
            filtered_graph = graph_undirected.copy()
            filtered_graph.remove_edges_from(edges_to_remove)
            
            st.write(f"Removed {len(edges_to_remove)} park-intersecting edges")
            
            # Convert filtered graph to NetworkKit with caching
            filtered_hash = str(hash(tuple(sorted(filtered_graph.edges()))))
            filtered_cache_key = f"filtered_{filtered_hash}"
            
            filtered_nk_graph, filtered_node_to_nk, filtered_nk_to_node = self.convert_networkx_to_networkit(
                filtered_graph, cache_key=filtered_cache_key
            )
            
            # Calculate AFTER accessibility with NetworkKit
            st.write("🚀 Calculating AFTER accessibility with NetworkKit...")
            after_accessibility = {}
            
            for i, node in enumerate(analysis_nodes):
                try:
                    if node in filtered_node_to_nk:
                        nk_node = filtered_node_to_nk[node]
                        
                        # Use Dijkstra for single-source shortest path
                        dijkstra = nk.distance.Dijkstra(filtered_nk_graph, nk_node)
                        dijkstra.run()
                        
                        # Get reachable nodes using optimized function
                        reachable_nodes = self.get_reachable_nodes_optimized(
                            dijkstra, filtered_nk_graph, nk_node, max_distance, 
                            filtered_nk_to_node, nodes_within_parks
                        )
                        
                        # Calculate area of reachable region
                        if len(reachable_nodes) >= 3:
                            accessible_area = self.calculate_area_from_nodes(reachable_nodes, nodes_gdf)
                            after_accessibility[node] = accessible_area
                        else:
                            after_accessibility[node] = 0
                    else:
                        after_accessibility[node] = 0
                        
                except Exception as e:
                    after_accessibility[node] = 0
                
                # Update progress less frequently
                if i % max(1, total_nodes // 50) == 0 or i == total_nodes - 1:
                    progress_bar.progress(0.5 + (i + 1) / total_nodes * 0.5)
            
            progress_bar.empty()
            st.success("✅ NetworkKit accessibility calculation completed!")
            
            # Calculate differences
            accessibility_diff = {}
            for node in analysis_nodes:
                before_area = before_accessibility.get(node, 0)
                after_area = after_accessibility.get(node, 0)
                accessibility_diff[node] = after_area - before_area
            
            # Cache management - limit cache size
            if hasattr(self, '_nk_cache') and len(self._nk_cache) > 10:
                # Keep only the 5 most recent entries
                cache_keys = list(self._nk_cache.keys())
                for old_key in cache_keys[:-5]:
                    del self._nk_cache[old_key]
            
            return {
                'before': before_accessibility,
                'after': after_accessibility,
                'difference': accessibility_diff,
                'analysis_nodes': analysis_nodes,
                'nodes_within_parks': nodes_within_parks,
                'max_distance': max_distance,
                'edges_removed_count': len(edges_to_remove),
                'method': 'NetworkKit (Optimized)',
                'conversion_time': conversion_time,
                'total_nodes_processed': len(analysis_nodes)
            }
            
        except Exception as e:
            st.error(f"NetworkKit calculation failed: {str(e)}")
            st.warning("Falling back to NetworkX calculation...")
            
            # Clear cache on error to prevent corrupted data
            if hasattr(self, '_nk_cache'):
                self._nk_cache.clear()
            
            # Fallback to NetworkX
            return self.calculate_accessibility_osmnx(
                graph_undirected, analysis_nodes, nodes_within_parks,
                intersecting_edge_ids, nodes_gdf, max_distance, original_graph
            )
    
    def calculate_accessibility_osmnx(self, graph_undirected, analysis_nodes, nodes_within_parks,
                                     intersecting_edge_ids, nodes_gdf, max_distance, original_graph):
        """Clean OSMnx accessibility calculation"""
        
        # BEFORE: Calculate accessibility with original network
        st.write("🔄 Calculating BEFORE accessibility...")
        before_accessibility = {}
        progress_bar = st.progress(0)
        total_nodes = len(analysis_nodes)
        
        # Debug: Check first few nodes
        debug_info = []
        
        for i, node in enumerate(analysis_nodes):
            try:
                # Use ego graph to find nodes within walking distance
                ego_graph = nx.ego_graph(graph_undirected, node, radius=max_distance, distance='length')
                
                # Get reachable nodes excluding parks
                reachable_nodes = [n for n in ego_graph.nodes() if n != node and n not in nodes_within_parks]
                
                # Debug: Check first few nodes
                if i < 3:
                    debug_info.append({
                        'node': node,
                        'ego_graph_nodes': len(ego_graph.nodes()),
                        'reachable_nodes': len(reachable_nodes),
                        'max_distance_threshold': max_distance
                    })
                
                # Calculate area of reachable region
                if len(reachable_nodes) >= 3:
                    accessible_area = self.calculate_area_from_nodes(reachable_nodes, nodes_gdf)
                    before_accessibility[node] = accessible_area
                    
                    # Debug: Report first successful calculation
                    if i < 3:
                        debug_info[-1]['calculated_area'] = accessible_area
                else:
                    before_accessibility[node] = 0
                    if i < 3:
                        debug_info[-1]['calculated_area'] = 0
                    
            except Exception as e:
                before_accessibility[node] = 0
                if i < 3:
                    debug_info.append({'node': node, 'error': str(e)})
            
            # Update progress
            if i % max(1, total_nodes // 100) == 0 or i == total_nodes - 1:
                progress_bar.progress((i + 1) / total_nodes * 0.5)
        
        # Show debug information
        if debug_info:
            st.write("🔍 **Debug Information (first 3 nodes):**")
            for i, info in enumerate(debug_info[:3]):
                st.write(f"**Node {i+1} ({info.get('node', 'unknown')}):**")
                if 'error' in info:
                    st.error(f"Error: {info['error']}")
                else:
                    st.write(f"- Ego graph nodes: {info.get('ego_graph_nodes', 0)}")
                    st.write(f"- Reachable nodes (excluding parks): {info.get('reachable_nodes', 0)}")
                    st.write(f"- Max distance threshold: {info.get('max_distance_threshold', 0):.1f}m")
                    st.write(f"- Calculated area: {info.get('calculated_area', 0):.2f} m²")
        
        # AFTER: Create filtered network
        st.write("🔄 Creating filtered network...")
        filtered_graph = graph_undirected.copy()
        edges_to_remove = []
        
        for u, v, k in original_graph.edges(keys=True):
            edge_id = (u, v, k)
            if edge_id in intersecting_edge_ids:
                if filtered_graph.has_edge(u, v):
                    edges_to_remove.append((u, v))
        
        edges_to_remove = list(set(edges_to_remove))
        filtered_graph.remove_edges_from(edges_to_remove)
        
        st.write(f"Removed {len(edges_to_remove)} park-intersecting edges")
        
        # Calculate AFTER accessibility
        st.write("🔄 Calculating AFTER accessibility...")
        after_accessibility = {}
        
        for i, node in enumerate(analysis_nodes):
            try:
                if node in filtered_graph:
                    # Use ego graph on filtered network
                    ego_graph = nx.ego_graph(filtered_graph, node, radius=max_distance, distance='length')
                    
                    # Get reachable nodes excluding parks
                    reachable_nodes = [n for n in ego_graph.nodes() if n != node and n not in nodes_within_parks]
                    
                    # Calculate area of reachable region
                    if len(reachable_nodes) >= 3:
                        accessible_area = self.calculate_area_from_nodes(reachable_nodes, nodes_gdf)
                        after_accessibility[node] = accessible_area
                    else:
                        after_accessibility[node] = 0
                else:
                    after_accessibility[node] = 0
            except:
                after_accessibility[node] = 0
            
            # Update progress
            if i % max(1, total_nodes // 100) == 0 or i == total_nodes - 1:
                progress_bar.progress(0.5 + (i + 1) / total_nodes * 0.5)
        
        progress_bar.empty()
        st.success("✅ OSMnx accessibility calculation completed!")
        
        # Calculate differences
        accessibility_diff = {}
        for node in analysis_nodes:
            before_area = before_accessibility.get(node, 0)
            after_area = after_accessibility.get(node, 0)
            accessibility_diff[node] = after_area - before_area
        
        return {
            'before': before_accessibility,
            'after': after_accessibility,
            'difference': accessibility_diff,
            'analysis_nodes': analysis_nodes,
            'nodes_within_parks': nodes_within_parks,
            'max_distance': max_distance,
            'edges_removed_count': len(edges_to_remove),
            'method': 'NetworkX'
        }
    
    def calculate_isochrone_area(self, ego_graph, nodes_gdf, max_distance):
        """Calculate the area covered by a 15-minute walking isochrone"""
        try:
            if len(ego_graph.nodes()) < 3:
                return 0
            
            # Get coordinates of all reachable nodes
            reachable_points = []
            for node in ego_graph.nodes():
                if node in nodes_gdf.index:
                    point = nodes_gdf.loc[node].geometry
                    reachable_points.append([point.x, point.y])
            
            if len(reachable_points) < 3:
                return 0
            
            # Create convex hull to estimate accessible area
            from scipy.spatial import ConvexHull
            import numpy as np
            
            points_array = np.array(reachable_points)
            
            # Create convex hull
            try:
                hull = ConvexHull(points_array)
                
                # Calculate area of convex hull (in coordinate units)
                # Convert to approximate square meters using rough conversion
                # (this is approximate since we're working in lat/lon)
                area_deg_sq = hull.volume  # In 2D, volume is area
                
                # Rough conversion: 1 degree ≈ 111 km at equator
                # This is approximate but gives relative comparison
                area_sq_km = area_deg_sq * (111 ** 2)  # Convert to km²
                area_sq_m = area_sq_km * 1e6  # Convert to m²
                
                return area_sq_m
                
            except Exception:
                # Fallback: estimate area using bounding box
                min_x, min_y = points_array.min(axis=0)
                max_x, max_y = points_array.max(axis=0)
                
                # Bounding box area
                width_deg = max_x - min_x
                height_deg = max_y - min_y
                
                area_sq_km = width_deg * height_deg * (111 ** 2)
                area_sq_m = area_sq_km * 1e6
                
                return area_sq_m
                
        except Exception:
            return 0
    
    def calculate_area_from_nodes(self, reachable_nodes, nodes_gdf):
        """Calculate area from a list of reachable nodes"""
        try:
            if len(reachable_nodes) < 3:
                return 0
            
            # Get coordinates of all reachable nodes
            reachable_points = []
            for node in reachable_nodes:
                if node in nodes_gdf.index:
                    point = nodes_gdf.loc[node].geometry
                    reachable_points.append([point.x, point.y])
            
            if len(reachable_points) < 3:
                return 0
            
            # Create convex hull to estimate accessible area
            from scipy.spatial import ConvexHull
            import numpy as np
            
            points_array = np.array(reachable_points)
            
            try:
                hull = ConvexHull(points_array)
                area_deg_sq = hull.volume  # In 2D, volume is area
                area_sq_km = area_deg_sq * (111 ** 2)  # Convert to km²
                area_sq_m = area_sq_km * 1e6  # Convert to m²
                return area_sq_m
                
            except Exception:
                # Fallback: bounding box
                min_x, min_y = points_array.min(axis=0)
                max_x, max_y = points_array.max(axis=0)
                width_deg = max_x - min_x
                height_deg = max_y - min_y
                area_sq_km = width_deg * height_deg * (111 ** 2)
                return area_sq_km * 1e6
                
        except Exception:
            return 0
    
    def calculate_isochrone_boundary(self, node, graph, max_distance, nodes_gdf):
        """Calculate isochrone boundary for a given node"""
        try:
            # Get reachable nodes within distance
            ego_graph = nx.ego_graph(graph, node, radius=max_distance, distance='length')
            reachable_nodes = [n for n in ego_graph.nodes() if n != node]
            
            if len(reachable_nodes) < 3:
                return None
            
            # Get coordinates of reachable nodes
            points = []
            for n in reachable_nodes:
                if n in nodes_gdf.index:
                    point = nodes_gdf.loc[n].geometry
                    points.append([point.y, point.x])  # lat, lon for mapping
            
            if len(points) < 3:
                return None
            
            # Create convex hull boundary
            from scipy.spatial import ConvexHull
            import numpy as np
            
            points_array = np.array(points)
            try:
                hull = ConvexHull(points_array)
                boundary_points = []
                for vertex in hull.vertices:
                    boundary_points.append(points_array[vertex])
                
                # Close the polygon by adding first point at the end
                boundary_points.append(boundary_points[0])
                return boundary_points
                
            except Exception:
                return None
                
        except Exception:
            return None
    
    def get_isochrone_data_for_nodes(self, accessibility_data: Dict, results: Dict, sample_nodes: list = None):
        """Pre-calculate isochrone data for nodes to enable hover visualization"""
        
        # Use a sample of nodes to avoid performance issues
        analysis_nodes = accessibility_data['analysis_nodes']
        
        # If sample_nodes not provided, use intelligent sampling based on user preferences
        if sample_nodes is None:
            # Check if user has set sampling preferences
            if 'analysis_params' in st.session_state:
                params = st.session_state.analysis_params
                if params.get('use_node_sampling', True):
                    # Use smaller sample for isochrones (max 15 nodes for performance)
                    max_isochrone_nodes = min(15, len(analysis_nodes))
                else:
                    # User wants full analysis, but still limit isochrones for UI performance
                    max_isochrone_nodes = min(25, len(analysis_nodes))
            else:
                # Default conservative sampling
                max_isochrone_nodes = min(10, len(analysis_nodes))
            
            step = max(1, len(analysis_nodes) // max_isochrone_nodes)
            sample_nodes = analysis_nodes[::step][:max_isochrone_nodes]
        
        isochrone_data = {}
        
        # Get the graphs we need
        graph = results['graph']
        graph_undirected = ox.convert.to_undirected(graph)
        nodes_gdf = results['nodes']
        max_distance = accessibility_data['max_distance']
        
        # For filtered graph (after removing park connections)
        intersecting_edge_ids = results['intersecting_edge_ids']
        filtered_graph = graph_undirected.copy()
        edges_to_remove = []
        
        for u, v, k in graph.edges(keys=True):
            edge_id = (u, v, k)
            if edge_id in intersecting_edge_ids:
                if filtered_graph.has_edge(u, v):
                    edges_to_remove.append((u, v))
        
        edges_to_remove = list(set(edges_to_remove))
        filtered_graph.remove_edges_from(edges_to_remove)
        
        st.write(f"🗺️ Pre-calculating isochrones for {len(sample_nodes)} sample nodes...")
        progress_bar = st.progress(0)
        
        for i, node in enumerate(sample_nodes):
            try:
                # Calculate before isochrone
                before_boundary = self.calculate_isochrone_boundary(
                    node, graph_undirected, max_distance, nodes_gdf
                )
                
                # Calculate after isochrone
                after_boundary = None
                if node in filtered_graph:
                    after_boundary = self.calculate_isochrone_boundary(
                        node, filtered_graph, max_distance, nodes_gdf
                    )
                
                isochrone_data[node] = {
                    'before': before_boundary,
                    'after': after_boundary,
                    'lat': nodes_gdf.loc[node].geometry.y,
                    'lon': nodes_gdf.loc[node].geometry.x
                }
                
            except Exception as e:
                isochrone_data[node] = {
                    'before': None,
                    'after': None,
                    'lat': nodes_gdf.loc[node].geometry.y if node in nodes_gdf.index else 0,
                    'lon': nodes_gdf.loc[node].geometry.x if node in nodes_gdf.index else 0
                }
            
            progress_bar.progress((i + 1) / len(sample_nodes))
        
        progress_bar.empty()
        st.success(f"✅ Isochrone data prepared for {len(sample_nodes)} nodes")
        
        return isochrone_data
    
    def render_accessibility_map(self, accessibility_data: Dict, view_mode: str, results: Dict):
        """Render accessibility map with color-coded nodes and isochrone visualization"""
        
        # Get data based on view mode
        if view_mode == "Before (with parks)":
            data = accessibility_data['before']
            title = "Accessibility Before (with park connections)"
            colorscale = 'RdYlGn'  # Red to Green
        elif view_mode == "After (without parks)":
            data = accessibility_data['after'] 
            title = "Accessibility After (without park connections)"
            colorscale = 'RdYlGn'  # Red to Green
        else:  # Difference
            data = accessibility_data['difference']
            title = "Accessibility Difference (After - Before)"
            colorscale = 'RdBu_r'  # Red for loss, Blue for gain (reversed)
        
        # Get node positions
        nodes = results['nodes']
        analysis_nodes = accessibility_data['analysis_nodes']
        
        # Pre-calculate isochrone data for interactive visualization
        st.info("🔄 Preparing interactive isochrone visualization...")
        isochrone_data = self.get_isochrone_data_for_nodes(accessibility_data, results)
        
        # Create lists for plotting
        node_ids = []
        lats = []
        lons = []
        values = []
        hover_texts = []
        custom_data = []  # For storing node IDs for interactivity
        
        for node in analysis_nodes:
            if node in nodes.index and node in data:
                node_ids.append(node)
                lats.append(nodes.loc[node].geometry.y)
                lons.append(nodes.loc[node].geometry.x)
                values.append(data[node])
                custom_data.append(node)
                
                if view_mode == "Difference":
                    area_change_km2 = data[node] / 1e6  # Convert to km²
                    before_area = accessibility_data['before'].get(node, 0) / 1e6
                    after_area = accessibility_data['after'].get(node, 0) / 1e6
                    hover_text = (f"Node {node}<br>"
                                f"Before: {before_area:.3f} km²<br>"
                                f"After: {after_area:.3f} km²<br>"
                                f"Change: {area_change_km2:+.3f} km²<br>"
                                f"{'🔍 Click to show isochrones' if node in isochrone_data else ''}")
                    hover_texts.append(hover_text)
                else:
                    area_km2 = data[node] / 1e6  # Convert to km²
                    hover_text = (f"Node {node}<br>"
                                f"Accessible Area: {area_km2:.3f} km²<br>"
                                f"{'🔍 Click to show isochrone' if node in isochrone_data else ''}")
                    hover_texts.append(hover_text)
        
        if not values:
            st.error("No accessibility data to display.")
            return
        
        # Create Streamlit columns for map and controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Map Controls")
            
            show_isochrones = st.checkbox("Show Isochrones", value=False, 
                                        help="Toggle isochrone visualization overlay")
            
            if view_mode == "Difference":
                st.markdown("**Isochrone Legend:**")
                st.markdown("🔴 Red: Before (with parks)")
                st.markdown("🔵 Blue: After (without parks)")
                st.markdown("💜 Purple: Overlap area")
            else:
                isochrone_color = st.color_picker("Isochrone Color", "#00FF00" if "Before" in view_mode else "#FF0000")
                st.markdown("**Isochrone Legend:**")
                if "Before" in view_mode:
                    st.markdown("🟢 Green: 15-min walking area (with parks)")
                else:
                    st.markdown("🔴 Red: 15-min walking area (without parks)")
            
            selected_node_for_isochrone = st.selectbox(
                "Show Isochrone for Node:",
                ["None"] + [str(node) for node in sorted(isochrone_data.keys())],
                help="Select a specific node to display its isochrone"
            )
        
        with col1:
            # Create the map using Plotly
            fig = go.Figure()
            
            # Add accessibility points
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=6,  # Slightly larger for better visibility
                    color=values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        title="Area Accessible<br>(km²)" if view_mode != "Difference" else "Area Change<br>(km²)",
                        x=1.02,
                        thickness=15
                    ),
                    sizemode='diameter'
                    # Removed 'line' property as it's not valid for scattermapbox markers
                ),
                text=hover_texts,
                customdata=custom_data,
                hovertemplate='%{text}<extra></extra>',
                name="Accessibility Nodes"
            ))
            
            # Add isochrone for selected node
            if selected_node_for_isochrone != "None" and show_isochrones:
                selected_node = int(selected_node_for_isochrone)
                if selected_node in isochrone_data:
                    iso_data = isochrone_data[selected_node]
                    
                    if view_mode == "Difference":
                        # Show both before and after isochrones
                        if iso_data['before']:
                            before_lats = [point[0] for point in iso_data['before']]
                            before_lons = [point[1] for point in iso_data['before']]
                            fig.add_trace(go.Scattermapbox(
                                lat=before_lats,
                                lon=before_lons,
                                mode='lines',
                                line=dict(color='red', width=3),
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                name="Before (with parks)",
                                hovertemplate='15-min walk area BEFORE<extra></extra>'
                            ))
                        
                        if iso_data['after']:
                            after_lats = [point[0] for point in iso_data['after']]
                            after_lons = [point[1] for point in iso_data['after']]
                            fig.add_trace(go.Scattermapbox(
                                lat=after_lats,
                                lon=after_lons,
                                mode='lines',
                                line=dict(color='blue', width=3),
                                fill='toself',
                                fillcolor='rgba(0,0,255,0.2)',
                                name="After (without parks)",
                                hovertemplate='15-min walk area AFTER<extra></extra>'
                            ))
                    
                    else:
                        # Show appropriate isochrone for Before/After view
                        isochrone_key = 'before' if "Before" in view_mode else 'after'
                        if iso_data[isochrone_key]:
                            iso_lats = [point[0] for point in iso_data[isochrone_key]]
                            iso_lons = [point[1] for point in iso_data[isochrone_key]]
                            color = isochrone_color if 'isochrone_color' in locals() else '#00FF00'
                            fig.add_trace(go.Scattermapbox(
                                lat=iso_lats,
                                lon=iso_lons,
                                mode='lines',
                                line=dict(color=color, width=3),
                                fill='toself',
                                fillcolor=f'rgba({",".join([str(int(color[i:i+2], 16)) for i in (1, 3, 5)])},0.2)',
                                name="15-min Walking Area",
                                hovertemplate=f'15-min walk area {view_mode.lower()}<extra></extra>'
                            ))
                    
                    # Add marker for selected node
                    fig.add_trace(go.Scattermapbox(
                        lat=[iso_data['lat']],
                        lon=[iso_data['lon']],
                        mode='markers',
                        marker=dict(size=12, color='yellow', symbol='star'),
                        name=f"Selected Node {selected_node}",
                        hovertemplate=f'Selected Node {selected_node}<extra></extra>'
                    ))
            
            # Add selected parks
            parks = results['parks']
            for idx, row in parks.iterrows():
                if row.geometry.geom_type == 'Polygon':
                    # Get park boundary
                    coords = list(row.geometry.exterior.coords)
                    park_lats = [coord[1] for coord in coords]
                    park_lons = [coord[0] for coord in coords]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=park_lats,
                        lon=park_lons,
                        mode='lines',
                        line=dict(color='green', width=3),
                        fill='toself',
                        fillcolor='rgba(0,128,0,0.3)',
                        name=f"Selected Park",
                        showlegend=False,
                        hovertemplate='Selected Park<extra></extra>'
                    ))
            
            # Update layout for full zoomability
            center_lat = sum(lats) / len(lats) if lats else 0
            center_lon = sum(lons) / len(lons) if lons else 0
            
            fig.update_layout(
                mapbox=dict(
                    style="carto-darkmatter",  # Black background map
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=13
                ),
                title=title,
                height=600,  # Taller for better visibility
                margin=dict(t=50, l=0, r=0, b=0),
                showlegend=True,
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add instructions and summary for current view
        st.markdown("---")
        st.info("🔍 **Interactive Features**: Use the controls on the right to toggle isochrone visualization. Select a node to see its walking area boundary.")
        
        if values:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes Displayed", f"{len(values):,}")
            with col2:
                if view_mode == "Difference":
                    avg_change_km2 = np.mean(values) / 1e6
                    st.metric("Average Change", f"{avg_change_km2:+.3f} km²")
                else:
                    avg_area_km2 = np.mean(values) / 1e6
                    st.metric("Average Accessibility", f"{avg_area_km2:.3f} km²")
            with col3:
                nodes_with_isochrones = len(isochrone_data)
                st.metric("Nodes with Isochrones", f"{nodes_with_isochrones}")
                
        # Color scale explanation
        if view_mode == "Difference":
            st.markdown("""
            **Color Scale**: 🔴 Red = Lost accessibility area, 🔵 Blue = Gained accessibility area  
            **Isochrone View**: Toggle to see before (red) and after (blue) walking areas for comparison.
            """)
        else:
            st.markdown("""
            **Color Scale**: 🔴 Red = Small accessible area, 🟡 Yellow = Medium, 🟢 Green = Large accessible area  
            **Isochrone View**: Toggle to see the 15-minute walking boundary from selected nodes.
            """)
            
        # Store isochrone data in session for potential reuse
        if 'isochrone_cache' not in st.session_state:
            st.session_state.isochrone_cache = {}
        st.session_state.isochrone_cache[view_mode] = isochrone_data
    
    def render_accessibility_stats(self, accessibility_data: Dict):
        """Render accessibility statistics"""
        
        before_data = list(accessibility_data['before'].values())
        after_data = list(accessibility_data['after'].values())
        diff_data = list(accessibility_data['difference'].values())
        
        st.markdown("### 📊 Accessibility Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Before (with parks)**")
            st.metric("Average Area", f"{np.mean(before_data)/1e6:.3f} km²")
            st.metric("Max Area", f"{max(before_data)/1e6:.3f} km²")
            st.metric("Min Area", f"{min(before_data)/1e6:.3f} km²")
        
        with col2:
            st.markdown("**After (without parks)**")
            st.metric("Average Area", f"{np.mean(after_data)/1e6:.3f} km²")
            st.metric("Max Area", f"{max(after_data)/1e6:.3f} km²")
            st.metric("Min Area", f"{min(after_data)/1e6:.3f} km²")
        
        with col3:
            st.markdown("**Impact Summary**")
            avg_change = np.mean(diff_data)
            nodes_worse = sum(1 for x in diff_data if x < 0)
            nodes_better = sum(1 for x in diff_data if x > 0)
            
            st.metric("Average Change", f"{avg_change/1e6:+.3f} km²")
            st.metric("Nodes Losing Area", f"{nodes_worse}")
            st.metric("Nodes Gaining Area", f"{nodes_better}")
        
        # Show distribution chart
        st.markdown("### 📈 Accessibility Distribution")
        
        fig = go.Figure()
        
        # Convert to km² for display
        before_data_km2 = [x/1e6 for x in before_data]
        after_data_km2 = [x/1e6 for x in after_data]
        
        fig.add_trace(go.Histogram(
            x=before_data_km2,
            name="Before (with parks)",
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=after_data_km2,
            name="After (without parks)",
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            title="Distribution of 15-Minute Accessibility",
            xaxis_title="Accessible Area (km²)",
            yaxis_title="Number of Nodes",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        total_nodes = len(accessibility_data['analysis_nodes'])
        nodes_in_parks = len(accessibility_data['nodes_within_parks'])
        severely_impacted = sum(1 for x in diff_data if x/1e6 < -0.1)  # >0.1 km² lost
        calculation_method = accessibility_data.get('method', 'Unknown')
        
        st.write(f"• **{total_nodes:,}** nodes analyzed (all nodes excluding those in parks)")
        st.write(f"• **{nodes_in_parks}** nodes within selected parks (excluded from analysis)")
        st.write(f"• **{accessibility_data['edges_removed_count']}** park connection edges removed")
        st.write(f"• **{severely_impacted}** nodes severely impacted (>0.1 km² lost)")
        st.write(f"• **{abs(avg_change)/1e6:.3f} km²** average change in accessible area")
        
        # Show calculation method and performance
        if 'NetworkKit' in calculation_method:
            st.success(f"⚡ **Calculation Method**: {calculation_method}")
            # Show performance metrics if available
            if 'conversion_time' in accessibility_data:
                conversion_time = accessibility_data['conversion_time']
                total_nodes_processed = accessibility_data.get('total_nodes_processed', total_nodes)
                st.info(f"🚀 **Performance**: Graph conversion took {conversion_time:.2f}s for {total_nodes_processed:,} nodes")
                if conversion_time < 5.0:
                    st.success("✅ **Excellent performance** - NetworkKit optimization working efficiently!")
                elif conversion_time < 15.0:
                    st.info("💡 **Good performance** - NetworkKit providing solid acceleration")
                else:
                    st.warning("⚠️ **Consider optimization** - Large network may benefit from caching")
        else:
            st.info(f"🔄 **Calculation Method**: {calculation_method}")
            if total_nodes > 100:
                st.info("💡 Consider using NetworkKit for faster performance on larger networks")
        
        if avg_change < 0:
            st.warning(f"⚠️ Removing park connections reduces average accessibility by {abs(avg_change)/1e6:.3f} km² per location")
        else:
            st.info(f"ℹ️ Park removal has minimal impact on overall accessibility")
    
    def render_network_analysis(self, results: Dict):
        st.write("**Network Connectivity Analysis**")
        
        # Create sample connectivity chart
        fig = go.Figure()
        
        # Sample data for demonstration
        scenarios = ['Original Network', 'Without Park Streets']
        connectivity_scores = [100, 92]  # Placeholder values
        
        fig.add_bar(x=scenarios, y=connectivity_scores, 
                   marker_color=['#2E86AB', '#A23B72'])
        
        fig.update_layout(
            title="Network Connectivity Comparison",
            yaxis_title="Connectivity Score",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Accessibility Impact**")
            st.write(f"• Streets affected: {results['intersecting_edges']:,}")
            st.write(f"• Network impact: {results['impact_percentage']:.1f}%")
            st.write("• Critical connections identified")
        
        with col2:
            st.markdown("**Key Insights**")
            st.write("• Parks serve as vital network connectors")
            st.write("• Some park paths are critical for accessibility")
            st.write("• Alternative routes may require longer detours")
    
    def render_detailed_results(self, results: Dict):
        st.write("**Comprehensive Analysis Results**")
        
        # Summary statistics
        st.markdown("### 📋 Summary Statistics")
        
        summary_data = {
            'Metric': ['Total Street Network Segments', 'Segments Through Parks', 
                      'Impact Percentage', 'Parks Analyzed', 'Average Segment Length'],
            'Value': [f"{results['total_edges']:,}", f"{results['intersecting_edges']:,}",
                     f"{results['impact_percentage']:.1f}%", len(results['parks']),
                     f"{results['edges']['length'].mean():.1f}m"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Edge analysis
        st.markdown("### 🛣️ Street Segment Analysis")
        
        # Sample edge data
        edge_sample = results['edges'].head(10)[['length', 'highway']].reset_index()
        st.dataframe(edge_sample, use_container_width=True)
    
    def render_report_generation(self):
        st.subheader("📄 Step 5: Share Your Evidence")
        
        st.write("Generate publication-ready reports and share your findings.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Report Options**")
            report_type = st.selectbox("Report Type", 
                ["Executive Summary", "Technical Report", "Policy Brief"])
            include_maps = st.checkbox("Include Maps", value=True)
            include_data = st.checkbox("Include Raw Data", value=False)
        
        with col2:
            st.markdown("**Export Format**")
            export_format = st.selectbox("Format", ["PDF", "HTML", "PowerPoint"])
            
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating your report..."):
                time.sleep(2)  # Simulate report generation
                st.success("Report generated successfully! 🎉")
                
                # Provide download options
                st.markdown("**Download Options:**")
                st.download_button(
                    label="📄 Download Report",
                    data="Sample report content",  # In real implementation, generate actual report
                    file_name=f"oasis_park_impact_report.{export_format.lower()}",
                    mime="application/pdf" if export_format == "PDF" else "text/html"
                )
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_progress_bar()
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### 🎛️ Navigation")
            if st.button("🔄 Reset Analysis"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
            
            st.markdown("### ℹ️ About OASIS")
            st.markdown("""
            OASIS analyzes how parks and green spaces impact urban accessibility and connectivity.
            
            **Features:**
            - Interactive park selection
            - Real-time network analysis  
            - Impact visualization
            - Automated reporting
            """)
        
        # Main content based on current step
        if st.session_state.current_step == 1:
            self.render_map_selection()
        elif st.session_state.current_step == 2:
            self.render_parameter_selection()
        elif st.session_state.current_step == 3:
            self.render_analysis_progress()
        elif st.session_state.current_step == 4:
            self.render_results()
        elif st.session_state.current_step == 5:
            self.render_report_generation()

if __name__ == "__main__":
    app = OasisApp()
    app.run()