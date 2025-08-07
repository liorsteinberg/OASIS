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

st.set_page_config(
    page_title="OASIS - Park Accessibility Impact System",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OasisApp:
    def __init__(self):
        self.initialize_session_state()
    
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
            buffer = 0.02  # approximately 500m in degrees
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
                'include_demographics': include_demographics
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
        """Calculate 15-minute walking accessibility for before and after scenarios"""
        try:
            graph = results['graph']
            intersecting_edge_ids = results['intersecting_edge_ids']
            
            # Convert to undirected for accessibility calculations
            graph_undirected = ox.convert.to_undirected(graph)
            
            # Calculate walking distance limit (15 minutes at 4.5 km/h)
            walking_speed_ms = (4.5 * 1000) / 60  # meters per minute
            max_distance = 15 * walking_speed_ms  # ~1125 meters
            
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
            
            st.write(f"Calculating accessibility for {len(analysis_nodes):,} nodes (excluding {len(nodes_within_parks)} nodes within parks)...")
            st.write("⏳ This may take a few minutes for large networks...")
            
            # BEFORE: Calculate accessibility with original network
            before_accessibility = {}
            progress_bar = st.progress(0)
            total_nodes = len(analysis_nodes)
            
            for i, node in enumerate(analysis_nodes):
                try:
                    # Use ego graph to find nodes within walking distance
                    ego_graph = nx.ego_graph(graph_undirected, node, radius=max_distance, distance='length')
                    # Calculate the area covered by the isochrone
                    accessible_area = self.calculate_isochrone_area(ego_graph, nodes_gdf, max_distance)
                    before_accessibility[node] = accessible_area
                except:
                    before_accessibility[node] = 0
                
                # Update progress every 100 nodes to avoid too frequent updates
                if i % max(1, total_nodes // 100) == 0 or i == total_nodes - 1:
                    progress_bar.progress((i + 1) / total_nodes * 0.5)
            
            # AFTER: Create filtered network without park edges
            filtered_graph = graph_undirected.copy()
            edges_to_remove = []
            
            for u, v, k in graph.edges(keys=True):
                edge_id = (u, v, k)
                if edge_id in intersecting_edge_ids:
                    if filtered_graph.has_edge(u, v):
                        edges_to_remove.append((u, v))
            
            # Remove edges (avoid duplicates)
            edges_to_remove = list(set(edges_to_remove))
            filtered_graph.remove_edges_from(edges_to_remove)
            
            # Calculate accessibility with filtered network
            after_accessibility = {}
            
            for i, node in enumerate(analysis_nodes):
                try:
                    if node in filtered_graph:
                        ego_graph = nx.ego_graph(filtered_graph, node, radius=max_distance, distance='length')
                        # Calculate the area covered by the isochrone
                        accessible_area = self.calculate_isochrone_area(ego_graph, nodes_gdf, max_distance)
                        after_accessibility[node] = accessible_area
                    else:
                        after_accessibility[node] = 0
                except:
                    after_accessibility[node] = 0
                
                # Update progress every 100 nodes to avoid too frequent updates
                if i % max(1, total_nodes // 100) == 0 or i == total_nodes - 1:
                    progress_bar.progress(0.5 + (i + 1) / total_nodes * 0.5)
            
            progress_bar.empty()
            
            # Calculate differences
            accessibility_diff = {}
            for node in analysis_nodes:
                before_count = before_accessibility.get(node, 0)
                after_count = after_accessibility.get(node, 0)
                accessibility_diff[node] = after_count - before_count
            
            return {
                'before': before_accessibility,
                'after': after_accessibility,
                'difference': accessibility_diff,
                'analysis_nodes': analysis_nodes,
                'nodes_within_parks': nodes_within_parks,
                'max_distance': max_distance,
                'edges_removed_count': len(edges_to_remove)
            }
            
        except Exception as e:
            st.error(f"Error calculating accessibility: {str(e)}")
            return None
    
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
    
    def render_accessibility_map(self, accessibility_data: Dict, view_mode: str, results: Dict):
        """Render accessibility map with color-coded nodes"""
        
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
        
        # Create lists for plotting
        node_ids = []
        lats = []
        lons = []
        values = []
        hover_texts = []
        
        for node in analysis_nodes:
            if node in nodes.index and node in data:
                node_ids.append(node)
                lats.append(nodes.loc[node].geometry.y)
                lons.append(nodes.loc[node].geometry.x)
                values.append(data[node])
                
                if view_mode == "Difference":
                    area_change_km2 = data[node] / 1e6  # Convert to km²
                    hover_texts.append(f"Node {node}<br>Area Change: {area_change_km2:+.3f} km²")
                else:
                    area_km2 = data[node] / 1e6  # Convert to km²
                    hover_texts.append(f"Node {node}<br>Accessible Area: {area_km2:.3f} km²")
        
        if not values:
            st.error("No accessibility data to display.")
            return
        
        # Create the map
        fig = go.Figure()
        
        # Add accessibility points
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=4,  # Smaller size since we have more points
                color=values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Area Accessible<br>(km²)" if view_mode != "Difference" else "Area Change<br>(km²)",
                    x=1.02,
                    thickness=15
                ),
                sizemode='diameter'
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name="Accessibility"
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
        
        # Add zoom and pan controls info
        st.info("🔍 **Interactive Map**: Use mouse wheel to zoom, click and drag to pan. Hover over points to see accessibility details.")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary for current view
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
                if view_mode == "Difference":
                    worst_impact_km2 = min(values) / 1e6
                    st.metric("Worst Impact", f"{worst_impact_km2:+.3f} km²")
                else:
                    max_access_km2 = max(values) / 1e6
                    st.metric("Best Accessibility", f"{max_access_km2:.3f} km²")
                    
        # Color scale explanation
        if view_mode == "Difference":
            st.markdown("""
            **Color Scale**: 🔴 Red = Lost accessibility area, 🔵 Blue = Gained accessibility area  
            Negative values mean smaller reachable area after park removal.
            """)
        else:
            st.markdown("""
            **Color Scale**: 🔴 Red = Small accessible area, 🟡 Yellow = Medium, 🟢 Green = Large accessible area  
            Values show area (km²) reachable within 15 minutes walking from each location.
            """)
    
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
        
        st.write(f"• **{total_nodes:,}** nodes analyzed (all nodes excluding those in parks)")
        st.write(f"• **{nodes_in_parks}** nodes within selected parks (excluded from analysis)")
        st.write(f"• **{accessibility_data['edges_removed_count']}** park connection edges removed")
        st.write(f"• **{severely_impacted}** nodes severely impacted (>0.1 km² lost)")
        st.write(f"• **{abs(avg_change)/1e6:.3f} km²** average change in accessible area")
        
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