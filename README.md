# Fast Park Accessibility App

## Overview
This FastAPI-based application analyzes park accessibility impacts using interactive maps and comprehensive street network analysis.

## Key Features
- **Real-time map interactions** without page reloads
- **Street network analysis** with accurate reachable infrastructure metrics
- **Dual visualization methods**: Convex Hull (fast) vs Buffered Streets (realistic)
- **Mass analysis capabilities** with comprehensive statistics and charts
- **5-level impact classification** with detailed tooltips
- **Interactive visualizations** using Chart.js
- **City quick-select** (Rotterdam, Munich) with smooth map transitions
- **CSV export** for detailed results

## Quick Start
```bash
# Install dependencies
pip install -r requirements_fast.txt

# Run the application
python run_fast_app.py
```

Then open http://localhost:8000

## Usage Workflow
1. **Choose Location**: Select Rotterdam/Munich or click anywhere on map
2. **Load Data**: Click "Load Parks & Network" button
3. **Select Park**: Click on green park polygon or use dropdown
4. **Select Node**: Click on blue node marker or use dropdown  
5. **Configure**: Set walk time, speed, and visualization method
6. **Analyze**: Run single analysis or mass analysis with interactive charts
7. **Export**: Download CSV results with detailed metrics

## Impact Assessment
The app provides detailed tooltips explaining each impact level:
- 🌟 **Highly Positive**: >5% improvement in street network accessibility
- ✅ **Positive**: 1-5% improvement
- ➖ **Neutral**: -1% to +1% change  
- 🔶 **Negative**: 1-5% reduction
- ⚠️ **Highly Negative**: >5% reduction in accessibility

## Technical Architecture
- **Backend**: FastAPI with async processing
- **Frontend**: Leaflet.js maps with Chart.js visualizations
- **Data Processing**: OSMnx + NetworkX for street network analysis
- **Geospatial**: GeoPandas + Shapely for geometric operations
- **Performance**: Multi-threading with in-memory caching

## Files
- `fast_park_app.py` - Main FastAPI application
- `run_fast_app.py` - Simple startup script
- `requirements_fast.txt` - Python dependencies
- `cache/` - Performance caching directory