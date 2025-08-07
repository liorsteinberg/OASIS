# 🌳 OASIS - Open Accessibility Spatial Impact System

> **Discover the hidden impact of parks on urban connectivity**

OASIS is a web-based platform that analyzes how parks and green spaces affect urban accessibility and network connectivity. Using advanced spatial analysis and network theory, it reveals the critical role that parks play in connecting urban areas.

## ✨ Features

### Phase 1: Interactive Frontend
- 🗺️ **Interactive Map Dashboard** - Click-and-explore park selection with real-time visualization
- 🎯 **Step-by-Step Wizard** - Guided user journey from park selection to impact analysis
- 📊 **Dynamic Visualizations** - Animated network changes and accessibility metrics
- 📱 **Mobile Responsive** - Works seamlessly across devices
- 📄 **Automated Reports** - Generate publication-ready analysis reports

### Phase 2: Advanced Analytics (Coming Soon)
- ⚡ **Microservices Architecture** - Scalable backend processing
- 🚀 **Performance Optimization** - Precomputed network tiles and intelligent caching
- 🔄 **Real-time Processing** - Stream results as they compute
- 🌍 **Global Coverage** - Support for cities worldwide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/OASIS.git
   cd OASIS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎮 How to Use

### Step 1: Choose Your Park 🎯
- Click anywhere on the interactive map to select a location
- OASIS will automatically find parks and green spaces in the area
- Preview the parks that will be analyzed

### Step 2: Set Parameters ⚙️
- Configure walking speed and maximum walking time
- Choose analysis depth (Quick/Standard/Deep)
- Enable demographic analysis if desired

### Step 3: Watch the Magic ✨
- Real-time progress tracking as analysis runs
- See network processing and impact calculations
- Animated visualization of results

### Step 4: Explore Results 📊
- Interactive impact map showing removed vs remaining streets
- Network connectivity analysis and metrics
- Detailed statistics and insights

### Step 5: Share Evidence 📄
- Generate professional reports in multiple formats
- Export maps and data for further analysis
- Share findings with stakeholders

## 🏗️ Architecture

### Current Implementation
```
OASIS/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── utils/
│   ├── network_analyzer.py  # Advanced network analysis
│   └── __init__.py
├── requirements.txt       # Python dependencies
└── cache/                # Cached analysis results
```

### Planned Architecture (Phase 2)
```
OASIS/
├── frontend/             # Streamlit/Dash web interface
├── backend/
│   ├── api/             # FastAPI microservices
│   ├── processors/      # Network analysis engines
│   ├── cache/          # Redis caching layer
│   └── database/       # Results storage
├── services/
│   ├── osm-fetcher/    # OSM data acquisition
│   ├── network-processor/ # Graph analysis
│   └── report-generator/  # Automated reporting
└── infrastructure/      # Docker, K8s configs
```

## 🔧 Configuration

Key settings can be modified in `config.py`:

```python
# Default location (lat, lon)
DEFAULT_LOCATION = (51.9225, 4.47917)  # Rotterdam

# Analysis parameters
SEARCH_RADIUS = 2000  # meters
WALKING_SPEEDS = {'slow': 3.0, 'normal': 4.5, 'fast': 6.0}  # km/h
```

## 📊 Analysis Methods

### Network Analysis
- **Street Network Extraction** - Uses OSMnx to download OpenStreetMap data
- **Park Intersection Detection** - Spatial analysis to find streets passing through parks
- **Centrality Measures** - Betweenness, closeness, degree, and PageRank centrality
- **Connectivity Impact** - Measures how park removal affects network connectivity

### Optimization
- **NetworkIt Integration** - 50-100x faster centrality calculations for large networks
- **Intelligent Sampling** - Statistical sampling for quick previews
- **Progressive Loading** - Show results as they compute
- **Caching Strategy** - Store results to avoid recomputation

## 🌟 Use Cases

### Urban Planning
- Assess impact of proposed park developments
- Understand connectivity value of existing green spaces
- Identify critical park connections for preservation

### Public Health
- Analyze walking accessibility to green spaces
- Evaluate equity in park access across neighborhoods
- Support active mobility planning

### Policy Research
- Generate evidence for park funding decisions
- Quantify benefits of green infrastructure
- Support environmental justice initiatives

### Academic Research
- Network analysis of urban systems
- Green space accessibility studies
- Transportation and land use interaction research

## 🛠️ Technical Details

### Dependencies
- **OSMnx** - Street network analysis
- **NetworkX** - Graph algorithms and centrality measures  
- **NetworkIt** - High-performance network analysis (optional)
- **GeoPandas** - Spatial data processing
- **Streamlit** - Web interface framework
- **Folium** - Interactive mapping
- **Plotly** - Advanced visualizations

### Performance Notes
- Networks with <10K edges: ~30 seconds analysis time
- Networks with 10K-50K edges: ~2-5 minutes analysis time  
- Networks with >50K edges: Uses sampling and NetworkIt optimization
- Caching reduces repeat analyses to <5 seconds

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/OASIS.git
cd OASIS

# Create virtual environment
python -m venv oasis-env
source oasis-env/bin/activate  # On Windows: oasis-env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev tools

# Run tests
pytest

# Run with debug mode
streamlit run app.py --server.runOnSave=true
```

## 📈 Roadmap

### Version 1.0 (Current)
- [x] Interactive map interface
- [x] Basic network analysis
- [x] Park intersection detection
- [x] Centrality calculations
- [x] Report generation

### Version 2.0 (Q2 2024)
- [ ] Microservices backend architecture
- [ ] FastAPI REST endpoints
- [ ] Redis caching layer
- [ ] Batch processing capabilities
- [ ] User accounts and project management

### Version 3.0 (Q4 2024)
- [ ] Global city database with precomputed networks
- [ ] Machine learning impact predictions
- [ ] Advanced demographic analysis
- [ ] Multi-language support
- [ ] Mobile app companion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenStreetMap** contributors for providing open geographic data
- **OSMnx** developers for excellent network analysis tools
- **NetworkIt** team for high-performance graph algorithms
- **Streamlit** for making web app development accessible

## 📞 Contact

- **Project Lead**: [Your Name](mailto:your.email@domain.com)
- **GitHub**: [https://github.com/your-username/OASIS](https://github.com/your-username/OASIS)
- **Issues**: [Report bugs and request features](https://github.com/your-username/OASIS/issues)

---

<div align="center">
  <strong>🌳 Built with ❤️ for sustainable cities 🌳</strong>
</div>