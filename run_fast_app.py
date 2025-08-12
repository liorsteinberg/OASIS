#!/usr/bin/env python3
"""
Simple script to start the Fast Park Accessibility App
"""
import uvicorn
from fast_park_app import app

if __name__ == "__main__":
    print("🌳 Starting Fast Park Accessibility App...")
    print("📍 Available cities: Rotterdam (default), Munich")
    print("🔧 Features: Interactive maps, dual visualization methods, mass analysis")
    print("🌐 Opening on http://localhost:8000")
    print("📝 Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )