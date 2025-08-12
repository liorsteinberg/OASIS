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
    
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )