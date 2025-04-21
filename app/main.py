"""
Application entry point for FastAPI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

# Import API routers
from app.apis.summarier import router as summarize_router

# Base directory for the application
BASE_DIR = Path(__file__).resolve().parent

# Create FastAPI app
app = FastAPI(
    title="Video Meeting Summarizer",
    description="API for summarizing Video Meeting calls",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include API routers
app.include_router(summarize_router)

# Root endpoint - serve the HTML file
@app.get("/")
async def read_index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))