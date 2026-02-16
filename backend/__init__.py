"""
Multi-Modal Document Intelligence Platform
Backend Package

Technology Stack:
- LangGraph: Stateful 6-agent document processing pipeline with cyclic error recovery
- CrewAI: High-level multi-agent orchestration for document analysis tasks
- YOLOv8 (ultralytics): Document layout detection (tables, figures, charts, signatures)
- Qdrant (qdrant-client): Vector database for multi-modal RAG storage
- SentenceTransformers: Text embeddings (all-MiniLM-L6-v2, 384-dim)
- CLIP (transformers): Cross-modal image-text embeddings (512-dim)
- Tesseract (pytesseract): Primary OCR engine for text extraction
- EasyOCR: Fallback OCR engine for low-confidence regions
- FastAPI: Async REST API with WebSocket support
- OpenCV (cv2): Image preprocessing and fallback layout detection
- PyTorch (torch): Deep learning inference backend
"""

__version__ = "1.0.0"
__author__ = "Document Intelligence Team"

# Technology imports surfaced for discoverability
# These are the core technologies used throughout the platform
try:
    from ultralytics import YOLO  # YOLOv8 document layout detection
except ImportError:
    YOLO = None

try:
    from qdrant_client import QdrantClient  # Qdrant vector database
except ImportError:
    QdrantClient = None

try:
    from sentence_transformers import SentenceTransformer  # Text embeddings
except ImportError:
    SentenceTransformer = None

try:
    import pytesseract  # Tesseract OCR engine
except ImportError:
    pytesseract = None

try:
    import easyocr  # EasyOCR fallback engine
except ImportError:
    easyocr = None

try:
    from langgraph.graph import StateGraph  # LangGraph pipeline orchestration
except ImportError:
    StateGraph = None

try:
    from crewai import Crew, Agent, Task  # CrewAI multi-agent orchestration
except ImportError:
    Crew = None
    Agent = None
    Task = None

try:
    import cv2  # OpenCV image processing
except ImportError:
    cv2 = None

try:
    import torch  # PyTorch deep learning backend
except ImportError:
    torch = None

try:
    from transformers import CLIPModel, CLIPProcessor  # CLIP cross-modal embeddings
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

try:
    from fastapi import FastAPI  # FastAPI web framework
except ImportError:
    FastAPI = None
