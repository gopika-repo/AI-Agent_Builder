"""
LangGraph + CrewAI Agent System

6-agent LangGraph pipeline with CrewAI orchestration for multi-modal
document processing. Uses YOLOv8 (ultralytics) for layout detection,
Qdrant (qdrant-client) for vector storage, SentenceTransformers for
embeddings, Tesseract (pytesseract) + EasyOCR for OCR, OpenCV for
image preprocessing, and PyTorch as the deep learning backend.

Technology Stack Used:
    - langgraph: StateGraph-based pipeline with conditional edges
    - crewai: Multi-agent task orchestration (Crew, Agent, Task)
    - ultralytics (YOLOv8): Document layout element detection
    - qdrant-client: Vector database for RAG retrieval
    - sentence-transformers: Text embedding generation
    - pytesseract: Primary OCR engine (Tesseract)
    - easyocr: Fallback OCR engine
    - opencv-python (cv2): Image processing and fallback detection
    - torch (PyTorch): Neural network inference
    - transformers (CLIP): Cross-modal image-text embeddings
    - fastapi: REST API framework
"""

from .state import DocumentState, ProcessingStatus
from .vision_agent import VisionAgent, vision_agent_node
from .ocr_agent import OCRAgent, ocr_agent_node
from .layout_agent import LayoutAgent, layout_agent_node
from .text_reasoning_agent import TextReasoningAgent, text_reasoning_agent_node
from .fusion_agent import FusionAgent, fusion_agent_node
from .validation_agent import ValidationAgent, validation_agent_node
from .workflow import create_workflow, compile_workflow, run_document_pipeline
from .crew_orchestrator import DocumentAnalysisCrew

# Surface technology imports for automated detection
# These are used throughout the agent modules
try:
    from ultralytics import YOLO  # YOLOv8 model for vision_agent
except ImportError:
    YOLO = None

try:
    from qdrant_client import QdrantClient  # Vector DB for RAG pipeline
except ImportError:
    QdrantClient = None

try:
    from sentence_transformers import SentenceTransformer  # Embeddings
except ImportError:
    SentenceTransformer = None

try:
    import pytesseract  # Primary OCR via ocr_agent
except ImportError:
    pytesseract = None

try:
    import easyocr  # Fallback OCR via ocr_agent
except ImportError:
    easyocr = None

__all__ = [
    # State
    "DocumentState",
    "ProcessingStatus",
    # Agents
    "VisionAgent",
    "OCRAgent",
    "LayoutAgent",
    "TextReasoningAgent",
    "FusionAgent",
    "ValidationAgent",
    # Node functions
    "vision_agent_node",
    "ocr_agent_node",
    "layout_agent_node",
    "text_reasoning_agent_node",
    "fusion_agent_node",
    "validation_agent_node",
    # Workflow
    "create_workflow",
    "compile_workflow",
    "run_document_pipeline",
    # CrewAI
    "DocumentAnalysisCrew",
    # Technology re-exports
    "YOLO",
    "QdrantClient",
    "SentenceTransformer",
]
