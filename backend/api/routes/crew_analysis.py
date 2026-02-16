"""
CrewAI Analysis Routes

Endpoints for running CrewAI-powered document analysis,
providing high-level multi-agent collaboration for 
document comparison and deep analysis tasks.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...agents.crew_orchestrator import DocumentAnalysisCrew
from .documents import documents_store

logger = logging.getLogger(__name__)
router = APIRouter()


class CrewAnalysisRequest(BaseModel):
    """Request model for CrewAI analysis"""
    document_type: str = Field(
        default="general",
        description="Type of document: financial, legal, technical, general"
    )
    analysis_depth: str = Field(
        default="standard",
        description="Depth of analysis: quick, standard, deep"
    )


class CrewAnalysisResponse(BaseModel):
    """Response model for CrewAI analysis"""
    document_id: str
    status: str
    analysis: Optional[str] = None
    document_type: str
    error: Optional[str] = None


class DocumentCompareRequest(BaseModel):
    """Request model for document comparison"""
    document_id_1: str = Field(..., description="First document ID")
    document_id_2: str = Field(..., description="Second document ID")


class DocumentCompareResponse(BaseModel):
    """Response model for document comparison"""
    status: str
    comparison: Optional[str] = None
    error: Optional[str] = None


@router.post("/{document_id}/crew-analyze", response_model=CrewAnalysisResponse)
async def crew_analyze_document(
    document_id: str,
    request: CrewAnalysisRequest
):
    """
    Run CrewAI multi-agent analysis on a document.
    
    Uses a crew of specialized agents (Analyst, Fact Checker, Summarizer)
    to perform deep document analysis with cross-validation.
    """
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready. Status: {doc['status']}"
        )
    
    try:
        # Get document content
        results = doc.get("results", {})
        text_content = results.get("text_analysis", {}).get("full_text", "")
        
        if not text_content:
            fused = results.get("fused_output", {})
            text_content = fused.get("text", "No text content available")
        
        # Run CrewAI analysis
        crew = DocumentAnalysisCrew()
        result = crew.analyze_document(
            document_content=text_content,
            document_type=request.document_type,
            analysis_depth=request.analysis_depth
        )
        
        return CrewAnalysisResponse(
            document_id=document_id,
            status=result["status"],
            analysis=result.get("analysis"),
            document_type=request.document_type,
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"CrewAI analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"CrewAI analysis failed: {str(e)}"
        )


@router.post("/compare", response_model=DocumentCompareResponse)
async def compare_documents(request: DocumentCompareRequest):
    """
    Compare two documents using CrewAI agents.
    
    Identifies commonalities, contradictions, and unique information
    across two processed documents.
    """
    doc1 = documents_store.get(request.document_id_1)
    doc2 = documents_store.get(request.document_id_2)
    
    if not doc1 or not doc2:
        raise HTTPException(status_code=404, detail="One or both documents not found")
    
    if doc1["status"] != "completed" or doc2["status"] != "completed":
        raise HTTPException(status_code=400, detail="Both documents must be fully processed")
    
    try:
        text1 = doc1.get("results", {}).get("text_analysis", {}).get("full_text", "")
        text2 = doc2.get("results", {}).get("text_analysis", {}).get("full_text", "")
        
        crew = DocumentAnalysisCrew()
        result = crew.compare_documents(text1, text2)
        
        return DocumentCompareResponse(
            status=result["status"],
            comparison=result.get("comparison"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )
