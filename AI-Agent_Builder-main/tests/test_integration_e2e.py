
import pytest
import httpx
import os
import time
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_DOC_PATH = Path("tests/data/sample_cancellation.pdf") # Ensure this file exists or use a dummy

@pytest.mark.asyncio
async def test_e2e_document_pipeline():
    """
    End-to-end test of the document processing pipeline.
    
    Steps:
    1. Check if system is healthy
    2. Upload a document
    3. Poll for completion
    4. Query the document
    """
    async with httpx.AsyncClient(base_url=API_URL, timeout=60.0) as client:
        # 1. Health Check
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Skip upload if sample file doesn't exist
        if not TEST_DOC_PATH.exists():
            pytest.skip(f"Test file not found: {TEST_DOC_PATH}")
            
        # 2. Upload Document
        files = {"file": ("sample.pdf", open(TEST_DOC_PATH, "rb"), "application/pdf")}
        response = await client.post("/api/documents/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        document_id = data["document_id"]
        assert document_id is not None
        
        print(f"Uploaded document: {document_id}")
        
        # 3. Poll for completion
        max_retries = 60
        for i in range(max_retries):
            response = await client.get(f"/api/documents/{document_id}")
            assert response.status_code == 200
            status = response.json().get("status")
            
            print(f"Poll {i}: Status = {status}")
            
            if status == "completed":
                break
            elif status == "failed":
                pytest.fail(f"Document processing failed: {response.json().get('error')}")
                
            await asyncio.sleep(1) # Wait 1s between polls
        else:
            pytest.fail("Timeout waiting for document processing")
            
        # 4. Query Document
        query_payload = {
            "query": "What is the cancellation policy?",
            "mode": "standard",
            "include_sources": True
        }
        response = await client.post(f"/api/chat/{document_id}", json=query_payload)
        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["answer"] is not None
        print(f"Query Answer: {chat_data['answer']}")
        
        # 5. Test Voice Endpoint (Mocked audio)
        # We can implement this if we have a sample audio file, otherwise skipping for now
