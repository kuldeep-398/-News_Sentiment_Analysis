"""
API endpoints for the news sentiment analysis application.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import time
import uuid
from utils import (
    process_company_news,
    generate_summary_for_tts,
    generate_hindi_tts
)

# Initialize FastAPI app
app = FastAPI(
    title="News Sentiment Analysis API",
    description="API for extracting and analyzing news articles about companies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Models
class CompanyRequest(BaseModel):
    company_name: str
    num_articles: int = 10

class CompanyResponse(BaseModel):
    request_id: str
    status: str
    message: str

class AnalysisResult(BaseModel):
    company: str
    articles: List[Dict[str, Any]]
    comparative_sentiment_score: Dict[str, Any]
    final_sentiment_analysis: str
    audio: str

# In-memory cache for analysis results
analysis_cache = {}

# Background task to process company news
def process_company_news_task(request_id: str, company_name: str, num_articles: int):
    try:
        # Process news articles
        result = process_company_news(company_name, num_articles)
        
        # Generate TTS summary
        summary_text = generate_summary_for_tts(result)
        audio_file = f"data/{request_id}.mp3"
        
        # Generate Hindi TTS
        generate_hindi_tts(summary_text, audio_file)
        
        # Update result with audio file path
        result["audio"] = audio_file
        
        # Save result to cache
        analysis_cache[request_id] = {
            "status": "completed",
            "result": result
        }
        
        # Save result to file for persistence
        with open(f"data/{request_id}.json", "w") as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        # Update cache with error
        analysis_cache[request_id] = {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/analyze", response_model=CompanyResponse)
async def analyze_company(request: CompanyRequest, background_tasks: BackgroundTasks):
    """
    Start analysis of news articles for a company.
    """
    request_id = str(uuid.uuid4())
    
    # Initialize cache entry
    analysis_cache[request_id] = {
        "status": "processing",
        "message": f"Processing news for {request.company_name}"
    }
    
    # Start background task
    background_tasks.add_task(
        process_company_news_task,
        request_id,
        request.company_name,
        request.num_articles
    )
    
    return {
        "request_id": request_id,
        "status": "processing",
        "message": f"Processing news for {request.company_name}"
    }

@app.get("/api/status/{request_id}")
async def get_analysis_status(request_id: str):
    """
    Get the status of an analysis request.
    """
    if request_id not in analysis_cache:
        # Check if result file exists
        result_file = f"data/{request_id}.json"
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            
            analysis_cache[request_id] = {
                "status": "completed",
                "result": result
            }
        else:
            raise HTTPException(status_code=404, detail="Analysis request not found")
    
    return analysis_cache[request_id]

@app.get("/api/result/{request_id}", response_model=AnalysisResult)
async def get_analysis_result(request_id: str):
    """
    Get the result of a completed analysis.
    """
    if request_id not in analysis_cache:
        # Check if result file exists
        result_file = f"data/{request_id}.json"
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            
            analysis_cache[request_id] = {
                "status": "completed",
                "result": result
            }
        else:
            raise HTTPException(status_code=404, detail="Analysis result not found")
    
    cache_entry = analysis_cache[request_id]
    
    if cache_entry["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {cache_entry['status']}"
        )
    
    return cache_entry["result"]

@app.get("/api/companies")
async def get_companies():
    """
    Get a list of example companies for the dropdown.
    """
    return {
        "companies": [
            "Tesla",
            "Apple",
            "Microsoft",
            "Google",
            "Amazon",
            "Meta",
            "Netflix",
            "Nvidia",
            "Intel",
            "AMD"
        ]
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "timestamp": time.time()}

# Serve audio files
@app.get("/api/audio/{request_id}")
async def get_audio(request_id: str):
    """
    Get the audio file for a completed analysis.
    """
    audio_file = f"data/{request_id}.mp3"
    
    if not os.path.exists(audio_file):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return {"audio_url": audio_file}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
