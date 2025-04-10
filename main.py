# Sorry I uploaded the wrong API endpoint. Here is the correct one
# https://huggingface.co/spaces/nishantdd/shl+

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from engine import setup_engine, get_top_k_recommendations, enrich_query_with_gemini
from functools import lru_cache

app = FastAPI()

@lru_cache()
def get_engine():
    """Cached engine setup to prevent multiple loads"""
    return setup_engine()

# Remove global initialization
# model, index, data, gemini_api_key = setup_engine()

class QueryRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: QueryRequest):
    """
    Endpoint to get assessment recommendations based on a job description or query.
    """
    try:
        # Get engine components only when needed
        model, index, data, gemini_api_key = get_engine()
        
        # Get the original query
        original_query = request.query
        
        # Enrich the query with Gemini if API key is available
        enriched_query = original_query
        if gemini_api_key:
            enriched_query = enrich_query_with_gemini(original_query, gemini_api_key)
        
        # Get recommendations
        results = get_top_k_recommendations(enriched_query, model, index, data, top_k=10)
        
        # Format the results according to the required response format
        formatted_results = []
        for result in results:
            # Extract duration from the Assessment Length field
            duration_str = result.get('duration', '')
            duration = 0
            if duration_str:
                # Try to extract numeric value from the string
                import re
                duration_match = re.search(r'(\d+)', duration_str)
                if duration_match:
                    duration = int(duration_match.group(1))
            
            # Format test types
            test_types = []
            test_type = result.get('test_type', '')
            if test_type:
                # Map test type codes to full names
                test_type_map = {
                    'K': 'Knowledge & Skills',
                    'P': 'Personality & Behaviour',
                    'C': 'Competencies',
                    'A': 'Aptitude',
                    'E': 'English',
                    'B': 'Behavioral',
                    'D': 'Development',
                    'AEBCDP': 'Multiple'
                }
                for code, name in test_type_map.items():
                    if code in test_type:
                        test_types.append(name)
            
            # If no test types were mapped, add a default
            if not test_types:
                test_types = ['Assessment']
            
            # Create the formatted assessment
            assessment = {
                "url": result.get('url', ''),
                "adaptive_support": "Yes" if result.get('adaptive_irt', '').lower() == 'yes' else "No",
                "description": result.get('description', ''),
                "duration": duration,
                "remote_support": "Yes" if result.get('remote_testing', '').lower() == 'yes' else "No",
                "test_type": test_types
            }
            formatted_results.append(assessment)
        
        return {"recommended_assessments": formatted_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")

@app.get("/")
def root():
    return {"message": "SHL Recommender API is running! You can go to /docs endpoint for an interactive API query"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
