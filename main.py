
# Sorry I uploaded the wrong API endpoint. Here is the correct one
# https://huggingface.co/spaces/nishantdd/shl

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn
from engine import setup_engine, get_top_k_recommendations, enrich_query_with_gemini

app = FastAPI()

# Assume these are defined globally or imported
model, index, data, gemini_api_key = setup_engine()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
async def recommend(request: QueryRequest):
    original_query = request.query
    enriched_query = enrich_query_with_gemini(original_query, gemini_api_key)

    results = get_top_k_recommendations(enriched_query, model, index, data, top_k=request.top_k)
    
    return {
        "original_query": original_query,
        "enriched_query": enriched_query,
        "recommendations": results
    }

@app.get("/")
def root():
    return {"message": "SHL Recommender API is running! You can go to /docs endpoint for an interactive API query"}
