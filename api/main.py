from pathlib import Path 
from fastapi import FastAPI, HTTPException
from src.recommender.serving.engine import RecommendationEngine
from pydantic import BaseModel 
from typing import List 
from functools import lru_cache
from fastapi import FastAPI,HTTPException, Depends

class PreferenceRequest(BaseModel):
    liked_movie_ids: List[int]
    k: int = 10 


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'

app = FastAPI( title = 'CineMatch Recommendation API', version = '1.0.0')

@lru_cache
def get_engine():
    return RecommendationEngine(ARTIFACT_DIR)

@app.get('/health')
def health():
    return {'status':'healthy'}

@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, k: int = 10, engine: RecommendationEngine = Depends(get_engine)): 
    if k < 1 or k > 100: 
        raise HTTPException(status_code=400, detail='k must be between 1 and 100')
    
    results = engine.recommend(user_id = user_id, k = k)
    
    if not recommendations:
        raise HTTPException(status_code=400, detail='No Recommendations available for user {user_id}')
    
    return {"user_id": user_id, "k": k, "recommendations": results}

@app.post('/recommendations/preferences')
def recommendations_from_preferences(request: PreferenceRequest, engine: RecommendationEngine = Depends(get_engine)):
    if len(request.liked_movie_ids) == 0:
        raise HTTPException(status_code=400, detail = 'At least one liked movie is required')
    if request.k < 1 or request.k > 100: 
        raise HTTPException(status_code = 400, detail = 'k must be between 1 and 100')
    results = engine.recommend_from_items(liked_movie_ids = request.liked_movie_ids, k = request.k)
    if not results: 
        raise HTTPException(status_code=400, detail = 'No recommendations could be generated')
    return {
        "liked_movie_ids": request.liked_movie_ids, 
        "k": request.k, 
        "recommendations": results
    }
    