from pathlib import Path 
from src.recommender.ranking.candidates import generate_candidates_for_user


import joblib 
import numpy as np 
import pandas as pd 

class RecommendationEngine:
    
    def __init__(self, artificat_dir):
        artificat_dir = Path(artificat_dir)

        self.popularity = joblib.load(artificat_dir/'popularity.pkl')
        self.content = joblib.load(artificat_dir/'content.pkl')
        self.item_cf = joblib.load(artificat_dir/'item_cf.pkl')
        self.als = joblib.load(artificat_dir/'als.pkl')

        self.ranker = joblib.load(artificat_dir/'ranker.pkl')
        self.feature_cols = joblib.load(artificat_dir/'ranker_features.pkl')
        self.movies = pd.read_parquet(artificat_dir / 'movies.parquet')

    def recommend(self, user_id, k = 10, candidate_k = 100):
        models = {
        "popularity" : self.popularity, 
        "content": self.content,
        "item_cf" : self.content, 
        "als" : self.content
        }

        candidate_df = generate_candidates_for_user(model=models, user_id=user_id, candidate_k=candidate_k)
        if candidate_df.empty:
            return []
        
        candidate_df['ranker_score'] = self.ranker.predict(candidate_df[self.feature_cols])
        candidate_df = candidate_df.sort_values("ranker_score", ascending = False).head(k)

        results = candidate_df.merge(self.movies[['movie_id','title','genres']], on = 'movie_id',how = "left")

        return results[['movie_id','title','genres', 'ranker_score']].to_dict(orient = 'records')
    

    def recommend_from_items(self, liked_movie_ids, k = 10, candidate_k = 100):
        liked_movie_ids = set(liked_movie_ids)
        models = {
        "popularity" : self.popularity, 
        "content": self.content,
        "item_cf" : self.content, 
        "als" : self.content }

        candidate_df = generate_candidates_for_user(model=models, liked_movie_ids=liked_movie_ids, candidate_k=candidate_k)
        if candidate_df.empty:
            return []

        candidate_df['ranker_score'] = (self.ranker.predict(candidate_df[self.feature_cols]))
        top_movies = (candidate_df.sort_values('ranker_score', ascending= False)).head(k)

        top_movies = top_movies.merge(self.movies[['movie_id','title','genres']], on = 'movie_id', how = 'left')

        return top_movies[['movie_id','title','genres','ranker_score']].to_dict(orient = 'records')
   
