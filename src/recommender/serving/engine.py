from pathlib import Path 

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

        candidates = {}
        for model_name, model in models.items():
            recommendations = model.recommend(user_id, candidate_k, return_scores = True)
            for rank, (movie_id, score) in enumerate(recommendations, start = 1):
                if movie_id not in candidates: 
                    candidates[movie_id] = {'movie_id': movie_id}
                candidates[movie_id][f'{model_name}_score'] = float(score)
                candidates[movie_id][f'{model_name}_rank'] = rank
        candidate_df = pd.DataFrame(candidates.values())
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

        candidates = {}
        for model_name, model in models.items():
            recommendations = model.recommend_from_items(liked_movie_ids = liked_movie_ids, k = candidate_k, return_scores = True)
            for rank, (movie_id, score) in enumerate(recommendations, start = 1):
                if movie_id in liked_movie_ids:
                    continue 
                if movie_id not in candidates: 
                    candidates[movie_id] = {'movie_id': movie_id}
                candidates[movie_id][f'{model_name}_score'] = float(score)
                candidates[movie_id][f'{model_name}_rank'] = rank
        if not candidates:
            return []
        candidate_df = pd.DataFrame(candidates.values())

        candidate_df['ranker_score'] = (self.ranker.predict(candidate_df[self.feature_cols]))
        top_movies = (candidate_df.sort_values('ranker_score', ascending= False)).head(k)

        top_movies = top_movies.merge(self.movies[['movie_id','title','genres']], on = 'movie_id', how = 'left')
        return top_movies[['movie_id','title','genres','ranker_score']].to_dict(orient = 'records')
   
