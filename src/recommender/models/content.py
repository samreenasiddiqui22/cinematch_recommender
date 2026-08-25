import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self):
        self.movie_ids = None 
        self.movie_features = None 
        self.movie_id_to_index = None 
        self.seen_by_user = None
        self.liked_by_user = None 
        self.mlb = None 

    def fit(self,train_df, movies_df):

        movies = movies_df.copy()
        movies["genre_list"] = movies["genres"].str.split("|")
        self.mlb = MultiLabelBinarizer()

        self.movie_features = self.mlb.fit_transform(movies['genre_list'])
        self.movie_ids = movies['movie_id'].to_numpy()
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}

        self.seen_by_user = train_df.groupby("user_id")['movie_id'].apply(set).to_dict()

        relevant_train = train_df[train_df['relevant']]
        self.liked_by_user = relevant_train.groupby("user_id")['movie_id'].apply(list).to_dict()

        return self 
    
    def recommend(self, user_id, k=10, return_scores = False):

        liked_movies = self.liked_by_user.get(user_id, [])
        if len(liked_movies) == 0: 
            return []
        
        seen_movies = self.seen_by_user.get(user_id, set())

        #feature vectors for liked movies 
        liked_indices = [self.movie_id_to_index[movie_id] for movie_id in liked_movies if movie_id in self.movie_id_to_index]
        if len(liked_indices) == 0:
            return []
        liked_features = self.movie_features[liked_indices]

        user_profile = np.mean(liked_features, axis = 0).reshape(1,-1)

        #compare user's profile to every movie 
        scores = cosine_similarity(user_profile, self.movie_features)[0] #how similar the movie is to the user's average profile 

        ranked_indices = np.argsort(scores)[::-1]

        recommendations = []
        for idx in ranked_indices:
            movie_id = self.movie_ids[idx]
            if movie_id not in seen_movies: 
                if return_scores:
                    recommendations.append((movie_id, float(scores[idx])))
                else:
                    recommendations.append(movie_id)
            if len(recommendations) == k: 
                break 
        return recommendations 
    

    def recommend_from_items(self, liked_movie_ids, k =10, return_scores = False):
        liked_movie_ids = set(liked_movie_ids)

        liked_indices = [self.movie_id_to_index[movie_id] for movie_id in liked_movie_ids if movie_id in self.movie_id_to_index]

        if not liked_indices:
            return []
        
        liked_features = self.movie_features[liked_indices]
        user_profile = np.mean(liked_features, axis = 0).reshape(1,-1)

        scores = cosine_similarity(user_profile, self.movie_features).ravel()

        ranked_indices = np.argsort(scores)[::-1]
        recommendations = []

        for idx in ranked_indices: 
            movie_id = self.movie_ids[idx]

            if movie_id not in liked_movie_ids:
                if return_scores:
                    recommendations.append((movie_id, float(scores[idx])))

                else:
                    recommendations.append(movie_id)
            if len(recommendations) == k:
                break 

        return recommendations 
