#which items tend to be liked by the same users (instead of comparing movies to user_profile averages) , collaborative filtering
import numpy as np
import pandas as pd 

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity 


class ItemCFRecommender:
    def __init__(self):
        self.movie_ids = None
        self.movie_to_idx = None 
        self.similarity_matrix = None
        self.seen_by_user = None
        self.liked_by_user = None
    
    def fit(self, train_df):
        
        relevant_df = train_df[train_df['relevant']].copy()

        #so the columns are movie ids, rows are user ids, 1 and 0s.  can compare similar movies (columns) 
        # by seeing if similar people watched them
        user_item = pd.crosstab(relevant_df['user_id'], relevant_df['movie_id'])
        self.movie_ids = user_item.columns.to_numpy()

        self.movie_to_idx = {movie_id : idx for idx, movie_id in enumerate(self.movie_ids)}
        
        #movie (rows) x user (columns). csr helps with the sparsity of the data
        item_user_matrix = csr_matrix(user_item.values.T)

        self.similarity_matrix = cosine_similarity(item_user_matrix, dense_output= False) #compares every movie to every movie (each row is a movie, 
        # and column is the user who watched (or didnt) watch it)
        self.seen_by_user = train_df.groupby("user_id")['movie_id'].apply(set).to_dict()
        self.liked_by_user = relevant_df.groupby("user_id")['movie_id'].apply(list).to_dict()

        return self 
    
    def recommend(self, user_id, k=10):
        liked_movies = self.liked_by_user.get(user_id, [])
        if not liked_movies:
            return []
        
        liked_indices = [self.movie_to_idx[movie_id] for movie_id in liked_movies if movie_id in self.movie_to_idx]

        scores = np.asarray(self.similarity_matrix[liked_indices].sum(axis = 0)).ravel() # takes the movies the person liked, and adds the 
        # other movies to see which has the greatest similarity. if person likes movie A and B, and movie c is 0.8 similar to movie A and 0.7
        # to movie B, movie C gets score 1.5. then get the greatest sum!
        ranked_indices = np.argsort(scores)[::-1]
        
        seen_movies = self.seen_by_user.get(user_id, set())

        recommendations = []
        for idx in ranked_indices: 
            movie_id = self.movie_ids[idx]
            if movie_id not in seen_movies:
                recommendations.append(movie_id)

            if len(recommendations) == k:
                break 

        return recommendations