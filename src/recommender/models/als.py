# learns compact latent vectors for users and movies (so info from the user factors and the movie factors)
# learned (not explicitly given) interaction behaviors 
# item cf builds from movie to movie similarities 

import numpy as np 

from scipy.sparse import csr_matrix 
from implicit.als import AlternatingLeastSquares

class ALSRecommender: 

    def __init__(self, 
                 factors = 32, #give every suer and move a learned vector of 64 numbers 
                 regularization = 0.01, #regular penality for large / complex factors 
                 iterations = 20, 
                 random_state = 42): 
        
        self.model = AlternatingLeastSquares(factors = factors, 
                                             regularization= regularization, 
                                             iterations = iterations, 
                                             random_state=random_state)
        self.user_to_idx = None
        self.idx_to_user = None 

        self.movie_to_idx = None 
        self.idx_to_movie = None 

        self.user_item_matrix = None 

        self.seen_by_user = None 
        self.liked_by_user = None

    def fit(self, train_df): 
        
        relevant_df = train_df[train_df['relevant']].copy()

        #internal user and movie ids 
        user_ids = train_df['user_id'].unique()
        movie_ids = train_df['movie_id'].unique()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}

        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}

        # interactions --> matrix coordinates 
        user_indices = relevant_df['user_id'].map(self.user_to_idx).to_numpy()
        movie_indices = relevant_df['movie_id'].map(self.movie_to_idx).to_numpy()

        values = np.ones(len(relevant_df))

        #sparese user-item matrix 
        self.user_item_matrix = csr_matrix((values, (user_indices, movie_indices)), shape = (len(self.user_to_idx),len(self.movie_to_idx)))
        # rows are users, columns are movies 

        self.seen_by_user = (train_df.groupby("user_id")["movie_id"].apply(set).to_dict())
        self.liked_by_user = (relevant_df.groupby("user_id")["movie_id"].apply(list).to_dict())

        self.model.fit(self.user_item_matrix)

        return self
    
    def recommend(self, user_id, k=10, return_scores = False):
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id] 

        movie_indices, scores = self.model.recommend(
            userid = user_idx, user_items = self.user_item_matrix[user_idx], N = k * 5, filter_already_liked_items = True)
        
        seen_movies = self.seen_by_user.get(user_id, set())
        
        recommendations = []
        for idx, score in zip(movie_indices, scores):
            movie_id = self.idx_to_movie[idx]

            if movie_id not in seen_movies:
                if return_scores:
                    recommendations.append((movie_id, float(score)))
                else:
                    recommendations.append(movie_id)
                
            if len(recommendations) == k:
                break

        return recommendations 
    
    def recommend_from_items(self, liked_movie_ids, k = 10, return_scores = False):
        liked_indices = [self.movie_to_idx[movie_id] for movie_id in liked_movie_ids if movie_id in self.movie_to_idx]
        if not liked_indices:
            return []
        
        #temporary 1-user x num_movies interaction matrix 
        values = np.ones(len(liked_indices))
        rows = np.zeros(len(liked_indices))
        user_items = csr_matrix((values, (rows, liked_indices)), shape = (1, len(self.movie_to_idx))) #row = user, column = movie  0s and 1s

        item_indices, scores = self.model.recommend(userid = 0, 
                                                    user_items = user_items, 
                                                    N=k, filter_already_liked_items = True, 
                                                    recalculate_user = True)
        
        recommendations = []
        for item_idx, score in zip(item_indices, scores):
            movie_id = self.idx_to_movie[item_idx]
            if return_scores:
                recommendations.append((movie_id, float(score)))
            else:
                recommendations.append(movie_id)

        return recommendations 

