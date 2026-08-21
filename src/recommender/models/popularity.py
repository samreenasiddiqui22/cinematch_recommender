class PopularityRecommender:

    def __init__(self):
        self.popular_movies = None
        self.seen_by_user = None 

    def fit(self, train_df):
        relevant_df = train_df[train_df['relevant']]
        self.popular_movies = (relevant_df.groupby("movie_id").size().sort_values(ascending = False))
        self.seen_by_user = (train_df.groupby('user_id')['movie_id'].apply(set).to_dict())

        return self 
    
    def recommend(self, user_id, k = 10, return_scores = False):
        seen_movies = self.seen_by_user.get(user_id,set())
        recommendations = []
        for movie_id, score in self.popular_movies.items(): #score is basically how many times that liked movie was seen 
            if movie_id not in seen_movies:
                if return_scores:
                    recommendations.append((movie_id, float(score)))
                else:
                    recommendations.append(movie_id)
            if len(recommendations) == k:
                break

        return recommendations[:k]
