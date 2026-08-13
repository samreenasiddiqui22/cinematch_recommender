class PopularityRecommender:

    def __init__(self):
        self.popular_movies = None
        self.seen_by_user = None 

    def fit(self, train_df):
        relevant_df = train_df[train_df['relevant']]
        self.popular_movies = (relevant_df.groupby("movie_id").size().sort_values(ascending = False).index.tolist())
        self.seen_by_user = (train_df.groupby('user_id')['movie_id'].apply(set).to_dict())

        return self 
    
    def recommend(self, user_id, k = 10):
        seen_movies = self.seen_by_user.get(user_id,set())
        recommendations = [movie_id for movie_id in self.popular_movies if movie_id not in seen_movies]

        return recommendations[:k]
