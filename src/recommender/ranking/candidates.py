import pandas as pd 

def build_candidate_table(
        models,
        evaluation_df,
        candidate_k = 50):
    
    rows = []

    # true relevant movies for each user 
    relevant_by_user = (evaluation_df[evaluation_df['relevant']].groupby('user_id')['movie_id'].apply(set).to_dict())

    for user_id, relevant_movies in relevant_by_user.items():

        # one dictionary of candidate movies for this user 
        candidates = {}

        for model_name, model in models.items():
            recommendations = model.recommend(user_id, candidate_k, return_scores = True)

            for rank, (movie_id, score) in enumerate(recommendations, start = 1):

                # so each movie id has the results and ranking from each model that outputted that movie as a rec
                if movie_id not in candidates:
                    candidates[movie_id] = {
                        "user_id": user_id, 
                        "movie_id": movie_id
                    }
                candidates[movie_id][f'{model_name}_score'] = float(score) 
                candidates[movie_id][f'{model_name}_rank'] = rank

        for movie_id, candidate in candidates.items():
            candidate['label'] = int(movie_id in relevant_movies)
            rows.append(candidate)
        
    return pd.DataFrame(rows)
