import pandas as pd 

def build_candidate_table(
        models,
        evaluation_df,
        candidate_k = 50):
    
    rows = []

    # true relevant movies for each user 
    relevant_by_user = (evaluation_df[evaluation_df['relevant']].groupby('user_id')['movie_id'].apply(set).to_dict())

    for user_id, relevant_movies in relevant_by_user.items():
        candidate_df = generate_candidates_for_user(model=models, user_id=user_id, candidate_k=candidate_k)

        if candidate_df.empty:
                continue
        candidate_df['user_id'] = user_id
        candidate_df['label'] = candidate_df['movie_id'].isin(relevant_movies).astype(int)

        rows.append(candidate_df)
    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

def generate_candidates_for_user(model, user_id=None, liked_movie_ids=None, candidate_k=100):

    candidates = {}

    for model_name, model in model.items():
        if liked_movie_ids is not None:
            recommendations = model.recommend_from_items(liked_movie_ids = liked_movie_ids, k=candidate_k, return_scores = True)

        else:
            recommendations = model.recommend(user_id=user_id, k=candidate_k, return_scores = True)

        for rank, (movie_id, score) in enumerate(recommendations, start=1):
            if movie_id not in candidates: 
                candidates[movie_id] = {'movie_id': movie_id}
            candidates[movie_id][f"{model_name}_score"] = float(score)
            candidates[movie_id][f"{model_name}_rank"] = rank

    return pd.DataFrame(candidates.values())

