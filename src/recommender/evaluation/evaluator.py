import pandas as pd 

from .metrics import precision_at_k, recall_at_k, ndcg_at_k

def evaluate_model(model, evaluation_df, k=10):
    results = []

    relevant_df = evaluation_df[evaluation_df['relevant']]

    for user_id, user_data in relevant_df.groupby("user_id"):
        relevant_movies = set(user_data["movie_id"])

        recommendations = model.recommend(user_id = user_id, k = k)

        results.append({
                "user_id": user_id, 
                f"precision_at_{k}": precision_at_k(recommendations, relevant_movies, k),
                f"recall_at_{k}": recall_at_k(recommendations, relevant_movies, k),
                f"ndcg_at_{k} ": ndcg_at_k(recommendations, relevant_movies, k)
                })
        
    return pd.DataFrame(results)