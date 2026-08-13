import math 

def precision_at_k(recomended, relevant, k):
    recomended_k = recomended[:k]
    if len(recomended_k) == 0:
        return 0 
    hits = set(recomended_k) & set(relevant)
    return len(hits)/len(recomended_k)

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0: 
        return 0
    recomended_k  = recommended[:k]
    hits = set(recomended_k) & set(relevant)
    return len(hits)/len(relevant)

def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant = set(relevant)

    if len(relevant) == 0: 
        return 0 
    
    dcg = 0.0

    for rank, movie_id in enumerate(recommended_k, start = 1):
        if movie_id in relevant: 
            dcg += 1/math.log2(rank+1)

        ideal_hits = min(len(relevant), k)
        idcg = sum (
            1 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

    return dcg / idcg