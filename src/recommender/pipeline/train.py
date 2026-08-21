from pathlib import Path 
import joblib 
import pandas as pd 
from xgboost import XGBRanker 

from src.recommender.models.als import ALSRecommender
from src.recommender.models.content import ContentRecommender
from src.recommender.models.item_cf import ItemCFRecommender
from src.recommender.models.popularity import PopularityRecommender
from src.recommender.data.split import chron_train_val_test 
from src.recommender.ranking.candidates import build_candidate_table


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT/"data"/"data_1m"
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(exist_ok = True)

def load_data():
    ratings_path = DATA_DIR/'ratings.dat'
    movies_path = DATA_DIR / 'movies.dat'

    ratings = pd.read_csv(ratings_path, sep='::',engine='python', encoding='latin-1', names = ["user_id","movie_id","rating","timestamp"])
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'],unit='s')

    movies = pd.read_csv(movies_path, sep='::', engine='python', encoding='latin-1',names=['movie_id', 'title', 'genres'])

    return ratings, movies 


def train(): 

    print('Loading Training Data')
    ratings_df, movies_df = load_data()
    ratings_df['relevant'] = ratings_df['rating'] >= 4.0
    print(f'Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies')

    print("Creating Test/Val/Train Split")
    train_df, val_df, test_df = chron_train_val_test(ratings_df)

    #deliberately only on train_df because its these scores / predictions that make up the candidate dataset 

    print('Training Popularity Model')
    popularity_model = PopularityRecommender()
    popularity_model.fit(train_df)

    print('Training Content Model')
    content_model = ContentRecommender()
    content_model.fit(train_df, movies_df)

    print('Training Item CF Model')
    item_cf_model = ItemCFRecommender()
    item_cf_model.fit(train_df)

    print('Training ALS Model')
    als_model = ALSRecommender(factors = 32, regularization= 0.01, iterations = 20)
    als_model.fit(train_df)


    print("Building Candidate Dataset")
    models = {
    "popularity" : popularity_model, 
    "content": content_model,
    "item_cf" : item_cf_model, 
    "als" : als_model}

    candidate_df = build_candidate_table(models, val_df, 100) #makes predictions on the validation set (from the models trained on the train_df)
    candidate_df = candidate_df.sort_values(by = 'user_id').reset_index(drop = True)

    print("Building Final Ranker")
    feature_cols = ["popularity_score", "popularity_rank","content_score","content_rank","item_cf_score","item_cf_rank","als_score","als_rank"]

    X_rank = candidate_df[feature_cols] 
    y_rank = candidate_df['label'] #if it was liked or not in the validation set 
    qid = candidate_df['user_id']

    print("Training Final Ranker") 
    ranker = XGBRanker (
    objective = "rank:ndcg", #should one particular movie be ranked above another (not is movie A relevant, or predict movie A)
    n_estimators = 300, 
    learning_rate = 0.05, 
    max_depth = 6, 
    subsample = 0.8, 
    colsample_bytree = 0.8, 
    tree_method = 'hist',
    random_state = 42) 

    ranker.fit(X_rank, y_rank, qid = qid, verbose = False) # X_rank purposefully only has outcomes from the val_df because otherwise the model (trained on train_df) would have been predicting scores for movies also in train_df which is leakage. 

    print("Refitting base models on train and validation")
    #these new models will be used to score / create candidate table for the test_df
    #then the trained_ranker (who was fit on the scores of val_df) will be used on test_df 

    final_train_df = pd.concat([train_df, val_df], ignore_index=True)

    final_popularity = PopularityRecommender()
    final_popularity.fit(final_train_df)

    final_content = ContentRecommender()
    final_content.fit(final_train_df, movies_df)

    final_item_cf = ItemCFRecommender()
    final_item_cf.fit(final_train_df)

    final_als = ALSRecommender(factors = 32, regularization= 0.01, iterations = 20)
    final_als.fit(final_train_df)

    print('Saving Model Artifacts')
    ARTIFACT_DIR.mkdir(parents = True, exist_ok= True)
    joblib.dump(final_popularity, ARTIFACT_DIR / 'popularity.pkl')
    joblib.dump(final_content, ARTIFACT_DIR / 'content.pkl')
    joblib.dump(final_item_cf, ARTIFACT_DIR / 'item_cf.pkl')
    joblib.dump(final_als, ARTIFACT_DIR / 'als.pkl')
    joblib.dump(ranker, ARTIFACT_DIR / 'ranker.pkl')
    joblib.dump(feature_cols, ARTIFACT_DIR/'ranker_features.pkl')

    movies_df.to_parquet(ARTIFACT_DIR / 'movies.parquet', index = False)

    print("Training complete")

if __name__ == '__main__':
    train()


                                            

    


    









