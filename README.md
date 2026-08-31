CineMatch: End-to-End Movie Recommendation Platform
CineMatch is an end-to-end movie recommendation system built to demonstrate both recommendation-system modeling and production ML engineering.

The system combines multiple recommendation strategies—popularity, content-based filtering, item-item collaborative filtering, and matrix factorization—with an XGBoost learning-to-rank model. It supports recommendations for both existing users and brand-new users who provide a small set of favorite movies.

The project also includes experiment tracking, reproducible training, model artifact persistence, a FastAPI inference service, automated testing, Docker packaging, GitHub Actions CI, and local Kubernetes deployment.

Architecture
                           MovieLens 1M
                                |
                                v
                    Chronological User Split
                     Train / Validation / Test
                                |
                                v
               +--------------------------------+
               |      Base Recommenders         |
               |                                |
               |  Popularity                    |
               |  Content-Based                 |
               |  Item-Item Collaborative       |
               |  ALS Matrix Factorization      |
               +---------------+----------------+
                               |
                               v
                     Candidate Generation
                               |
                               v
                Scores + Ranks From Each Model
                               |
                               v
                    XGBoost Learning-to-Rank
                         objective: rank:ndcg
                               |
                               v
                       Top-K Recommendations
                               |
                         Saved Artifacts
                               |
                               v
                    RecommendationEngine
                         /             \
                        /               \
               Existing User        New User
                  userId          favorite movies
                        \               /
                         \             /
                              FastAPI
                                |
                              Docker
                                |
                            Kubernetes
Recommendation Approach
CineMatch uses a two-stage recommendation architecture.

Stage 1: Candidate Generation
Four recommendation models independently identify potentially relevant movies.

Popularity
Ranks movies based on the number of positive interactions in the training data.

This provides a strong non-personalized baseline and fallback signal.

Content-Based Filtering
Movies are represented using multi-hot encoded genre features.

For an existing user, the system creates a preference profile from genres of positively rated movies and ranks unseen movies using cosine similarity.

For a new user, the profile is constructed directly from the movies supplied during onboarding.

Item-Item Collaborative Filtering
Movies are represented by the users who positively interacted with them.

Cosine similarity is used to identify movies with similar audiences.

A user's candidate score is produced by aggregating similarity scores from movies they previously liked.

ALS Matrix Factorization
Implicit-feedback Alternating Least Squares learns latent embeddings for users and movies.

For existing users, recommendations use the learned user factor.

For brand-new users, CineMatch builds a temporary interaction vector from their selected favorite movies and recalculates a temporary ALS user factor without retraining the model.

Stage 2: Hybrid Learning-to-Rank
Candidate movies from the four base recommenders are combined into a shared feature table.

Features include:

popularity_score
popularity_rank
content_score
content_rank
item_cf_score
item_cf_rank
als_score
als_rank
An XGBRanker with:

objective = rank:ndcg
learns how to combine these signals and rerank candidate movies for each user.

This architecture separates:

Candidate Generation
"What movies might be relevant?"

from

Ranking
"In what order should we show them?"
Data
The project uses the MovieLens 1M dataset.

The data contains approximately:

6,040 users

3,883 movies

1 million ratings

A rating is considered a positive/relevant interaction when:

rating >= 4.0
Train / Validation / Test Strategy
Interactions are split chronologically within each user rather than randomly.

earliest interactions -------------------------- latest interactions

|---------------- train ----------------|-- val --|-- test --|
This prevents future interactions from leaking into model training and more closely resembles a real recommendation scenario:

Given what the user had interacted with in the past, can the system predict what they interact with later?

Validation data is used for model selection and ranker training.

The test set remains untouched until final evaluation.

Evaluation Metrics
The project focuses on ranking quality instead of rating prediction.

Primary metrics:

Precision@K
Measures how many of the recommended items are relevant.

Recall@K
Measures how many of the user's relevant held-out items were retrieved.

NDCG@K
Measures ranking quality while rewarding relevant items that appear closer to the top of the recommendation list.

Model Results
Approximate validation performance at K=10:

Model
Precision@10
Recall@10
NDCG@10
Popularity
0.04
0.02
0.02
Item-CF
0.06
0.08
0.08
ALS
0.06
0.09
0.09
Hybrid XGBoost Ranker
0.059
0.095
0.092
Final held-out test performance:

Metric
Score
Precision@10
0.0592
Recall@10
0.0906
NDCG@10
0.0846
The hybrid model primarily improved ranking quality, especially NDCG, by learning how to combine complementary recommendation signals.

Cold-Start Recommendations
CineMatch supports users who do not yet have historical interactions.

A new user can provide a small list of favorite movies:

{
  "liked_movie_ids": [260, 1196, 2571],
  "k": 10
}
The system then generates signals using:

Content similarity
Item-item collaborative filtering
Temporary ALS user factor
Popularity
These signals are passed through the same hybrid XGBoost ranker used for existing users.

This allows CineMatch to produce personalized recommendations without requiring a stored user account or historical interaction history.

Project Structure
cinematch_recommender/
│
├── api/
│   └── main.py
│
├── src/
│   └── recommender/
│       ├── data/
│       │   └── split.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── metrics.py
│       │
│       ├── models/
│       │   ├── popularity.py
│       │   ├── content.py
│       │   ├── item_cf.py
│       │   └── als.py
│       │
│       ├── ranking/
│       │   └── candidates.py
│       │
│       ├── pipeline/
│       │   └── train.py
│       │
│       └── serving/
│           └── engine.py
│
├── tests/
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
Training Pipeline
The complete training process can be run from the project root with:

python -m src.recommender.pipeline.train
The pipeline:

Load MovieLens data
        |
        v
Create relevance labels
        |
        v
Chronological split
        |
        v
Train base recommenders
        |
        v
Generate validation candidates
        |
        v
Train XGBoost ranker
        |
        v
Refit base models on train + validation
        |
        v
Save production artifacts
Saved artifacts include the trained base recommenders, XGBoost ranker, feature definitions, and movie metadata.

Model artifacts are intentionally excluded from Git because of their size.

Experiment Tracking
ALS tuning and hybrid ranking experiments were tracked using MLflow.

Experiments include:

model hyperparameters

Precision@10

Recall@10

NDCG@10

ranking configuration

candidate-generation settings

This makes model experiments reproducible and easier to compare.

Running the API Locally
Install dependencies:

pip install -r requirements.txt
Start FastAPI:

uvicorn api.main:app --reload
Interactive API documentation:

http://127.0.0.1:8000/docs
Health endpoint:

GET /health
API Endpoints
Existing User Recommendations
GET /recommendations/{user_id}?k=10
Example:

GET /recommendations/1?k=10
The endpoint uses the user's historical MovieLens interactions to generate hybrid recommendations.

New User / Preference-Based Recommendations
POST /recommendations/preferences
Example request:

{
  "liked_movie_ids": [260, 1196, 2571],
  "k": 10
}
The endpoint produces recommendations without requiring a previously known user ID.

Testing
API behavior is tested using pytest and FastAPI's TestClient.

Run tests with:

python -m pytest tests/test_api.py -v
Tests cover:

API health

known-user recommendations

preference-based recommendations

invalid recommendation counts

empty cold-start preferences

The API tests use FastAPI dependency injection to replace the expensive production recommendation engine with a deterministic fake engine.

This allows CI to validate routing, request validation, response contracts, and error handling without requiring large model artifacts.

The real recommender is validated separately through the training pipeline, inference engine, Docker execution, and model evaluation.

Continuous Integration
GitHub Actions automatically runs the API test suite on pushes and pull requests.

The CI workflow:

Git push / pull request
        |
        v
Fresh Ubuntu runner
        |
        v
Install Python
        |
        v
Install dependencies
        |
        v
Run pytest
        |
        v
Pass / Fail
This helps detect missing dependencies, broken imports, API regressions, and other issues that may not appear in a developer's local environment.

Docker
Build the API image:

docker build -t cinematch-api .
Run it:

docker run --rm -p 8000:8000 cinematch-api
The container packages the API, recommendation engine, runtime dependencies, and trained model artifacts into a reproducible environment.

Kubernetes
CineMatch can also be deployed to a local Kubernetes cluster using kind.

Create the cluster:

kind create cluster --name cinematch
Load the local Docker image:

kind load docker-image cinematch-api --name cinematch
Deploy:

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
Verify:

kubectl get pods
kubectl get services
Access the API locally:

kubectl port-forward service/cinematch-api 8000:80
The Kubernetes Deployment includes readiness and liveness probes using:

GET /health
Readiness determines whether a pod should receive traffic, while liveness allows Kubernetes to restart an unhealthy application.

Key Engineering Decisions
Chronological rather than random splitting
Recommendation systems must avoid learning from future user behavior.

Ranking rather than rating prediction
The product goal is deciding which movies should appear at the top of a recommendation feed, not predicting an exact star rating.

Multiple candidate generators
Different models capture different signals:

Popularity → global preference
Content → metadata similarity
Item-CF → behavioral item similarity
ALS → latent user-item preference
Learning-to-rank
XGBoost learns how much to trust each recommendation signal instead of relying on manually chosen weights.

Shared candidate-generation code
Training and serving reuse the same candidate feature-generation logic to reduce training-serving skew.

Dependency injection in FastAPI
Production uses the real recommendation engine while automated API tests can substitute a lightweight fake implementation.

Model artifacts outside Git
Large binary model files are kept outside source control while source code, configuration, tests, and deployment definitions remain versioned.

Docker + Kubernetes
Docker provides a reproducible runtime environment.

Kubernetes demonstrates container orchestration, health monitoring, service abstraction, and deployment management.

Technology Stack
Machine Learning

Python

pandas

NumPy

SciPy

scikit-learn

implicit ALS

XGBoost / LambdaMART

MLflow

Serving

FastAPI

Pydantic

Uvicorn

Testing

pytest

FastAPI TestClient

MLOps / Infrastructure

Git

GitHub Actions

Docker

Kubernetes

kind

What This Project Demonstrates
CineMatch was designed as more than a recommendation-model notebook.

It demonstrates an end-to-end ML engineering workflow:

data
→ experimentation
→ recommendation modeling
→ ranking
→ evaluation
→ reproducible training
→ artifact persistence
→ inference
→ API serving
→ automated testing
→ CI
→ containerization
→ orchestration
The resulting system supports both historical-user recommendation and cold-start onboarding while maintaining clear separation between model training, inference, API serving, testing, and infrastructure.

