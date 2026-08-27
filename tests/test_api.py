from fastapi.testclient import TestClient
from api.main import app, get_engine 

class FakeRecommendationEngine:
    def recommend(self, user_id, k=10, candidate_k = 100):
        return [
            {
                "movie_id":i, 
                "title": f"Movie {i}",
                "genres": "Test",
                "ranker_score": 1.0
            }
            for i in range(1,k+1)
        ]
    def recommend_from_items(self, liked_movie_ids, k=10, candidate_k = 100):
        return [
            {
                "movie_id": i + 100, 
                "title": f"Movie {i+100}",
                "genres": "Test",
                "ranker_score": 1.0
            }
            for i in range(1,k+1)
        ]
def override_get_engine():
    return FakeRecommendationEngine()

app.dependency_overrides[get_engine] = override_get_engine

client = TestClient(app) #make http like requests to fastapi without starting uvicorn 

def test_health():
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}

def test_known_user_recommendations():
    response = client.get('/recommendations/1?k=10')

    assert response.status_code == 200
    data = response.json() #the json response from the .get call 

    assert data['user_id'] == 1
    assert data['k'] == 10
    assert len(data['recommendations']) == 10

def test_preference_recommendations():
    response = client.post('/recommendations/preferences', json={'liked_movie_ids':[260,1196,2571], 'k': 10},)
    assert response.status_code == 200

    data = response.json()
    assert data['liked_movie_ids'] == [260, 1196, 2571]
    assert data['k'] == 10
    assert len(data['recommendations']) == 10

def test_preferences_requires_liked_movies(): 
    response = client.post('/recommendations/preferences', json={'liked_movie_ids':[], 'k': 10},)
    assert response.status_code == 400
    data = response.json()

    assert data['detail'] == 'At least one liked movie is required'

def test_invalid_k(): 
    response = client.get('/recommendations/1?k=0')
    assert response.status_code == 400
    data = response.json()

    assert data['detail'] == 'k must be between 1 and 100'