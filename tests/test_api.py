from fastapi.testclient import TestClient
from api.FastAPIHandler import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Artify! Use /apply_style to stylize your images."}

def test_apply_style_endpoint_success():
    with open("images/content/sample_content.jpg", "rb") as content:
        response = client.post(
            "/apply_style/",
            files={"content": ("sample_content.jpg", content, "image/jpeg")},
            data={"style_category": "impressionism"}
        )
        assert response.status_code == 200
        assert "output_path" in response.json()
        assert response.json()["message"] == "Style applied successfully!"

def test_apply_style_endpoint_invalid_style():
    with open("images/content/sample_content.jpg", "rb") as content:
        response = client.post(
            "/apply_style/",
            files={"content": ("sample_content.jpg", content, "image/jpeg")},
            data={"style_category": "invalid_style"}
        )
        assert response.status_code == 400
        assert "error" in response.json()

def test_apply_style_endpoint_missing_content():
    response = client.post(
        "/apply_style/",
        data={"style_category": "impressionism"}
    )
    assert response.status_code == 422
