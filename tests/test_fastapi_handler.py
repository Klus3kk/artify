from fastapi.testclient import TestClient
from api.FastAPIHandler import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Artify! Use /apply_style to stylize your images."
    }

def test_apply_style():
    with open("path/to/content_image.jpg", "rb") as content, open("path/to/style_image.jpg", "rb") as style:
        response = client.post(
            "/apply_style/",
            files={"content": ("content_image.jpg", content, "image/jpeg"),
                   "style": ("style_image.jpg", style, "image/jpeg")},
        )
        assert response.status_code == 200
        assert "output_path" in response.json()
