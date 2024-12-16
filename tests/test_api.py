from fastapi.testclient import TestClient
from api.FastAPIHandler import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Artify! Use /apply_style to stylize your images."}

def test_apply_style_endpoint():
    with open("images/content/sample_content.jpg", "rb") as content, open("images/style/impressionism/sample_style.jpg", "rb") as style:
        response = client.post(
            "/apply_style/",
            files={"content": ("sample_content.jpg", content, "image/jpeg")},
            data={"style_category": "impressionism"}
        )
        assert response.status_code == 200
        assert "output_path" in response.json(), "API should return the output path."
