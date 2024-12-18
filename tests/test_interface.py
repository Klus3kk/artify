import subprocess
import os

def test_cli_handler_success():
    command = [
        "python",
        "interface/CLIHandler.py",
        "--content",
        "images/content/sample_content.jpg",
        "--style_category",
        "impressionism",
        "--output",
        "images/output/test_output.jpg"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Styled image saved to:" in result.stdout

def test_cli_handler_invalid_content():
    command = [
        "python",
        "interface/CLIHandler.py",
        "--content",
        "invalid_path.jpg",
        "--style_category",
        "impressionism",
        "--output",
        "images/output/test_output.jpg"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Error" in result.stderr

def test_cli_handler_invalid_style():
    command = [
        "python",
        "interface/CLIHandler.py",
        "--content",
        "images/content/sample_content.jpg",
        "--style_category",
        "invalid_style",
        "--output",
        "images/output/test_output.jpg"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Error" in result.stderr
