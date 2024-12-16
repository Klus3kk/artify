import subprocess

def test_cli_handler():
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
    assert result.returncode == 0, "CLI should execute without errors."
    assert "Styled image saved to:" in result.stdout, "CLI should output success message."
