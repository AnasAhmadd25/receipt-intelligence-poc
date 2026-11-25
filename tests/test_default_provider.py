from app.main import ExtractionInput

def test_default_pipeline_uses_nvidia_without_ollama():
    assert ExtractionInput(image_url='https://example.org/receipt.png').provider == 'ocr_nvidia'
