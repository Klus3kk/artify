from core.StyleTransferModel import StyleTransferModel

def test_model_initialization():
    model = StyleTransferModel()
    assert model.model is not None
