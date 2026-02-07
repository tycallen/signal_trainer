from .base import BaseModel
from .cnn import CNNModel
from .lgbm import LGBMModel

MODELS = {
    "lgbm": LGBMModel,
    "cnn": CNNModel,
}


def create_model(model_config) -> BaseModel:
    cls = MODELS.get(model_config.type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_config.type}")
    return cls(params=model_config.params)
