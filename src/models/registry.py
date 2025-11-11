from src.models.llava15 import Llava15Wrapper
from src.models.idefics2 import Idefics2Wrapper

def build_model(name:str, models_cfg:dict, runtime_cfg:dict):
    if name == "llava15_7b":
        return Llava15Wrapper(models_cfg[name]["repo_id"], runtime_cfg)
    if name == "idefics2_8b":
        return Idefics2Wrapper(models_cfg[name]["repo_id"], runtime_cfg)
    raise ValueError(f"Unknown model: {name}")
