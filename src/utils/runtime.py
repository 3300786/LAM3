from dataclasses import dataclass
import torch, yaml, os, random, numpy as np

@dataclass
class GenCfg:
    max_new_tokens:int=256; temperature:float=0.2; top_p:float=0.9

def load_yaml(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_kwargs(runtime_cfg):
    kwargs={}
    if runtime_cfg.get("load_in_4bit", True):
        kwargs.update(dict(load_in_4bit=True, torch_dtype=torch.bfloat16 if runtime_cfg.get("precision","bf16")=="bf16" else torch.float16))
    else:
        kwargs.update(dict(torch_dtype=torch.bfloat16 if runtime_cfg.get("precision","bf16")=="bf16" else torch.float16))
    kwargs["device_map"]=runtime_cfg.get("device_map","auto")
    return kwargs
