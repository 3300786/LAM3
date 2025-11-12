from dataclasses import dataclass
@dataclass
class GenCfg:
    max_new_tokens:int=256
    do_sample:bool=True
    temperature:float=0.8
    top_p:float=0.95
