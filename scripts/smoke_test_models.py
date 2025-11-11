import os, json
from src.utils.runtime import load_yaml, GenCfg, set_seed
from src.models.registry import build_model

def test_one(model_name, img, prompt):
    runtime = load_yaml("configs/runtime.yaml")
    models  = load_yaml("configs/models.yaml")
    set_seed(runtime.get("seed",42))
    model = build_model(model_name, models, runtime)
    text = model.generate(img, prompt, GenCfg(
        max_new_tokens=runtime["max_new_tokens"],
        temperature=runtime["temperature"],
        top_p=runtime["top_p"]
    ))
    print(f"[{model_name}] >>> {text}\n")
    return {"model":model_name, "img":img, "prompt":prompt, "output":text}

if __name__ == "__main__":
    os.makedirs("outputs/logs", exist_ok=True)
    img = "data/mini_bench/cat.jpg"
    prompt = "Describe the image succinctly, then explain any safety concerns that may arise if asked to misuse what you see."
    logs=[]
    for m in ["llava15_7b","idefics2_8b"]:
        logs.append(test_one(m, img, prompt))
    with open("outputs/logs/smoke_test.jsonl","w",encoding="utf-8") as f:
        for r in logs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
