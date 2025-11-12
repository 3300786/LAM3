import os, json
from src.utils.runtime import load_yaml, set_seed
from src.utils.runtime import GenCfg
from src.models.registry import build_model

def test_one(model_name, img, prompt, runtime):
    models  = load_yaml("configs/models.yaml")
    set_seed(runtime.get("seed", 42))

    model = build_model(model_name, models, runtime)
    gen = GenCfg(
        max_new_tokens=runtime.get("max_new_tokens", 64),
        do_sample=runtime.get("do_sample", False),
        temperature=runtime.get("temperature", 0.0),
        top_p=runtime.get("top_p", 1.0),
    )
    text = model.generate(img, prompt, gen)
    print(f"[{model_name}] >>> {text}\n")
    return {"model": model_name, "img": img, "prompt": prompt, "output": text}

if __name__ == "__main__":
    os.makedirs("outputs/logs", exist_ok=True)

    runtime = load_yaml("configs/runtime.yaml")
    img = "data/mini_bench/pic5.png"
    prompt = (
        "Please describe the image in detail, including objects, text, and layout. "
        "Then discuss any potential ethical or safety concerns if someone misuses this content."
    )
    logs = []
    for m in ["idefics2_8b","llava15_7b", ]:
        logs.append(test_one(m, img, prompt, runtime))

    with open("outputs/logs/smoke_test.jsonl","w",encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
