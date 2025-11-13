# scripts/run_attack.py
import argparse
from src.pipelines.attack_runner import run_attack_experiment
from src.attacks.prompt_injection import SimplePromptInjection
from src.data.datasets import load_mini_bench  # 你自己实现

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="idefics2_8b")
    parser.add_argument("--exp_name", type=str, default="prompt_injection_debug")
    args = parser.parse_args()

    # runtime / device 配置可以从 configs 里读
    runtime_cfg = {
        "device": "cuda:0",
        "dtype": "bf16",
        # ...
    }

    samples = load_mini_bench("data/mini_bench/config.json")
    attack = SimplePromptInjection()
    run_attack_experiment(
        model_name=args.model,
        runtime_cfg=runtime_cfg,
        attack=attack,
        samples=samples,
        exp_name=args.exp_name
    )

if __name__ == "__main__":
    main()
