set -e

echo "[Stage 1] Preparing Queries ..."
python -m scripts.run_carrier_equiv_qwen --exp_cfg configs/exp_carrier_equiv.yaml   --models_cfg configs/models.yaml --runtime_cfg configs/runtime.yaml   --stage gen

echo "[Stage 2] Victim model Inferring ..."


CUDA_VISIBLE_DEVICES=0 python -m scripts.run_carrier_equiv_qwen --exp_cfg configs/exp_carrier_equiv.yaml \
  --models_cfg configs/models.yaml --runtime_cfg configs/runtime.yaml --model_name qwen3_vl_8b \
  --stage infer --num_shards 3 --shard_idx 0 --resume &

CUDA_VISIBLE_DEVICES=1 python -m scripts.run_carrier_equiv_qwen --exp_cfg configs/exp_carrier_equiv.yaml \
  --models_cfg configs/models.yaml --runtime_cfg configs/runtime.yaml --model_name qwen3_vl_8b \
  --stage infer --num_shards 3 --shard_idx 1 --resume &

CUDA_VISIBLE_DEVICES=2 python -m scripts.run_carrier_equiv_qwen --exp_cfg configs/exp_carrier_equiv.yaml \
  --models_cfg configs/models.yaml --runtime_cfg configs/runtime.yaml --model_name qwen3_vl_8b \
  --stage infer --num_shards 3 --shard_idx 2 --resume &

wait

echo "[Stage 3] Merge Inference results ..."

python -m scripts.merge_jsonl_shards \
  --inputs outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.shard0-of3.jsonl \
          outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.shard1-of3.jsonl \
          outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.shard2-of3.jsonl \
  --out outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.merged.jsonl

echo "[Stage 4] Judge model Inferring ..."

CUDA_VISIBLE_DEVICES=0 python -m src.metrics.eval_carrier_equiv_qwen_asr \
  --raw_in outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.merged.jsonl \
  --judged_out outputs/logs/graded/carrier_equiv/qwen3_vl_8b_judged.jsonl \
  --metrics_dir outputs/metrics/carrier_equiv/qwen3_vl_8b \
  --judge_model_tag llama32_11b \
  --num_shards 3 --shard_idx 0 &

CUDA_VISIBLE_DEVICES=1 python -m src.metrics.eval_carrier_equiv_qwen_asr \
  --raw_in outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.merged.jsonl \
  --judged_out outputs/logs/graded/carrier_equiv/qwen3_vl_8b_judged.jsonl \
  --metrics_dir outputs/metrics/carrier_equiv/qwen3_vl_8b \
  --judge_model_tag llama32_11b \
  --num_shards 3 --shard_idx 1 &

CUDA_VISIBLE_DEVICES=2 python -m src.metrics.eval_carrier_equiv_qwen_asr \
  --raw_in outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.merged.jsonl \
  --judged_out outputs/logs/graded/carrier_equiv/qwen3_vl_8b_judged.jsonl \
  --metrics_dir outputs/metrics/carrier_equiv/qwen3_vl_8b \
  --judge_model_tag llama32_11b \
  --num_shards 3 --shard_idx 2 &

wait

echo "[Stage 5] Summarize Judge results ..."

python -m src.metrics.eval_carrier_equiv_qwen_asr \
  --raw_in outputs/logs/ungraded/carrier_equiv/qwen3_vl_8b.merged.jsonl \
  --judged_out outputs/logs/graded/carrier_equiv/qwen3_vl_8b_judged.jsonl \
  --metrics_dir outputs/metrics/carrier_equiv/qwen3_vl_8b \
  --skip_judge