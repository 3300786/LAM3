set -e  # 任意命令失败则终止整个脚本
echo "[Stage 1] Running synergy inference..."
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 0 &
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 1 &
CUDA_VISIBLE_DEVICES=2 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 2 &
CUDA_VISIBLE_DEVICES=3 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 3 &
wait
echo "[Stage 1] Completed."
echo "[Stage 2] Running Qwen/Llama judge..."

CUDA_VISIBLE_DEVICES=0 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_ide.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_ide.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/ide/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 0 &
CUDA_VISIBLE_DEVICES=1 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_ide.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_ide.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/ide/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 1 &
CUDA_VISIBLE_DEVICES=2 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_ide.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_ide.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/ide/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 2 &
CUDA_VISIBLE_DEVICES=3 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_ide.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_ide.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/ide/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 3 &
wait
echo "[Stage 2] Completed. All synergy metrics ready."
