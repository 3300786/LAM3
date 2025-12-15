CUDA_VISIBLE_DEVICES=0 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_lva.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_lva.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/lva/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 2 --shard_idx 0 &
CUDA_VISIBLE_DEVICES=1 python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_lva.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_lva.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/lva/  \
  --cfg configs/synergy_jbv28k.yaml\
  --judge_model_tag llama32_11b --num_shards 2 --shard_idx 1 &
#CUDA_VISIBLE_DEVICES=2 python -m src.metrics.eval_synergy_qwen_asr \
#  --raw_in outputs/logs/ungraded/max/jbv28k_lva.jsonl \
#  --judged_out outputs/logs/graded/max/jbv28k_judge_lva.jsonl \
#  --metrics_dir outputs/metrics/synergy/jbv28k/lva/  \
#  --cfg configs/synergy_jbv28k.yaml\
#  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 2 &
#CUDA_VISIBLE_DEVICES=3 python -m src.metrics.eval_synergy_qwen_asr \
#  --raw_in outputs/logs/ungraded/max/jbv28k_lva.jsonl \
#  --judged_out outputs/logs/graded/max/jbv28k_judge_lva.jsonl \
#  --metrics_dir outputs/metrics/synergy/jbv28k/lva/  \
#  --cfg configs/synergy_jbv28k.yaml\
#  --judge_model_tag llama32_11b --num_shards 4 --shard_idx 3 &
wait