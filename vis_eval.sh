python -m src.metrics.eval_synergy_qwen_asr \
  --raw_in outputs/logs/ungraded/max/jbv28k_lma.jsonl \
  --judged_out outputs/logs/graded/max/jbv28k_judge_lma.fixed.jsonl \
  --metrics_dir outputs/metrics/synergy/jbv28k/lma/  \
  --cfg configs/synergy_jbv28k.yaml\
  --skip_judge \
  --judge_model_tag llama32_11b