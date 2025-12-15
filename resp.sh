CUDA_VISIBLE_DEVICES=0 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 0 &
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 1 &
CUDA_VISIBLE_DEVICES=2 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 2 &
CUDA_VISIBLE_DEVICES=3 python -m scripts.run_synergy_jbv28k \
  --cfg configs/synergy_jbv28k.yaml --num_shards 4 --shard_idx 3 &
wait