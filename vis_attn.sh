layers="27"
image="data/mini_bench/icon.png"
text="Which icon in image is most AI-related?"

for l in $layers; do
    python -m scripts.min_attn_heatmap \
        --image "$image" \
        --text "$text" \
        --layer $l \
        --agg max \
        --alpha 0.7 --try_generate
done