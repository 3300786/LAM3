import pandas as pd
import matplotlib.pyplot as plt

data = [
    ["baseline_easy", 30, 0.00, 3.33, 0, 1, 29],
    ["baseline_hard", 80, 1.25, 3.75, 1, 3, 76],
    ["both_harm", 500, 34.60, 62.20, 173, 311, 16],
    ["multi_image_distract", 120, 40.00, 10.83, 48, 13, 59],
    ["multi_image_ocr", 120, 3.33, 60.00, 4, 72, 44],
    ["multi_image_reinforce", 40, 40.00, 55.00, 16, 22, 2],
    ["ocr", 20, 15.00, 85.00, 3, 17, 0],
    ["semantic", 100, 40.00, 30.00, 40, 30, 30],
    ["text", 500, 11.00, 88.20, 55, 441, 4],
    ["text_with_image", 500, 14.80, 83.20, 74, 416, 10],
    ["text_with_ocr", 500, 11.40, 88.00, 57, 440, 3],
]

df = pd.DataFrame(
    data,
    columns=["Group Name", "Total", "ASR (%)", "RR (%)", "Jailbroken", "Refused", "Safe"],
)

# sort by ASR ascending (left->right)
df = df.sort_values("ASR (%)", ascending=True).reset_index(drop=True)

# normalized percentages from counts (robust to rounding)
df["Jailbroken (%)"] = df["Jailbroken"] / df["Total"] * 100.0
df["Refused (%)"] = df["Refused"] / df["Total"] * 100.0
df["Safe (%)"] = df["Safe"] / df["Total"] * 100.0

x = df["Group Name"].tolist()

plt.figure(figsize=(16, 7))
bottom = [0.0] * len(df)

plt.bar(x, df["Jailbroken (%)"], bottom=bottom, label="Jailbroken (%)")
bottom = (df["Jailbroken (%)"]).tolist()

plt.bar(x, df["Refused (%)"], bottom=bottom, label="Refused (%)")
bottom = (df["Jailbroken (%)"] + df["Refused (%)"]).tolist()

plt.bar(x, df["Safe (%)"], bottom=bottom, label="Safe (%)")

plt.xticks(rotation=35, ha="right")
plt.ylabel("Percentage of samples (%)")
plt.title("Outcome Distribution by Group (Normalized, Sorted by ASR)")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

out_path = "/mnt/data/outcome_distribution_sorted_by_asr.png"
plt.savefig(out_path, dpi=200)
plt.close()

out_path
