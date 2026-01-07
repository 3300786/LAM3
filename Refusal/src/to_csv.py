import json
import csv
import argparse
import os
import sys


def main():
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="Convert JSONL file to CSV for Excel viewing.")
    parser.add_argument("input_file", help="The path to the input .jsonl file")
    args = parser.parse_args()

    input_path = args.input_file

    # 2. 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    # 3. 确定输出文件名 (将 .jsonl 替换为 .csv)
    # os.path.splitext 会分离文件名和后缀
    base_name, _ = os.path.splitext(input_path)
    output_path = base_name + ".csv"

    print(f"Processing: {input_path} -> {output_path} ...")

    try:
        data_list = []
        all_keys = set()

        # 4. 读取 JSONL 文件
        # 使用 utf-8 读取，防止读取源文件时乱码
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue  # 跳过空行
                try:
                    obj = json.loads(line)
                    data_list.append(obj)
                    # 收集所有的 key，以防不同行的字段不一致
                    all_keys.update(obj.keys())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipped invalid JSON at line {line_num}: {e}")

        if not data_list:
            print("Error: No valid data found in the input file.")
            return

        # 对 Key 进行排序，保证 CSV 列的顺序固定
        fieldnames = sorted(list(all_keys))

        # 5. 写入 CSV 文件
        # 关键点: encoding='utf-8-sig' 是为了让 Excel 正确识别中文
        # newline='' 是为了防止 Windows 下出现空行
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 写入数据 (DictWriter 会自动处理缺失的字段，填入空值)
            writer.writerows(data_list)

        print(f"Success! Saved to: {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()