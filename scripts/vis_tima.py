import json
import sys
from pathlib import Path


def sort_jsonl_by_id(input_path, output_path=None):
    """
    读取 JSONL 文件，按 'id' 字段排序，并写入输出文件。

    :param input_path: 输入的 .jsonl 文件路径（str 或 Path）
    :param output_path: 输出文件路径；若为 None，则在原文件名后加 .sorted.jsonl
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"错误：文件 {input_path} 不存在。", file=sys.stderr)
        return

    # 读取所有行并解析 JSON
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                record = json.loads(line)
                if 'id' not in record:
                    print(f"警告：第 {line_num} 行缺少 'id' 字段，跳过。", file=sys.stderr)
                    continue
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"错误：第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                continue

    # 按 id 排序（支持字符串或数字）
    try:
        records.sort(key=lambda x: x['id'])
    except TypeError as e:
        print(f"排序时发生类型错误（可能 id 类型不一致）: {e}", file=sys.stderr)
        # 可选：强制转为字符串排序
        records.sort(key=lambda x: str(x['id']))

    # 确定输出路径
    if output_path is None:
        output_path = input_path.with_suffix('.sorted.jsonl')

    # 写入排序后的结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"排序完成！结果已保存至: {output_path}")

def show_output_sorted_by_id(jsonl_path):
    """
    读取 JSONL 文件，按 'id' 排序，并逐行打印 'output' 字段内容，
    每次打印后暂停，等待用户输入（按回车继续）。
    """
    path = Path(jsonl_path)
    if not path.exists():
        print(f"错误：文件 {path} 不存在。", file=sys.stderr)
        return

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'id' not in obj:
                    print(f"警告：第 {line_num} 行缺少 'id'，跳过。", file=sys.stderr)
                    continue
                if 'output' not in obj:
                    print(f"警告：第 {line_num} 行缺少 'output'，跳过。", file=sys.stderr)
                    continue
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"错误：第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
                continue

    # 尝试按 id 排序（支持数字/字符串）
    try:
        records.sort(key=lambda x: x['id'])
    except TypeError:
        # 如果类型不一致，转为字符串排序
        records.sort(key=lambda x: str(x['id']))

    print("按 'id' 排序后的 'output' 内容（按回车继续，Ctrl+C 退出）：\n")
    for i, record in enumerate(records, start=1):
        output_val = record.get('output', '<missing>')
        print(f"[{i}] ID: {record['id']}")
        print("Prompt:", record['prompt'], record['image'])
        print("Output:")
        print(output_val)
        print("-" * 50)
        print(record['judge_refusal'], record['judge_attack_success'])
        print("-" * 50)
        try:
            input("👉 按回车继续...")
        except KeyboardInterrupt:
            print("\n用户中断。")
            break
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python sort_jsonl.py <input_file.jsonl> [output_file.jsonl]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    sort_jsonl_by_id(input_file, output_file)
    show_output_sorted_by_id(output_file)