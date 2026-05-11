#!/usr/bin/env python3
"""
合并生成的数据分片文件,并可选地拆分为训练集和测试集。

用法:
    # 合并所有shard文件
    python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct

    # 合并并随机选择10000条作为测试集
    python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct --eval-size 10000

    # 指定输出目录
    python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct --output-dir ./cache/train_dataset/llama-3.1-8b-instruct --eval-size 5000

    # 限制训练集大小为50000条，测试集5000条
    python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct --train-size 50000 --eval-size 5000

    # 使用自定义随机种子
    python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct --train-size 100000 --eval-size 10000 --seed 123
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict


def load_shard_files(input_dir: str) -> List[Dict]:
    """加载所有shard文件并返回数据列表"""
    input_path = Path(input_dir)
    all_data = []

    # 找到所有shard文件(忽略error文件)
    shard_files = sorted([f for f in input_path.glob("shard_*.jsonl")])

    if not shard_files:
        raise ValueError(f"在 {input_dir} 中未找到任何shard文件")

    print(f"找到 {len(shard_files)} 个shard文件")

    # 逐个读取shard文件
    for shard_file in shard_files:
        print(f"正在读取: {shard_file.name}")
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"警告: 跳过无效的JSON行: {e}")

    print(f"总共加载了 {len(all_data)} 条数据")
    return all_data


def split_and_save(data: List[Dict], output_dir: str, eval_size: int = 0, train_size: int = 0):
    """拆分数据为训练集和测试集,并保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 随机打乱数据
    random.shuffle(data)

    # 如果需要拆分测试集和训练集
    if eval_size > 0 or train_size > 0:
        total_needed = (train_size if train_size > 0 else 0) + eval_size
        if total_needed > len(data):
            raise ValueError(f"需要的总数据量 ({total_needed}) 超过了可用数据量 ({len(data)})")

        # 先分配测试集
        if eval_size > 0:
            eval_data = data[:eval_size]
            remaining_data = data[eval_size:]
        else:
            eval_data = []
            remaining_data = data

        # 再分配训练集
        if train_size > 0:
            train_data = remaining_data[:train_size]
        else:
            train_data = remaining_data

        print(f"训练集: {len(train_data)} 条")
        if eval_size > 0:
            print(f"测试集: {len(eval_data)} 条")

        # 保存测试集
        if eval_size > 0:
            eval_file = output_path / "eval.jsonl"
            with open(eval_file, 'w', encoding='utf-8') as f:
                for item in eval_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"测试集已保存到: {eval_file}")
    else:
        train_data = data
        print(f"训练集: {len(train_data)} 条 (无测试集)")

    # 保存训练集
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"训练集已保存到: {train_file}")


def main():
    parser = argparse.ArgumentParser(
        description="合并数据分片并可选地拆分为训练集和测试集"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="包含shard文件的输入目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认与输入目录相同)"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=0,
        help="从数据中随机选择作为测试集的样本数量 (默认: 0, 不创建测试集)"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="训练集的最大样本数量 (默认: 0, 使用所有剩余数据作为训练集)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子,用于可复现的数据拆分 (默认: 42)"
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 如果未指定输出目录,使用输入目录
    output_dir = args.output_dir if args.output_dir else args.input_dir

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练集大小: {args.train_size if args.train_size > 0 else '全部'}")
    print(f"测试集大小: {args.eval_size if args.eval_size > 0 else '无'}")
    print(f"随机种子: {args.seed}")
    print("-" * 50)

    # 加载所有shard文件
    all_data = load_shard_files(args.input_dir)

    # 拆分并保存
    split_and_save(all_data, output_dir, args.eval_size, args.train_size)

    print("-" * 50)
    print("完成!")


if __name__ == "__main__":
    main()
