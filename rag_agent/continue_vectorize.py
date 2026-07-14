"""兼容入口：复用统一、幂等且带 manifest 校验的索引流程。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_db import build_indexes


def main():
    result = build_indexes()
    print(f"完成：{len(result['chunks'])} chunks，向量库 {result['vector_store'].count()} 条")


if __name__ == "__main__":
    main()
