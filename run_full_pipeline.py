#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 全流程自动化流水线
步骤：
1. 红队测试 (promptfoo redteam run)
2. 批量生成测试用例 (generate_cases.py)
3. 幻觉检测 (hallucination_check.py)
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 环境变量加载（可选，如果你使用 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装 python-dotenv，手动设置环境变量即可

# 配置
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_FILE = OUTPUT_DIR / f"pipeline_summary_{TIMESTAMP}.json"

def print_header(msg):
    print("\n" + "="*70)
    print(f"  {msg}")
    print("="*70 + "\n")

def run_step(step_name, command, env=None):
    """执行单个步骤，返回成功标志和输出信息"""
    print_header(f"▶ 开始执行: {step_name}")
    print(f"命令: {command}")
    try:
        # 使用 shell=True 以支持管道和复杂命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            env=env or os.environ,
            capture_output=False,  # 实时显示输出
            check=False
        )
        success = (result.returncode == 0)
        if success:
            print(f"✅ {step_name} 执行成功")
        else:
            print(f"❌ {step_name} 执行失败，返回码: {result.returncode}")
        return success, result.returncode
    except Exception as e:
        print(f"❌ {step_name} 执行异常: {e}")
        return False, -1

def main():
    start_time = datetime.now()
    print_header(f"🚀 LLM 全流程自动化流水线启动 - {start_time}")

    # 记录每一步的结果
    steps = []

    # 1. 红队测试
    success, code = run_step(
        "红队测试 (Red Team)",
        "npx promptfoo@latest redteam run --config promptfooconfig.yaml"
    )
    steps.append({"name": "redteam", "success": success, "returncode": code})

    # 2. 批量生成测试用例
    success, code = run_step(
        "批量生成测试用例",
        "python generate_cases.py"
    )
    steps.append({"name": "generate_cases", "success": success, "returncode": code})

    # 3. 幻觉检测
    success, code = run_step(
        "幻觉检测 (Faithfulness)",
        "python hallucination_check.py"
    )
    steps.append({"name": "hallucination_check", "success": success, "returncode": code})

    # 汇总结果
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    all_success = all(s["success"] for s in steps)

    summary = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "overall_success": all_success,
        "steps": steps
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_header("📊 流水线执行汇总")
    for s in steps:
        status = "✅ 成功" if s["success"] else "❌ 失败"
        print(f"  {s['name']}: {status} (返回码: {s['returncode']})")
    print(f"\n总耗时: {duration:.1f} 秒")
    print(f"汇总报告已保存: {SUMMARY_FILE}")

    if all_success:
        print("\n🎉 所有步骤均成功完成！")
        sys.exit(0)
    else:
        print("\n⚠️ 部分步骤执行失败，请检查日志。")
        sys.exit(1)

if __name__ == "__main__":
    main()