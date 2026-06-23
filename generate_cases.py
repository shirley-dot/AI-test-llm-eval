#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ======================== 配置区（一次性设置） ========================
API_KEY = os.environ.get("DASHSCOPE_API_KEY")
MODEL = os.environ.get("MODEL_NAME", "qwen-max")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = MODEL

# 公共背景（所有需求共享）
BACKGROUND = "【业务背景】当前平台的登录用户名与密码仅支持设定一组，无法让用户按照不同区域进行设定。"

# CSV 文件路径（需求点列表）
CSV_FILE = "requirements.csv"

# 输出文件
os.makedirs("outputs", exist_ok=True)
OUTPUT_JSON = "outputs/generated_cases.json"
OUTPUT_MD = "outputs/generated_cases.md"
OUTPUT_CSV = "outputs/generated_cases.csv"

# 请求延时（秒），避免 API 限流
REQUEST_DELAY = 1.5

# ======================== 最佳 Prompt 模板 ========================
PROMPT_TEMPLATE = """# Role
你是一位极度严谨、追求零缺陷的资深测试架构师，精通等价类划分和边界值分析。

# Context
我需要为以下功能模块设计全面的测试用例。
【功能模块描述及业务规则】：
{requirement}

# Task
请为该功能设计测试用例，**重点覆盖边界值场景和错误推测**。
1. 正向场景：简要覆盖核心流程。
2. 边界值场景：针对所有数值、长度、时间字段进行极限值测试（最小值、最大值、临界值）。
3. 错误推测：必须根据业务逻辑，列出开发最容易犯的 5 个编码错误并设计对应用例。

# Constraints
- 预期结果必须包含具体的"报错码"或"数据库字段变化"，不能只说"成功"或"失败"。
- 边界值必须使用具体的数字（例如：输入 0, 1, 15, 16, 999...）。

# Output Format
请以Markdown表格形式输出：
| 用例编号 | 测试维度(正向/边界/错误推测) | 用例标题 | 输入数据 | 预期结果(含具体值) | 优先级 |"""


# ======================== 1. 从 CSV 读取需求点 ========================
def load_requirements_from_csv(file_path):
    """读取 CSV 中的需求点，拼接公共背景，生成完整需求列表"""
    points = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig 兼容 Excel 保存的 CSV
            reader = csv.DictReader(f)
            for row in reader:
                # 适配列名：可能是 'point' 或 '需求点' 或 'content'
                point = row.get('point') or row.get('需求点') or row.get('content')
                if point:
                    points.append(point.strip())
    except FileNotFoundError:
        print(f"❌ 文件 {file_path} 不存在，请创建该文件。")
        print("   格式示例：")
        print('   point')
        print('   "自动展示Authority Management添加的Area..."')
        return []

    if not points:
        print("⚠️ CSV 文件为空或未找到 'point' 列，请检查格式。")
        return []

    # 拼接背景和每个需求点
    full_requirements = [f"{BACKGROUND}【需求点】{p}" for p in points]
    return full_requirements


# ======================== 2. 调用千问 API ========================
def generate_test_case(requirement):
    """调用 API 生成单个需求的测试用例"""
    prompt = PROMPT_TEMPLATE.format(requirement=requirement)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        output = data['choices'][0]['message']['content'].strip()
        return output, None
    except Exception as e:
        return None, str(e)


# ======================== 3. 保存结果 ========================
def save_results(results):
    """保存 JSON、Markdown、CSV 三种格式"""
    # JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON 已保存: {OUTPUT_JSON}")

    # Markdown
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(f"# 批量生成的测试用例\n\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for idx, item in enumerate(results, start=1):
            f.write(f"## 需求 {idx}: {item['requirement']}\n\n")
            f.write(f"{item['output']}\n\n")
            f.write("---\n\n")
    print(f"💾 Markdown 已保存: {OUTPUT_MD}")

    # CSV（Excel 友好）
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['需求编号', '需求内容', '生成的测试用例'])
        for idx, item in enumerate(results, start=1):
            clean_output = item['output'].replace('\n', '\\n').replace('"', '""')
            writer.writerow([idx, item['requirement'], clean_output])
    print(f"💾 CSV 已保存: {OUTPUT_CSV}")


# ======================== 主流程 ========================
def main():
    print("🚀 开始批量生成测试用例...")

    # 加载需求
    requirements = load_requirements_from_csv(CSV_FILE)
    if not requirements:
        return

    print(f"✅ 共读取到 {len(requirements)} 条需求")
    print(f"📌 公共背景: {BACKGROUND[:50]}...\n")

    # 逐条生成
    results = []
    for idx, req in enumerate(requirements, start=1):
        print(f"⏳ [{idx}/{len(requirements)}] 正在处理: {req[:40]}...")
        output, error = generate_test_case(req)
        if error:
            print(f"   ❌ 失败: {error}")
            results.append({"requirement": req, "output": f"生成失败: {error}", "success": False})
        else:
            print(f"   ✅ 成功 (长度: {len(output)} 字符)")
            results.append({"requirement": req, "output": output, "success": True})

        if idx < len(requirements):
            time.sleep(REQUEST_DELAY)

    # 统计
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n📊 生成完成: 成功 {success_count}/{len(results)}")

    # 保存结果
    save_results(results)
    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
