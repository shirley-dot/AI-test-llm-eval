#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import openai
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
MODEL = os.environ.get("MODEL_NAME", "qwen-max")
# ======================== 1. 自定义通义千问模型类 ========================
class Qwen(DeepEvalBaseLLM):
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or os.environ.get("MODEL_NAME", "qwen-max")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或在初始化时传入 api_key")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        # 异步方法，直接调用同步方法（或可改用异步客户端）
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# ======================== 2. 读取 generated_cases.csv ========================
def load_generated_cases(csv_path: str):
    """
    读取 generated_cases.csv，返回列表，每个元素为 (需求内容, 生成的测试用例)
    适配常见的列名：需求内容 / requirement / 需求  / 生成的测试用例 / output / 实际输出
    """
    cases = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # 打印列名，方便调试
        print("📋 CSV 列名:", reader.fieldnames)
        for row in reader:
            # 智能匹配需求列
            req = (row.get('需求内容') or row.get('requirement') or
                   row.get('需求') or row.get('Requirement'))
            # 智能匹配输出列
            out = (row.get('生成的测试用例') or row.get('output') or
                   row.get('实际输出') or row.get('Actual Output'))
            if req and out:
                cases.append((req.strip(), out.strip()))
    return cases


# ======================== 3. 批量评估 ========================
def run_faithfulness_eval(cases, threshold=0.7, model=None):
    if model is None:
        model = Qwen()

    results = []
    total = len(cases)

    for idx, (req, output) in enumerate(cases, start=1):
        print(f"🔍 [{idx}/{total}] 正在评估: {req[:40]}...")

        test_case = LLMTestCase(
            input="请根据需求生成测试用例。",
            actual_output=output,
            retrieval_context=[req]
        )

        metric = FaithfulnessMetric(
            threshold=threshold,
            include_reason=True,
            model=model
        )

        # 初始化默认值
        score = 0.0
        reason = "未知错误"
        passed = False

        try:
            # 运行评估（如果断言失败，会抛出 AssertionError）
            assert_test(test_case, [metric])
            # 如果通过，获取分数（可能为 None，但通常有值）
            score = metric.score if metric.score is not None else 1.0
            reason = metric.reason or "通过"
            passed = True
            print(f"   ✅ 通过 (得分: {score:.2f})")
        except AssertionError:
            # 断言失败，获取分数（可能为 None）
            score = metric.score if metric.score is not None else 0.0
            reason = metric.reason or "未通过断言"
            passed = False
            print(f"   ❌ 失败 (得分: {score:.2f})")
            if reason:
                print(f"      理由: {reason[:200]}...")
        except Exception as e:
            # 捕获其他所有异常（如网络错误、API 限流等）
            print(f"   ⚠️ 评估异常: {e}")
            score = 0.0
            reason = str(e)
            passed = False

        results.append({
            "index": idx,
            "requirement": req,
            "score": score,
            "passed": passed,
            "reason": reason
        })

    return results


# ======================== 4. 主函数 ========================
def main():
    # 配置阈值
    THRESHOLD = 0.7

    # CSV 文件路径（请根据实际情况修改）
    csv_file = "generated_cases.csv"

    # 读取数据
    try:
        cases = load_generated_cases(csv_file)
    except FileNotFoundError:
        print(f"❌ 文件 {csv_file} 不存在，请确认路径。")
        return

    if not cases:
        print("❌ 未读取到任何用例，请检查 CSV 列名是否匹配。")
        return

    print(f"📊 共读取到 {len(cases)} 条测试用例，开始评估幻觉风险...\n")

    # 运行评估（自动从环境变量读取 DASHSCOPE_API_KEY）
    results = run_faithfulness_eval(cases, threshold=THRESHOLD)

    # 汇总统计
    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    print("\n" + "="*50)
    print("📈 评估汇总")
    print(f"   总用例数: {len(results)}")
    print(f"   通过数: {passed_count} ({passed_count/len(results)*100:.1f}%)")
    print(f"   平均得分: {avg_score:.2f}")
    print("="*50)

    # 保存详细结果到 CSV
    output_file = "faithfulness_results.csv"
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["用例编号", "需求内容", "得分", "是否通过", "评估理由"])
        for r in results:
            writer.writerow([r["index"], r["requirement"], f"{r['score']:.2f}", r["passed"], r["reason"]])
    print(f"💾 详细结果已保存至 {output_file}")


if __name__ == "__main__":
    main()