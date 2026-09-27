"""
快速测试 vllm_service2.py 的返回是否正确。
用法：先启动服务，再运行此脚本。
  python utils/vllm_service2.py --model Qwen/Qwen3-8B --port 7780
  python utils/test_vllm_service.py
"""
import requests

URL = "http://127.0.0.1:7780/generate"
N = 4  # 每个 prompt 采样次数


def test(prompts, n):
    payload = {
        "requests": [
            {"prompt": p, "max_tokens": 128, "temperature": 0.9, "n": n}
            for p in prompts
        ]
    }
    resp = requests.post(URL, json=payload, timeout=120)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    results = resp.json()["results"]

    batch_size = len(prompts)
    expected = batch_size * n
    assert len(results) == expected, (
        f"数量错误: 期望 {batch_size} prompts × {n} = {expected} 条, 实际返回 {len(results)} 条"
    )
    print(f"[OK] 数量正确: {batch_size} prompts × {n} = {len(results)} 条\n")

    # 检查同一 prompt 的 n 条 response 是否不同
    for i, prompt in enumerate(prompts):
        group = [results[i * n + j]["text"] for j in range(n)]
        unique = set(group)
        print(f"Prompt {i}: {repr(prompt[:40])}")
        for j, text in enumerate(group):
            print(f"  [{j}] {repr(text[:80])}")
        if len(unique) == 1:
            print(f"  [WARN] {n} 条 response 完全相同（temperature 可能太低，或模型确定性太强）")
        else:
            print(f"  [OK] {len(unique)}/{n} 条不同")
        print()


INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then "
    "provide the final answer. The reasoning process MUST BE enclosed within <think> "
    "</think> tags. The final answer MUST BE put in \\boxed{}."
)

def make_prompt(problem: str) -> str:
    """模拟 skip_special_tokens=True 解码后的 prompt 格式（chat template 去掉特殊token后的文本）"""
    return (
        f"system\n\nuser\n\n{problem} {INSTRUCTION}\n\nassistant\n\n"
    )


if __name__ == "__main__":
    # 健康检查
    health = requests.get("http://127.0.0.1:7780/health", timeout=5)
    assert health.status_code == 200, "服务未启动"
    print("[OK] 服务健康\n")

    prompts = [
        make_prompt("What is 1 + 1?"),
        make_prompt("Solve: x^2 - 5x + 6 = 0."),
    ]
    test(prompts, n=N)

    # 额外检查：response 里是否包含 \boxed{}
    payload = {"requests": [{"prompt": prompts[0], "max_tokens": 512, "temperature": 0.7, "n": 1}]}
    results = requests.post(URL, json=payload, timeout=120).json()["results"]
    text = results[0]["text"]
    has_boxed = "\\boxed{" in text
    print(f"\\boxed{{}} 检查: {'[OK] 有' if has_boxed else '[WARN] 没有'}")
    print(f"Response 预览: {repr(text[:200])}")
