"""可靠调用 DeepSeek JSON 模式的公共辅助函数。"""

import json
from typing import Any, Dict


def _decode_json(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise ValueError("模型返回了空 JSON content")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型 JSON 顶层必须是对象")
    return value


def request_json_object(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1200,
    attempts: int = 2,
) -> Dict[str, Any]:
    """关闭思考模式获取短 JSON；空响应或截断时自动重试一次。"""
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        retry_note = "\n请只输出完整 JSON 对象，不要输出 Markdown。" if attempt else ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt + retry_note}],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("模型 JSON 因 max_tokens 被截断")
            return _decode_json(getattr(choice.message, "content", "") or "")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"模型 JSON 请求连续失败 {max(1, attempts)} 次: {last_error}")
