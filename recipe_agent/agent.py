"""DeepSeek 工具调用循环及本地校验。"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import AgentTrace, RecipeRecord
from .tools import normalize_quantity, parse_markdown_sections, validate_recipe_payload


@dataclass
class AgentRunResult:
    recipe: Optional[RecipeRecord]
    traces: List[AgentTrace] = field(default_factory=list)
    error: str = ""
    retries: int = 0
    duration_ms: int = 0


class RecipeParsingAgent:
    """只执行白名单工具、有限轮次运行的菜谱解析 Agent。"""

    def __init__(
        self,
        client: Any,
        model: str = "deepseek-v4-flash",
        max_rounds: int = 6,
        max_retries: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.max_rounds = max(1, max_rounds)
        self.max_retries = max(1, max_retries)
        self.sleeper = sleeper

    @staticmethod
    def _recipe_tool_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"recipe": RecipeRecord.model_json_schema()},
            "required": ["recipe"],
            "additionalProperties": False,
        }

    @classmethod
    def tool_definitions(cls) -> List[Dict[str, Any]]:
        recipe_schema = cls._recipe_tool_schema()
        return [
            {
                "type": "function",
                "function": {
                    "name": "parse_markdown_sections",
                    "description": "拆分当前菜谱 Markdown。解析任务必须先调用此工具。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "normalize_quantity",
                    "description": "规范化一条食材用量。每个食材的用量都必须调用一次。",
                    "parameters": {
                        "type": "object",
                        "properties": {"raw": {"type": "string"}},
                        "required": ["raw"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_recipe",
                    "description": "校验候选菜谱结构和原文证据，返回需要修正的问题。",
                    "parameters": recipe_schema,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_recipe",
                    "description": "提交最终菜谱。只有此工具校验通过，任务才会成功。",
                    "parameters": recipe_schema,
                },
            },
        ]

    def _create_completion(self, messages: List[Dict[str, Any]]) -> tuple[Any, int, int]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions(),
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=4096,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return response, int((time.perf_counter() - started) * 1000), attempt
            except Exception as exc:  # API SDK 抛出的异常类型随版本变化
                last_error = exc
                if attempt + 1 < self.max_retries:
                    self.sleeper(float(2 ** attempt))
        raise RuntimeError(f"DeepSeek API 连续失败 {self.max_retries} 次: {last_error}")

    @staticmethod
    def _tool_call_dict(tool_call: Any) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def parse(self, markdown: str, source_path: str) -> AgentRunResult:
        started_at = time.perf_counter()
        traces: List[AgentTrace] = []
        retries = 0
        sections_called = False
        normalized_calls = 0
        submitted: Optional[RecipeRecord] = None

        system_prompt = (
            "你是中文菜谱结构化 Agent。你只能依据用户给出的 Markdown 工作，禁止补充原文没有的事实。"
            "必须先调用 parse_markdown_sections，再为每个食材调用 normalize_quantity。"
            "将工具返回的 Quantity 原样放入对应食材。每个食材和步骤的 evidence 必须是源文档中的连续原文。"
            "先调用 validate_recipe；修正全部错误后调用 submit_recipe。"
            "适量等模糊表达不得改成数字，按份数公式不得擅自展开，勺等计数单位不得换算为克。"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"源文件：{source_path}\n\n请解析以下 Markdown：\n\n{markdown}",
            },
        ]

        for round_number in range(1, self.max_rounds + 1):
            try:
                response, latency_ms, retry_count = self._create_completion(messages)
                retries += retry_count
                if retry_count:
                    traces.append(AgentTrace(
                        source_path=source_path,
                        round=round_number,
                        event="api_retry",
                        result_summary=str(retry_count),
                    ))
            except RuntimeError as exc:
                retries += self.max_retries - 1
                traces.append(AgentTrace(
                    source_path=source_path,
                    round=round_number,
                    event="api_error",
                    arguments_summary=str(self.max_retries - 1),
                    result_summary=str(exc)[:500],
                ))
                return AgentRunResult(
                    recipe=None,
                    traces=traces,
                    error=str(exc),
                    retries=retries,
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )

            prompt_tokens, completion_tokens = self._usage(response)
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                traces.append(AgentTrace(
                    source_path=source_path,
                    round=round_number,
                    event="missing_tool_call",
                    result_summary=(getattr(message, "content", "") or "")[:500],
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ))
                messages.append({"role": "assistant", "content": getattr(message, "content", "") or ""})
                messages.append({
                    "role": "user",
                    "content": "任务尚未完成。请按要求调用白名单工具，最终必须调用 submit_recipe。",
                })
                continue

            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or "",
                "tool_calls": [self._tool_call_dict(call) for call in tool_calls],
            })

            for tool_call in tool_calls:
                name = tool_call.function.name
                raw_arguments = tool_call.function.arguments or "{}"
                validation_errors: List[str] = []
                try:
                    arguments = json.loads(raw_arguments)
                    if name == "parse_markdown_sections":
                        result: Dict[str, Any] = parse_markdown_sections(markdown)
                        sections_called = True
                    elif name == "normalize_quantity":
                        result = normalize_quantity(str(arguments["raw"])).model_dump()
                        normalized_calls += 1
                    elif name in {"validate_recipe", "submit_recipe"}:
                        if not sections_called:
                            validation_errors.append("必须先调用 parse_markdown_sections")
                        recipe_payload = arguments.get("recipe", {})
                        ingredient_count = len(recipe_payload.get("ingredients", [])) if isinstance(recipe_payload, dict) else 0
                        if normalized_calls < ingredient_count:
                            validation_errors.append(
                                f"共 {ingredient_count} 个食材，但只调用了 {normalized_calls} 次 normalize_quantity"
                            )
                        recipe, model_errors = validate_recipe_payload(recipe_payload, markdown, source_path)
                        validation_errors.extend(model_errors)
                        if validation_errors:
                            result = {"valid": False, "errors": validation_errors}
                        else:
                            result = {"valid": True, "message": "菜谱结构与证据校验通过"}
                            if name == "submit_recipe":
                                submitted = recipe
                    else:
                        validation_errors.append(f"未知或未授权工具: {name}")
                        result = {"valid": False, "errors": validation_errors}
                except Exception as exc:
                    validation_errors.append(str(exc))
                    result = {"valid": False, "errors": validation_errors}

                result_json = json.dumps(result, ensure_ascii=False)
                traces.append(AgentTrace(
                    source_path=source_path,
                    round=round_number,
                    event="tool_call",
                    tool_name=name,
                    arguments_summary=raw_arguments[:500],
                    result_summary=result_json[:500],
                    validation_errors=validation_errors,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_json,
                })

            if submitted is not None:
                return AgentRunResult(
                    recipe=submitted,
                    traces=traces,
                    retries=retries,
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )

        error = f"达到最大工具调用轮数 {self.max_rounds}，未得到有效 submit_recipe"
        traces.append(AgentTrace(
            source_path=source_path,
            round=self.max_rounds,
            event="max_rounds_exceeded",
            result_summary=error,
        ))
        return AgentRunResult(
            recipe=None,
            traces=traces,
            error=error,
            retries=retries,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
