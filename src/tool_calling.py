"""Helpers for schema-first tool call extraction and validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

SchemaMap = dict[str, type[Any]]


@dataclass
class ToolCallValidationIssue:
    index: int
    message: str
    tool: str = ""


class ToolCallValidationError(ValueError):
    """Raised when one or more tool calls violate contract."""

    def __init__(self, issues: list[ToolCallValidationIssue]):
        self.issues = issues
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if not self.issues:
            return "invalid tool call payload"
        parts = []
        for issue in self.issues:
            tool_part = f" tool={issue.tool!r}" if issue.tool else ""
            parts.append(f"[#{issue.index}{tool_part}] {issue.message}")
        return "; ".join(parts)


def canonical_tool_name(name: Any) -> str:
    raw = str(name or "").strip()
    if raw.startswith("functions."):
        return raw.split(".", 1)[1].strip()
    return raw


def extract_native_tool_calls(result: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of structured native tool calls from Crew result."""

    entries: list[Any] = []
    direct_fields = ("tool_calls", "tools_called", "tool_invocations")
    for field in direct_fields:
        value = _extract_attr_or_key(result, field)
        if isinstance(value, list):
            entries.extend(value)

    tasks_output = _extract_attr_or_key(result, "tasks_output")
    if isinstance(tasks_output, list):
        for task_output in tasks_output:
            for field in direct_fields:
                value = _extract_attr_or_key(task_output, field)
                if isinstance(value, list):
                    entries.extend(value)

    parsed: list[dict[str, Any]] = []
    for entry in entries:
        call = _coerce_tool_call(entry)
        if call is not None:
            parsed.append(call)
    return parsed


def validate_and_normalize_tool_calls(
    tool_calls: Any,
    *,
    schema_by_tool: SchemaMap,
) -> list[dict[str, Any]]:
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise ToolCallValidationError(
            [ToolCallValidationIssue(index=0, message="tool_calls must be a list")]
        )

    normalized: list[dict[str, Any]] = []
    issues: list[ToolCallValidationIssue] = []
    for idx, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            issues.append(ToolCallValidationIssue(index=idx, message="tool call must be an object"))
            continue

        tool_name = canonical_tool_name(raw_call.get("tool"))
        if not tool_name:
            issues.append(ToolCallValidationIssue(index=idx, message="tool is required"))
            continue
        if tool_name not in schema_by_tool:
            issues.append(
                ToolCallValidationIssue(index=idx, tool=tool_name, message="unknown tool name")
            )
            continue

        reason = str(raw_call.get("reason") or "").strip() or f"validated:{tool_name}"
        args_raw = raw_call.get("args", {})
        if not isinstance(args_raw, dict):
            issues.append(
                ToolCallValidationIssue(
                    index=idx,
                    tool=tool_name,
                    message="args must be an object",
                )
            )
            continue

        args = _validate_schema(schema_by_tool[tool_name], args_raw)
        if args is None:
            issues.append(
                ToolCallValidationIssue(
                    index=idx,
                    tool=tool_name,
                    message="args do not match args_schema",
                )
            )
            continue

        normalized.append(
            {
                "tool": tool_name,
                "reason": reason,
                "args": args,
            }
        )

    if issues:
        raise ToolCallValidationError(issues)
    return normalized


def _extract_attr_or_key(container: Any, key: str) -> Any:
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _coerce_tool_call(raw_call: Any) -> dict[str, Any] | None:
    if isinstance(raw_call, dict):
        if isinstance(raw_call.get("tool"), (str, int, float)):
            return {
                "tool": raw_call.get("tool"),
                "reason": raw_call.get("reason", ""),
                "args": raw_call.get("args", {}),
            }
        function_payload = raw_call.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name")
            arguments = function_payload.get("arguments")
            return {
                "tool": name,
                "reason": raw_call.get("reason", ""),
                "args": _parse_arguments(arguments),
            }
        name = raw_call.get("name") or raw_call.get("tool_name")
        if name:
            arguments = raw_call.get("arguments", raw_call.get("tool_input", raw_call.get("input", {})))
            return {
                "tool": name,
                "reason": raw_call.get("reason", ""),
                "args": _parse_arguments(arguments),
            }
        return None

    function_payload = getattr(raw_call, "function", None)
    if function_payload is not None:
        name = _extract_attr_or_key(function_payload, "name")
        arguments = _extract_attr_or_key(function_payload, "arguments")
        return {
            "tool": name,
            "reason": str(getattr(raw_call, "reason", "") or ""),
            "args": _parse_arguments(arguments),
        }

    name = getattr(raw_call, "name", None) or getattr(raw_call, "tool_name", None)
    if name:
        arguments = (
            getattr(raw_call, "arguments", None)
            or getattr(raw_call, "tool_input", None)
            or getattr(raw_call, "input", None)
        )
        return {
            "tool": name,
            "reason": str(getattr(raw_call, "reason", "") or ""),
            "args": _parse_arguments(arguments),
        }
    return None


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}
    return {}


def _validate_schema(schema: type[Any], args: dict[str, Any]) -> dict[str, Any] | None:
    try:
        if hasattr(schema, "model_validate"):  # pydantic v2
            validated = schema.model_validate(args)
            if hasattr(validated, "model_dump"):
                return validated.model_dump(exclude_none=True)
            return dict(args)
        if hasattr(schema, "parse_obj"):  # pydantic v1 fallback
            validated = schema.parse_obj(args)
            if hasattr(validated, "dict"):
                return validated.dict(exclude_none=True)
            return dict(args)
    except ValidationError:
        return None
    return dict(args)
