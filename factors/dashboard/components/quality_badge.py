"""Quality badge helpers kept independent of Streamlit for testability."""

from __future__ import annotations


def quality_label(value: str | None) -> str:
    labels = {"A": "A 原始/已验证", "B": "B 原始/有限制", "C": "C proxy/人工分类", "D": "D 不足"}
    return labels.get(str(value), "未知")
