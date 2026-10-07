"""Hosted chat UI for read-only Airbnb Lakehouse analysis."""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

from tools import (
    compare_rental_scenarios,
    get_largest_listing_errors,
    get_latest_data_quality_audit,
    get_model_performance,
    search_airbnb_listings,
)


MODEL_ENDPOINT = os.getenv("DATABRICKS_SERVING_ENDPOINT", "databricks-gpt-oss-120b")
SYSTEM_PROMPT = """You are the Airbnb Valuation Analyst for a Databricks Lakehouse.
Use the available tools for current metrics, quality, listing errors, rental scenarios, or semantically similar listings.
Use semantic listing search when a user describes what they want in natural language (features, amenities, or neighborhood).
Explain results in plain language. Nightly prices are advertised rates, not booked revenue.
Rental income and property values are assumption-based estimates, not appraisals.
Compare model metrics to the mean-price baseline and be candid when the model is weak.
Never claim to change data. All available tools are read-only."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_airbnb_listings",
            "description": "Find semantically matching Airbnb listings by description, amenities, property type, and neighborhood. Use for natural-language requests about listing features.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 1000}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_performance",
            "description": "Compare held-out model RMSE and R2 with the mean-price baseline.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_data_quality_audit",
            "description": "Get the latest audit counts and quality warning flags.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_largest_listing_errors",
            "description": "Show held-out listings with the largest nightly-price prediction errors.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 25}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_rental_scenarios",
            "description": "Compare average monthly gross rent, NOI, and income-based value under occupancy, expense, and cap-rate assumptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "occupancy_rates": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "maxItems": 5},
                    "expense_ratios": {"type": "array", "items": {"type": "number", "minimum": 0, "exclusiveMaximum": 1}, "maxItems": 5},
                    "cap_rate": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                },
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_airbnb_listings": search_airbnb_listings,
    "get_model_performance": get_model_performance,
    "get_latest_data_quality_audit": get_latest_data_quality_audit,
    "get_largest_listing_errors": get_largest_listing_errors,
    "compare_rental_scenarios": compare_rental_scenarios,
}


@st.cache_resource
def get_model_client() -> DatabricksOpenAI:
    return DatabricksOpenAI(workspace_client=WorkspaceClient())


def answer_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "I couldn’t produce a text answer. Please try asking in a different way."


def ask_agent(history: list[dict[str, str]], prompt: str) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-8:],
        {"role": "user", "content": prompt},
    ]
    client = get_model_client()

    for _ in range(4):
        response = client.chat.completions.create(
            model=MODEL_ENDPOINT,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=900,
            temperature=0.1,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            return answer_text(message.content) or "I couldn’t find a text answer. Please try again."

        normalized_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": normalized_calls})

        for call in calls:
            name = call.function.name
            if name not in TOOL_FUNCTIONS:
                result: Any = {"error": "Tool is not allowed."}
            else:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = TOOL_FUNCTIONS[name](**arguments)
                except (TypeError, ValueError, RuntimeError) as error:
                    result = {"error": str(error)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    return "I reached the tool-call limit for this question. Please try a narrower request."


st.set_page_config(page_title="Airbnb Valuation Analyst", page_icon="A", layout="centered")
st.title("Airbnb Valuation Analyst")
st.caption("Semantic listing search plus read-only model, data quality, and rental scenario analysis.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

for turn in st.session_state.conversation:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input("Describe a listing, ask about model results, or compare rental scenarios"):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the Lakehouse..."):
            try:
                history = st.session_state.conversation[:-1]
                reply = ask_agent(history, prompt)
            except Exception as error:
                reply = f"I couldn’t reach the Databricks model or Lakehouse: {error}"
        st.markdown(reply)
    st.session_state.conversation.append({"role": "assistant", "content": reply})