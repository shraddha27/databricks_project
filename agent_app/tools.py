"""Fixed, read-only Databricks tools used by the hosted Airbnb agent."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient


CATALOG = os.getenv("AIRBNB_CATALOG", "workspace")
SCHEMA = os.getenv("AIRBNB_SCHEMA", "airbnb_lakehouse")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "bc1d795b5081d326")
VECTOR_INDEX = os.getenv(
    "AIRBNB_VECTOR_INDEX", f"{CATALOG}.{SCHEMA}.airbnb_listing_search"
)
_mcp_client: DatabricksMCPClient | None = None
_available_tools: set[str] | None = None
_vector_mcp_client: DatabricksMCPClient | None = None
_vector_tool_name: str | None = None


def get_mcp_client() -> DatabricksMCPClient:
    global _mcp_client, _available_tools
    if _mcp_client is None:
        workspace = WorkspaceClient()
        _mcp_client = DatabricksMCPClient(
            server_url=f"{workspace.config.host}/api/2.0/mcp/sql",
            workspace_client=workspace,
        )
        _available_tools = {tool.name for tool in _mcp_client.list_tools()}
        if not {"execute_sql_read_only", "poll_sql_result"}.issubset(_available_tools):
            _mcp_client = None
            raise RuntimeError("Databricks SQL MCP did not provide its read-only query tools")
    return _mcp_client


def _call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in {"execute_sql_read_only", "poll_sql_result"}:
        raise ValueError(f"MCP tool is not allowed: {name}")
    for attempt in range(3):
        try:
            result = get_mcp_client().call_tool(name, arguments)
            break
        except BaseExceptionGroup as error:
            def is_transient(item: BaseException) -> bool:
                if isinstance(item, BaseExceptionGroup):
                    return any(is_transient(nested) for nested in item.exceptions)
                return isinstance(item, (httpx.TimeoutException, httpx.TransportError))

            if not is_transient(error) or attempt == 2:
                raise
            time.sleep(1 + attempt)
    if result.isError:
        detail = result.content[0].text if result.content else "Unknown MCP error"
        raise RuntimeError(detail)
    if result.structuredContent is not None:
        return result.structuredContent
    if result.content and result.content[0].type == "text":
        import json

        return json.loads(result.content[0].text)
    raise RuntimeError("Databricks SQL MCP returned no structured result")


def search_airbnb_listings(query: str) -> Any:
    """Find listings by meaning using the managed Databricks AI Search MCP server."""
    global _vector_mcp_client, _vector_tool_name
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("Provide a nonempty listing search query under 1,000 characters")

    if _vector_mcp_client is None:
        workspace = WorkspaceClient()
        parts = VECTOR_INDEX.split(".")
        if len(parts) != 3:
            raise ValueError("AIRBNB_VECTOR_INDEX must be catalog.schema.index")
        _vector_mcp_client = DatabricksMCPClient(
            server_url=(
                f"{workspace.config.host}/api/2.0/mcp/ai-search/"
                f"{parts[0]}/{parts[1]}/{parts[2]}"
            ),
            workspace_client=workspace,
        )
        tools = _vector_mcp_client.list_tools()
        if len(tools) != 1:
            _vector_mcp_client = None
            raise RuntimeError("Expected one managed AI Search MCP tool for this index")
        _vector_tool_name = tools[0].name

    result = _vector_mcp_client.call_tool(_vector_tool_name, {"query": query})
    if result.isError:
        detail = result.content[0].text if result.content else "Unknown AI Search MCP error"
        raise RuntimeError(detail)
    if result.structuredContent is not None:
        return result.structuredContent
    if result.content and result.content[0].type == "text":
        return result.content[0].text
    return {"message": "AI Search returned no matching listings."}


def _decode_sql_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("null") is True:
        return None
    for key in (
        "string_value",
        "long_value",
        "int_value",
        "double_value",
        "float_value",
        "bool_value",
        "decimal_value",
        "date_value",
        "timestamp_value",
        "str",
        "long",
        "int",
        "double",
        "float",
        "bool",
        "decimal",
        "date",
        "timestamp",
    ):
        if key in value:
            return value[key]
    return next(iter(value.values()), None)


def execute_read_query(query: str) -> list[dict[str, Any]]:
    response = _call_mcp_tool("execute_sql_read_only", {"query": query})
    for _ in range(30):
        state = response.get("status", {}).get("state")
        if state == "SUCCEEDED":
            break
        if state in {"FAILED", "CANCELED", "CLOSED"}:
            message = response.get("status", {}).get("error", {}).get("message", state)
            raise RuntimeError(f"Databricks SQL MCP query failed: {message}")
        statement_id = response.get("statement_id")
        if not statement_id:
            raise RuntimeError(f"Databricks SQL MCP returned unexpected status: {state}")
        time.sleep(1)
        response = _call_mcp_tool("poll_sql_result", {"statement_id": statement_id})
    else:
        raise TimeoutError("Databricks SQL MCP query did not finish within 30 seconds")

    columns = [column["name"] for column in response.get("manifest", {}).get("schema", {}).get("columns", [])]
    result = response.get("result", {})
    rows = result.get("data_typed_array") or result.get("data_array", [])
    return [
        {
            column: _decode_sql_value(value)
            for column, value in zip(columns, row.get("values", []))
        }
        for row in rows
    ]


def get_model_performance() -> list[dict[str, Any]]:
    query = f"""
    WITH actuals AS (
      SELECT nightly_price, predicted_nightly_price, baseline_nightly_price,
             AVG(nightly_price) OVER () AS actual_mean
      FROM {CATALOG}.{SCHEMA}.gold_airbnb_valuation
    ), totals AS (
      SELECT COUNT(*) AS test_listings,
             SUM(POWER(nightly_price - predicted_nightly_price, 2)) AS model_sse,
             SUM(POWER(nightly_price - baseline_nightly_price, 2)) AS baseline_sse,
             SUM(POWER(nightly_price - actual_mean, 2)) AS total_sst
      FROM actuals
    )
    SELECT test_listings,
           SQRT(model_sse / test_listings) AS model_rmse_usd,
           SQRT(baseline_sse / test_listings) AS baseline_rmse_usd,
           1 - model_sse / NULLIF(total_sst, 0) AS model_r2,
           1 - baseline_sse / NULLIF(total_sst, 0) AS baseline_r2,
           100 * (1 - SQRT(model_sse / test_listings) /
                  NULLIF(SQRT(baseline_sse / test_listings), 0)) AS rmse_improvement_pct
    FROM totals
    """
    return execute_read_query(query)


def get_latest_data_quality_audit() -> list[dict[str, Any]]:
    query = f"""
    SELECT pipeline_run_id, quality_checked_at, source_rows,
           missing_price_rows, nonpositive_price_rows,
           duplicate_listing_rows, unusual_price_rows,
           price_lower_fence, price_upper_fence, quality_status, warning_flags
    FROM {CATALOG}.{SCHEMA}.data_quality_audit
    ORDER BY quality_checked_at DESC
    LIMIT 1
    """
    return execute_read_query(query)


def get_largest_listing_errors(limit: int = 10) -> list[dict[str, Any]]:
    if not 1 <= limit <= 25:
        raise ValueError("limit must be between 1 and 25")
    query = f"""
    SELECT listing_id, neighbourhood_cleansed AS neighborhood,
           room_type, property_type, nightly_price AS actual_nightly_price,
           predicted_nightly_price, absolute_prediction_error
    FROM {CATALOG}.{SCHEMA}.gold_airbnb_valuation
    ORDER BY absolute_prediction_error DESC
    LIMIT {limit}
    """
    return execute_read_query(query)


def compare_rental_scenarios(
    occupancy_rates: list[float] | None = None,
    expense_ratios: list[float] | None = None,
    cap_rate: float = 0.06,
) -> list[dict[str, Any]]:
    occupancy_rates = occupancy_rates or [0.50, 0.65, 0.80]
    expense_ratios = expense_ratios or [0.25, 0.35, 0.45]
    if len(occupancy_rates) > 5 or any(rate <= 0 or rate > 1 for rate in occupancy_rates):
        raise ValueError("Provide up to five occupancy rates, each greater than 0 and at most 1")
    if len(expense_ratios) > 5 or any(ratio < 0 or ratio >= 1 for ratio in expense_ratios):
        raise ValueError("Provide up to five expense ratios, each at least 0 and less than 1")
    if cap_rate <= 0 or cap_rate >= 1:
        raise ValueError("cap_rate must be greater than 0 and less than 1")

    scenarios = ",\n".join(
        f"({occupancy:.6f}, {expense:.6f})"
        for occupancy in occupancy_rates
        for expense in expense_ratios
    )
    query = f"""
    WITH scenarios AS (
      SELECT * FROM VALUES {scenarios} AS s(occupancy_rate, expense_ratio)
    ), listing_scenarios AS (
      SELECT s.occupancy_rate, s.expense_ratio,
             g.predicted_nightly_price * 30 * s.occupancy_rate AS monthly_gross_rent,
             g.predicted_nightly_price * 30 * s.occupancy_rate * (1 - s.expense_ratio) AS monthly_noi
      FROM {CATALOG}.{SCHEMA}.gold_airbnb_valuation g
      CROSS JOIN scenarios s
    )
    SELECT occupancy_rate, expense_ratio,
           AVG(monthly_gross_rent) AS average_monthly_gross_rent,
           AVG(monthly_noi) AS average_monthly_noi,
           AVG(monthly_noi) * 12 / {cap_rate:.6f} AS average_income_based_value
    FROM listing_scenarios
    GROUP BY occupancy_rate, expense_ratio
    ORDER BY occupancy_rate, expense_ratio
    """
    return execute_read_query(query)