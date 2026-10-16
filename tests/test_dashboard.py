"""Dashboard contract tests.

The dashboard is exercised manually in a browser, but two classes of bug are
cheap to catch statically and were both hit during development:

1. Streamlit raises ``StreamlitDuplicateElementId`` when two element calls are
   given identical arguments and no ``key``; every ``st.plotly_chart`` call must
   therefore carry a unique key.
2. The live controls forward a payload to ``extend_dataset``; a renamed
   parameter would only fail when the button is clicked, so the keyword names are
   checked against the real signature here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from streamguard.simulator import extend_dataset

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def parse_dashboard() -> ast.Module:
    return ast.parse(DASHBOARD.read_text(encoding="utf-8"), filename=str(DASHBOARD))


def called_name(node: ast.Call) -> str | None:
    """Name of the callable for ``obj.method(...)`` and ``function(...)`` alike."""
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def keyword_literal(node: ast.Call, name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return None


def test_every_plotly_chart_has_a_unique_key() -> None:
    calls = [
        node
        for node in ast.walk(parse_dashboard())
        if isinstance(node, ast.Call) and called_name(node) == "plotly_chart"
    ]
    assert calls, "expected the dashboard to render charts"
    keys = [keyword_literal(node, "key") for node in calls]
    assert all(key is not None for key in keys), "a plotly_chart call is missing key="
    assert len(set(keys)) == len(keys), f"duplicate plotly_chart keys: {keys}"


def test_dataframe_calls_render_distinct_content() -> None:
    """Streamlit derives an element id from the call's arguments.

    Two identical ``st.dataframe`` calls (or two with the same explicit key)
    would raise ``StreamlitDuplicateElementId`` at runtime.
    """
    calls = [
        node
        for node in ast.walk(parse_dashboard())
        if isinstance(node, ast.Call) and called_name(node) == "dataframe"
    ]
    assert calls
    signatures = [ast.dump(node) for node in calls]
    assert len(set(signatures)) == len(signatures), "two dataframe calls share identical arguments"
    keys = [keyword_literal(node, "key") for node in calls if keyword_literal(node, "key")]
    assert len(set(keys)) == len(keys), f"duplicate dataframe keys: {keys}"


def test_live_control_payloads_match_the_simulator_signature() -> None:
    parameters = set(inspect.signature(extend_dataset).parameters)
    calls = [
        node
        for node in ast.walk(parse_dashboard())
        if isinstance(node, ast.Call) and called_name(node) == "next_append"
    ]
    assert calls, "expected live controls to call next_append"
    for node in calls:
        for keyword in node.keywords:
            assert keyword.arg in parameters, f"next_append passes unknown argument {keyword.arg!r}"


def test_alert_state_path_is_writable_and_relative() -> None:
    calls = [
        node
        for node in ast.walk(parse_dashboard())
        if isinstance(node, ast.Call) and called_name(node) == "AlertStore"
    ]
    assert calls
    for node in calls:
        assert node.args, "AlertStore must be given an explicit path"
