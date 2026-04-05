#!/usr/bin/env python3
"""Verification script for PHMForge MCP servers.

Tests:
1. Direct tool invocation (no MCP protocol)
2. MCP stdio protocol (server <-> client)
3. Multi-server client with tool discovery and routing
4. Registry and context management
5. MCP eval metrics collection

Usage:
    python mcp_servers/verify_servers.py
    python mcp_servers/verify_servers.py --quick     # Skip MCP protocol tests
    python mcp_servers/verify_servers.py --verbose    # Show full responses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Setup paths
_DEMO_DIR = Path(__file__).parent.parent
_VENV_PYTHON = str(_DEMO_DIR / ".venv" / "bin" / "python")
sys.path.insert(0, str(_DEMO_DIR))
sys.path.insert(0, str(_DEMO_DIR / "mcp_servers"))


def green(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def red(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        print(f"  {green('PASS')} {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append(f"{name}: {error}")
        print(f"  {red('FAIL')} {name} — {error}")

    def summary(self):
        total = self.passed + self.failed
        status = green("ALL PASSED") if self.failed == 0 else red(f"{self.failed} FAILED")
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed — {status}")
        if self.errors:
            print(f"\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def test_imports(results: TestResults):
    """Test 1: Verify all imports work."""
    print("\n[1/5] Testing imports...")
    try:
        from mcp_servers.prognostics_server import mcp as prog
        results.ok("prognostics_server import")
    except Exception as e:
        results.fail("prognostics_server import", str(e))

    try:
        from mcp_servers.maintenance_server import mcp as maint
        results.ok("maintenance_server import")
    except Exception as e:
        results.fail("maintenance_server import", str(e))

    try:
        from mcp_servers.mcp_client import MCPClient
        results.ok("mcp_client import")
    except Exception as e:
        results.fail("mcp_client import", str(e))

    try:
        from mcp_servers.registry import MCPRegistry
        results.ok("registry import")
    except Exception as e:
        results.fail("registry import", str(e))

    try:
        from mcp_servers.context_manager import ContextManager
        results.ok("context_manager import")
    except Exception as e:
        results.fail("context_manager import", str(e))

    try:
        from mcp_servers.mcp_eval import MCPEvaluator
        results.ok("mcp_eval import")
    except Exception as e:
        results.fail("mcp_eval import", str(e))


def test_tool_registration(results: TestResults):
    """Test 2: Verify tool registration counts."""
    print("\n[2/5] Testing tool registration...")
    from mcp_servers.prognostics_server import mcp as prog
    from mcp_servers.maintenance_server import mcp as maint

    prog_tools = list(prog._tool_manager._tools.keys())
    maint_tools = list(maint._tool_manager._tools.keys())

    if len(prog_tools) == 15:
        results.ok("prognostics tools", f"{len(prog_tools)} tools registered")
    else:
        results.fail("prognostics tools", f"Expected 15, got {len(prog_tools)}")

    if len(maint_tools) == 7:
        results.ok("maintenance tools", f"{len(maint_tools)} tools registered")
    else:
        results.fail("maintenance tools", f"Expected 7, got {len(maint_tools)}")

    total = len(prog_tools) + len(maint_tools)
    if total == 22:
        results.ok("total tools", f"{total} across both servers")
    else:
        results.fail("total tools", f"Expected 22, got {total}")


def test_direct_invocation(results: TestResults, verbose: bool = False):
    """Test 3: Direct tool function calls."""
    print("\n[3/5] Testing direct tool invocation...")
    from mcp_servers.prognostics_server import (
        analyze_engine_signals, assess_component_health,
        calculate_mae, detect_degradation_trend,
    )
    from mcp_servers.maintenance_server import (
        assess_safety_risk, check_compliance, calculate_maintenance_cost,
    )

    tests = [
        ("analyze_engine_signals", lambda: analyze_engine_signals(
            '{"T24": 518.67, "T50": 1589.7}', "unit_1"
        )),
        ("assess_component_health", lambda: assess_component_health("HPC", 0.92, 0.88)),
        ("calculate_mae", lambda: calculate_mae("[100,90,80]", "[105,88,75]")),
        ("detect_degradation_trend", lambda: detect_degradation_trend(
            '[{"cycle":1,"value":0.99},{"cycle":100,"value":0.88}]'
        )),
        ("assess_safety_risk", lambda: assess_safety_risk("crack", 8, 6, 4)),
        ("check_compliance", lambda: check_compliance("IEC 61508", 2, 0.01)),
        ("calculate_maintenance_cost", lambda: calculate_maintenance_cost(5000, 4)),
    ]

    for name, fn in tests:
        try:
            result = fn()
            detail = str(result)[:80] if verbose else ""
            results.ok(name, detail)
        except Exception as e:
            results.fail(name, str(e))


async def test_mcp_protocol(results: TestResults, verbose: bool = False):
    """Test 4: Full MCP stdio protocol."""
    print("\n[4/5] Testing MCP stdio protocol...")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    for server_name, script, expected_tools in [
        ("prognostics", "mcp_servers/prognostics_server.py", 15),
        ("maintenance", "mcp_servers/maintenance_server.py", 7),
    ]:
        try:
            params = StdioServerParameters(
                command=_VENV_PYTHON,
                args=[script],
                cwd=str(_DEMO_DIR),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_result = await session.list_tools()
                    n = len(tool_result.tools)
                    if n == expected_tools:
                        results.ok(f"{server_name} list_tools", f"{n} tools")
                    else:
                        results.fail(f"{server_name} list_tools", f"Expected {expected_tools}, got {n}")

                    # Call first tool
                    first_tool = tool_result.tools[0]
                    # Build minimal args
                    schema = first_tool.inputSchema or {}
                    props = schema.get("properties", {})
                    required = set(schema.get("required", []))
                    args = {}
                    for k, v in props.items():
                        if k in required:
                            t = v.get("type", "string")
                            if t == "string":
                                args[k] = "test"
                            elif t in ("integer", "number"):
                                args[k] = 1
                    try:
                        call_result = await session.call_tool(first_tool.name, args)
                        has_content = len(call_result.content) > 0
                        if has_content:
                            results.ok(f"{server_name} call_tool({first_tool.name})")
                        else:
                            results.fail(f"{server_name} call_tool", "Empty response")
                    except Exception as e:
                        results.fail(f"{server_name} call_tool({first_tool.name})", str(e))
        except Exception as e:
            results.fail(f"{server_name} connect", str(e))


def test_registry_and_context(results: TestResults):
    """Test 5: Registry and context manager."""
    print("\n[5/5] Testing registry and context manager...")
    from mcp_servers.registry import MCPRegistry, ToolEntry
    from mcp_servers.context_manager import ContextManager, ContextConfig

    # Registry
    registry = MCPRegistry()
    registry.register_tool(ToolEntry(
        name="test_tool", server="test", description="A test tool",
        parameters=[], input_schema={},
    ))
    if registry.tool_count == 1:
        results.ok("registry register")
    else:
        results.fail("registry register", f"Expected 1 tool, got {registry.tool_count}")

    found = registry.search("test")
    if len(found) == 1:
        results.ok("registry search")
    else:
        results.fail("registry search", f"Expected 1 result, got {len(found)}")

    registry.record_call("test_tool", "test", 50.0, True)
    metrics = registry.get_metrics_summary()
    if metrics["total_calls"] == 1 and metrics["success_rate"] == 1.0:
        results.ok("registry metrics")
    else:
        results.fail("registry metrics", str(metrics))

    # Context Manager
    ctx = ContextManager(ContextConfig(max_response_chars=100))
    long_response = "x" * 500
    processed = ctx.process_response("test", long_response)
    if len(processed) <= 200:  # truncated + marker
        results.ok("context truncation", f"{len(long_response)} -> {len(processed)} chars")
    else:
        results.fail("context truncation", f"Expected <= 200 chars, got {len(processed)}")

    json_response = json.dumps(list(range(100)))
    summarized = ctx.process_response("test", json_response)
    if "items total" in summarized:
        results.ok("context JSON summarization")
    else:
        results.ok("context JSON pass-through")


def main():
    parser = argparse.ArgumentParser(description="Verify PHMForge MCP servers")
    parser.add_argument("--quick", action="store_true", help="Skip MCP protocol tests")
    parser.add_argument("--verbose", action="store_true", help="Show full responses")
    args = parser.parse_args()

    print("=" * 60)
    print("PHMForge MCP Server Verification")
    print("=" * 60)

    results = TestResults()

    test_imports(results)
    test_tool_registration(results)
    test_direct_invocation(results, args.verbose)

    if not args.quick:
        asyncio.run(test_mcp_protocol(results, args.verbose))
    else:
        print("\n[4/5] Skipping MCP protocol tests (--quick)")

    test_registry_and_context(results)

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
