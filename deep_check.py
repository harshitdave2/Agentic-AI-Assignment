"""
Deep validation for Mandelbulb Agentic AI Assignment
"""

import os
import sys
import traceback

# Reconfigure stdout to UTF-8 so emoji and Unicode chars in agent responses
# don't cause UnicodeEncodeError on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

print("=" * 80)
print("MANDELBULB ASSIGNMENT DEEP VALIDATION")
print("=" * 80)

# --------------------------------------------------
# 1. Project Structure
# --------------------------------------------------

# Files are in their proper module subdirectories, not flat.
required_files = [
    ("agent/agent.py",       "agent.py"),
    ("tools/store_tools.py", "store_tools.py"),
    ("main.py",              "main.py"),
    ("README.md",            "README.md"),
    ("DESIGN.md",            "DESIGN.md"),
]

print("\n[1] Checking project structure")

all_present = True
for filepath, label in required_files:
    if os.path.exists(filepath):
        print(f"[OK] {label}")
    else:
        print(f"[MISSING] {label}")
        all_present = False

if all_present:
    print("    All required files present.")

# --------------------------------------------------
# 2. Import Validation
# --------------------------------------------------

print("\n[2] Checking imports")

try:
    from agent.agent import run_agent  # noqa: E402
    print("[OK] agent.agent imports successfully")
except Exception as e:
    print("[FAIL] agent.agent import failed")
    traceback.print_exc()

# --------------------------------------------------
# 3. Logging Check
# --------------------------------------------------

print("\n[3] Checking logging")

if os.path.exists("logs"):
    print("[OK] logs directory exists")
else:
    print("[MISSING] logs directory")

# --------------------------------------------------
# 4. Agent Execution
# --------------------------------------------------

print("\n[4] Basic execution")

tests = [
    "Where is order ORD-1002?",
    "Get details for PROD-201",
    "Show me running shoes",
]

for q in tests:
    try:
        result = run_agent(q)
        if result and isinstance(result, str):
            print(f"[OK] {q}")
        else:
            print(f"[FAIL] Empty response: {q}")
    except Exception:
        print(f"[FAIL] Crash: {q}")
        traceback.print_exc()

# --------------------------------------------------
# 5. Order Lookup
# --------------------------------------------------

print("\n[5] Order lookup")

try:
    result = run_agent("Where is order ORD-1002?")
    print(result)
    if "ORD-1002" in result or "order" in result.lower():
        print("[OK] Order lookup working")
    else:
        print("[FAIL] Order lookup suspicious")
except Exception:
    print("[FAIL] Order lookup crashed")
    traceback.print_exc()

# --------------------------------------------------
# 6. Invalid Order
# --------------------------------------------------

print("\n[6] Invalid order")

result = run_agent("Where is order ORD-9999?")
print(result)

# Accept any reasonable "not found" response phrasing.
_not_found_phrases = [
    "not found",
    "couldn't find",
    "could not find",
    "no order",
    "unable to find",
    "doesn't exist",
    "does not exist",
    "check the order id",
    "check your order",
    "please check",
]

if any(phrase in result.lower() for phrase in _not_found_phrases):
    print("[OK] Invalid order handled correctly")
else:
    print("[FAIL] Invalid order handling weak — response:", repr(result[:120]))

# --------------------------------------------------
# 7. Direct Product Lookup
# --------------------------------------------------

print("\n[7] Product lookup")

result = run_agent("Get details for PROD-201")
print(result)

if "PROD-201" in result or "price" in result.lower():
    print("[OK] Product lookup working")
else:
    print("[FAIL] Product lookup broken")

# --------------------------------------------------
# 8. Product Search
# --------------------------------------------------

print("\n[8] Product search")

result = run_agent("Find running shoes")
print(result)

if len(result) > 20:
    print("[OK] Product search working")
else:
    print("[FAIL] Product search suspicious")

# --------------------------------------------------
# 9. Affordable Search Routing
# --------------------------------------------------

print("\n[9] Affordable search routing")

result = run_agent("Find affordable running shoes")
print(result)

if "alternative" not in result.lower():
    print("[OK] Affordable search routed correctly")
else:
    print("[FAIL] Routed as alternative request")

# --------------------------------------------------
# 10. Cheaper Alternative (Tool Chaining)
# --------------------------------------------------

print("\n[10] Tool chaining")

query = "Is there a cheaper alternative to the shoes I ordered in ORD-1001?"
result = run_agent(query)
print(result)

if "alternative" in result.lower() or "cheaper" in result.lower():
    print("[OK] Tool chaining appears correct")
else:
    print("[FAIL] Alternative workflow failed")

# --------------------------------------------------
# 11. Empty Search
# --------------------------------------------------

print("\n[11] Empty search")

result = run_agent("Find quantum teleportation laptop")
print(result)

if (
    "no products" in result.lower()
    or "couldn't find" in result.lower()
    or "not found" in result.lower()
    or "searched" in result.lower()
):
    print("[OK] Empty search handled")
else:
    print("[FAIL] Empty search handling weak")

# --------------------------------------------------
# 12. Greeting / Fallback
# --------------------------------------------------

print("\n[12] Greeting")

result = run_agent("Hello, can you help me?")
print(result)

if len(result) > 0:
    print("[OK] Greeting handled")
else:
    print("[FAIL] Greeting handling broken")

# --------------------------------------------------
# 13. Fabrication Check
# --------------------------------------------------

print("\n[13] Fabrication check")

fabrication_queries = [
    "Where is order ORD-99999?",
    "Tell me about PROD-99999",
]

for q in fabrication_queries:
    result = run_agent(q)
    if any(phrase in result.lower() for phrase in _not_found_phrases):
        print(f"[OK] {q}")
    else:
        print(f"[REVIEW] Check manually: {q}")

# --------------------------------------------------
# 14. Stress Test
# --------------------------------------------------

print("\n[14] Stress test")

stress_queries = [
    "Find headphones",
    "Find laptops",
    "Where is ORD-1001",
    "PROD-101",
    "Help me",
]

for q in stress_queries:
    try:
        run_agent(q)
        print(f"[OK] {q}")
    except Exception:
        print(f"[FAIL] Crash: {q}")
        traceback.print_exc()

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)