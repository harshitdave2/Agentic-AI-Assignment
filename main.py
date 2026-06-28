"""
main.py -- CLI demo runner for the AI Store Agent.

Run:  python main.py

Demonstrates the full range of agent capabilities:
  order lookup, invalid order, product lookup, invalid product,
  search, empty search, cheaper alternative, greeting, fallback.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Ensure the console handles UTF-8 safely on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from agent.agent import run_agent
from config.settings import LLM_PROVIDER, GEMINI_API_KEY

# --- Demo Scenarios -----------------------------------------------------------
# Each tuple: (question, category)
# Covers all required demo cases from the assignment brief.

DEMO_QUESTIONS: list[tuple[str, str]] = [
    # Order lookup
    ("Where is my order ORD-1002?",                                         "Order Lookup"),
    ("What is the status of ORD-1001?",                                     "Order Lookup"),

    # Invalid order
    ("Where is order ORD-9999?",                                            "Invalid Order"),

    # Product lookup
    ("Get details for PROD-201",                                            "Product Lookup"),
    ("Tell me about PROD-305",                                              "Product Lookup"),

    # Invalid product
    ("Get details for PROD-0000",                                           "Invalid Product"),

    # Product search
    ("Show me some wireless headphones",                                    "Product Search"),
    ("I'm looking for jeans",                                               "Product Search"),

    # Empty search (no matching products)
    ("Find quantum teleportation laptop",                                   "Empty Search"),

    # Cheaper alternative (tool chaining: order -> product -> search)
    ("Is there a cheaper alternative to the shoes I ordered in ORD-1001?", "Tool Chaining"),
    ("Can you suggest a budget-friendly option for what I bought in ORD-1002?", "Tool Chaining"),

    # Greeting
    ("Hello, can you help me?",                                             "Greeting"),

    # Fallback (unrecognised intent)
    ("What is the weather today?",                                          "Fallback"),
]


def main() -> None:
    from agent.planner import get_planner
    planner = get_planner()
    planner_mode = (
        "Gemini" if planner.__class__.__name__ == "GeminiProvider"
        else "Deterministic"
    )

    print("=" * 65)
    print("   AI Store Agent -- Sample Inputs & Outputs")
    print(f"   Planner: {planner_mode}")
    print("=" * 65)

    current_category = ""
    for i, (question, category) in enumerate(DEMO_QUESTIONS, 1):
        if category != current_category:
            current_category = category
            print(f"\n{'-' * 65}")
            print(f"  [{category}]")
            print(f"{'-' * 65}")

        print(f"\n[Q{i}] {question}")
        print("-" * 55)
        response = run_agent(question)
        print(response)
        print()


if __name__ == "__main__":
    main()
