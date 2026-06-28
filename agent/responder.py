"""
agent/responder.py — Response generation.

Converts a populated ExecutionContext into a customer-friendly response string.

Design principles
─────────────────
• Never expose raw dicts, JSON, or internal structures to the customer.
• Always produce warm, professional, natural-language responses.
• Each response section is a separate function — easy to modify independently.
• Every possible ExecutionContext state is handled explicitly (no silent gaps).

Response section functions
──────────────────────────
_format_order(order)                 — order status section
_format_alternatives(ctx)           — cheaper-alternative section
_format_product_detail(product)     — product detail section
_format_search_results(results, q)  — product search section
_format_error_response(errors)      — error-only response
_fallback_response()                — unknown intent guidance
"""

from __future__ import annotations

import logging

from llm.base import (
    AgentPlan,
    INTENT_CHEAPER_ALTERNATIVE,
    INTENT_PRODUCT_SEARCH,
)
from agent.executor import ExecutionContext

logger = logging.getLogger("agent.responder")


def build_response(question: str, plan: AgentPlan, ctx: ExecutionContext) -> str:
    """
    Build a customer-friendly response from the execution context.

    Parameters
    ──────────
    question : str
        The original customer question (available for future personalisation).
    plan : AgentPlan
        Used for intent and entity access (e.g. search query label).
    ctx : ExecutionContext
        The fully populated result of tool execution.

    Returns
    ───────
    str
        A warm, natural, customer-facing response. Never exposes raw data.
    """
    # ── Error only: no usable results came back ────────────────────────────
    if ctx.has_errors and not ctx.has_results:
        return _format_error_response(ctx.errors)

    lines: list[str] = []

    # ── Order section (shared by order_status and cheaper_alternative) ─────
    if ctx.order:
        lines.extend(_format_order(ctx.order))

    # ── Cheaper alternative section ────────────────────────────────────────
    if plan.intent == INTENT_CHEAPER_ALTERNATIVE:
        lines.extend(_format_alternatives(ctx))
        return "\n".join(lines)

    # ── Direct product detail section ──────────────────────────────────────
    if ctx.product_detail:
        lines.extend(_format_product_detail(ctx.product_detail))
        return "\n".join(lines)

    # ── Product search section ─────────────────────────────────────────────
    if plan.intent == INTENT_PRODUCT_SEARCH:
        query_label = plan.entities.get("search_query") or "your search"
        lines.extend(_format_search_results(ctx.search_results, query_label))
        return "\n".join(lines)

    # ── Pure order status (formatted above, return as-is) ─────────────────
    if ctx.order:
        return "\n".join(lines)

    # ── Unknown intent or no data ──────────────────────────────────────────
    return _fallback_response()


# ─── Section Formatters ───────────────────────────────────────────────────────

def _format_order(order: dict) -> list[str]:
    """Format an order status section with friendly, professional language."""
    lines: list[str] = []
    status = order["status"]

    lines.append(f"Here's the update for order **{order['order_id']}**:")
    lines.append(f"   Status: **{status}**")

    if status == "Delivered":
        lines.append(
            f"   Your order was delivered on "
            f"{order.get('delivered_on', 'the expected date')}."
        )
    elif status == "In Transit":
        lines.append(
            f"   Your package is on its way! "
            f"Expected delivery: **{order.get('estimated_delivery', 'soon')}**."
        )
        if order.get("tracking_url"):
            lines.append(f"   Track it here: {order['tracking_url']}")
    elif status == "Processing":
        lines.append(
            f"   Your order is being prepared. "
            f"Estimated delivery: **{order.get('estimated_delivery', 'coming soon')}**."
        )
    elif status == "Cancelled":
        lines.append(
            "   This order has been cancelled. "
            "If you believe this is a mistake, please contact support."
        )

    if order.get("items"):
        items_str = ", ".join(
            f"{i['name']} (x{i['qty']})" for i in order["items"]
        )
        lines.append(f"   Items: {items_str}")
        lines.append(f"   Order Total: Rs. {order['total']:,}")

    return lines


def _format_alternatives(ctx: ExecutionContext) -> list[str]:
    """Format the cheaper-alternative section."""
    lines: list[str] = []

    if ctx.ordered_product:
        op = ctx.ordered_product
        lines.append(
            f"\nYou asked for cheaper alternatives to "
            f"**{op['name']}** (Rs. {op['price']:,})."
        )

    if ctx.search_results:
        lines.append("   Here are some budget-friendly options you might like:\n")
        for i, p in enumerate(ctx.search_results[:3], 1):
            stock = "In Stock" if p["in_stock"] else "Out of Stock"
            lines.append(
                f"   {i}. **{p['name']}** -- Rs. {p['price']:,}  |  "
                f"Rating: {p['rating']}  |  {stock}"
            )
            lines.append(f"      {p['description']}")
            lines.append(f"      Product ID: `{p['product_id']}`")
    else:
        lines.append(
            "   I couldn't find any cheaper in-stock alternatives right now. "
            "Our catalogue is updated regularly -- please check back soon!"
        )

    return lines


def _format_product_detail(product: dict) -> list[str]:
    """Format a single product's details."""
    stock = "In Stock" if product["in_stock"] else "Out of Stock"
    return [
        f"Here are the details for **{product['name']}**:",
        f"   Price    : Rs. {product['price']:,}",
        f"   Rating   : {product['rating']} / 5",
        f"   Category : {product['category']}",
        f"   Stock    : {stock}",
        f"   About    : {product['description']}",
    ]


def _format_search_results(results: list[dict], query_label: str) -> list[str]:
    """Format product search results, or a helpful no-results message."""
    lines: list[str] = []

    if results:
        lines.append(f"Here's what I found for **\"{query_label}\"**:\n")
        for i, p in enumerate(results[:5], 1):
            stock = "In Stock" if p["in_stock"] else "Out of Stock"
            lines.append(
                f"   {i}. **{p['name']}** -- Rs. {p['price']:,}  |  "
                f"Rating: {p['rating']}  |  {stock}"
            )
            lines.append(f"      {p['description']}")
            lines.append(f"      Product ID: `{p['product_id']}`")
    else:
        lines.append(
            f"I searched for **\"{query_label}\"** but couldn't find any "
            f"matching products."
        )
        lines.append("   Here are a few things you can try:")
        lines.append(
            "     - Use different keywords "
            "(e.g. 'headphones' instead of 'earphones')"
        )
        lines.append("     - Browse by category: Footwear, Electronics, Clothing")
        lines.append("     - Contact our support team for personal recommendations")

    return lines


def _format_error_response(errors: list[str]) -> str:
    """Format an error response when no usable data was returned."""
    lines = ["I'm sorry, I wasn't able to complete your request:"]
    for error in errors:
        lines.append(f"  - {error}")
    lines.append(
        "\nPlease double-check your order or product ID and try again. "
        "If the issue persists, contact our support team."
    )
    return "\n".join(lines)


def _fallback_response() -> str:
    """Return a helpful fallback response when no intent was recognised."""
    return "\n".join([
        "I'm not sure how to help with that. Here are some things you can ask:",
        "  - 'Where is my order ORD-1002?'",
        "  - 'Is there a cheaper alternative to the shoes in my order ORD-1001?'",
        "  - 'Show me wireless headphones'",
        "  - 'Get details for PROD-201'",
    ])
