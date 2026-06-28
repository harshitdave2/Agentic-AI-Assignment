"""
tools/store_tools.py — Tool implementations and mock data store.

These functions represent what would normally be API calls to a real backend
(e.g. a REST API, SQL database, or third-party logistics service).

The tool signatures are the stable interface:
  get_order(order_id)       → dict
  search_products(query)    → list[dict]
  get_product(product_id)   → dict

Swapping the mock data for real API calls only requires modifying the
internals of these three functions — the agent logic never needs to change.

Error convention
────────────────
On failure, tools return a dict with a single "error" key:
  {"error": "human-readable explanation"}

The executor checks for this key and records it in ExecutionContext.errors.
Tools never raise exceptions for expected failure cases (not found, etc.).
"""

from __future__ import annotations

import logging
from typing import Union

logger = logging.getLogger("agent.tools")

# ─── Mock Database ────────────────────────────────────────────────────────────
#
# In a production system, these dicts would be replaced with database queries.
# The tool function signatures remain identical — no changes to agent logic.

ORDERS_DB: dict[str, dict] = {
    "ORD-1001": {
        "order_id":           "ORD-1001",
        "customer":           "Rahul Sharma",
        "status":             "Delivered",
        "estimated_delivery": "2025-06-10",
        "delivered_on":       "2025-06-09",
        "items": [
            {"product_id": "PROD-201", "name": "Nike Air Max 270", "qty": 1, "price": 8999}
        ],
        "total":       8999,
        "tracking_url": "https://track.store.com/ORD-1001",
    },
    "ORD-1002": {
        "order_id":           "ORD-1002",
        "customer":           "Priya Mehta",
        "status":             "In Transit",
        "estimated_delivery": "2025-06-26",
        "items": [
            {"product_id": "PROD-305", "name": "Sony WH-1000XM5 Headphones", "qty": 1, "price": 24999}
        ],
        "total":       24999,
        "tracking_url": "https://track.store.com/ORD-1002",
    },
    "ORD-1003": {
        "order_id":           "ORD-1003",
        "customer":           "Amit Verma",
        "status":             "Processing",
        "estimated_delivery": "2025-06-28",
        "items": [
            {"product_id": "PROD-102", "name": "Levi's 511 Slim Jeans", "qty": 2, "price": 3499}
        ],
        "total":       6998,
        "tracking_url": "https://track.store.com/ORD-1003",
    },
    "ORD-1004": {
        "order_id": "ORD-1004",
        "customer": "Sneha Gupta",
        "status":   "Cancelled",
        "items": [
            {"product_id": "PROD-410", "name": "Apple AirPods Pro", "qty": 1, "price": 19999}
        ],
        "total": 19999,
    },
}

PRODUCTS_DB: dict[str, dict] = {
    "PROD-101": {
        "product_id":  "PROD-101",
        "name":        "Campus OG Running Shoes",
        "category":    "Footwear",
        "price":       1299,
        "rating":      4.1,
        "in_stock":    True,
        "description": "Lightweight running shoes with cushioned sole.",
    },
    "PROD-102": {
        "product_id":  "PROD-102",
        "name":        "Levi's 511 Slim Jeans",
        "category":    "Clothing",
        "price":       3499,
        "rating":      4.4,
        "in_stock":    True,
        "description": "Classic slim fit jeans with stretch fabric.",
    },
    "PROD-103": {
        "product_id":  "PROD-103",
        "name":        "Puma Softride Shoes",
        "category":    "Footwear",
        "price":       2799,
        "rating":      4.2,
        "in_stock":    True,
        "description": "Soft, comfortable everyday running shoes.",
    },
    "PROD-104": {
        "product_id":  "PROD-104",
        "name":        "Adidas Ultraboost 22",
        "category":    "Footwear",
        "price":       12999,
        "rating":      4.6,
        "in_stock":    False,
        "description": "Premium running shoes with Boost cushioning.",
    },
    "PROD-201": {
        "product_id":  "PROD-201",
        "name":        "Nike Air Max 270",
        "category":    "Footwear",
        "price":       8999,
        "rating":      4.5,
        "in_stock":    True,
        "description": "Iconic Air Max design with 270-degree air unit.",
    },
    "PROD-202": {
        "product_id":  "PROD-202",
        "name":        "Bata Casual Sneakers",
        "category":    "Footwear",
        "price":       1599,
        "rating":      3.9,
        "in_stock":    True,
        "description": "Affordable everyday casual sneakers.",
    },
    "PROD-301": {
        "product_id":  "PROD-301",
        "name":        "boAt Rockerz 450 Headphones",
        "category":    "Electronics",
        "price":       1499,
        "rating":      4.0,
        "in_stock":    True,
        "description": "Wireless headphones with 15 hours battery.",
    },
    "PROD-302": {
        "product_id":  "PROD-302",
        "name":        "JBL Tune 510BT",
        "category":    "Electronics",
        "price":       2999,
        "rating":      4.2,
        "in_stock":    True,
        "description": "On-ear wireless headphones, 40 hrs battery.",
    },
    "PROD-303": {
        "product_id":  "PROD-303",
        "name":        "Noise Cancelling Headset Pro",
        "category":    "Electronics",
        "price":       5499,
        "rating":      4.3,
        "in_stock":    True,
        "description": "Active noise cancelling over-ear headphones.",
    },
    "PROD-304": {
        "product_id":  "PROD-304",
        "name":        "OnePlus Bullets Wireless Z2",
        "category":    "Electronics",
        "price":       1999,
        "rating":      4.1,
        "in_stock":    True,
        "description": "In-ear wireless earbuds with bass boost.",
    },
    "PROD-305": {
        "product_id":  "PROD-305",
        "name":        "Sony WH-1000XM5 Headphones",
        "category":    "Electronics",
        "price":       24999,
        "rating":      4.8,
        "in_stock":    True,
        "description": "Industry-leading noise cancellation headphones.",
    },
    "PROD-410": {
        "product_id":  "PROD-410",
        "name":        "Apple AirPods Pro",
        "category":    "Electronics",
        "price":       19999,
        "rating":      4.7,
        "in_stock":    True,
        "description": "Premium wireless earbuds with ANC and transparency mode.",
    },
}


# ─── Tool Functions ───────────────────────────────────────────────────────────

def get_order(order_id: str) -> dict:
    """
    Fetch full order details by order ID.

    Parameters
    ──────────
    order_id : str
        The order identifier (e.g. "ORD-1002"). Case-insensitive.

    Returns
    ───────
    dict
        Full order record on success.
        {"error": "<message>"} if the order is not found.
    """
    order_id = order_id.strip().upper()
    logger.info(f"[TOOL] get_order(order_id={order_id!r})")

    order = ORDERS_DB.get(order_id)
    if not order:
        msg = f"No order found with ID '{order_id}'. Please check the order ID and try again."
        logger.warning(f"[TOOL] get_order -> not found: {order_id!r}")
        return {"error": msg}

    logger.info(f"[TOOL] get_order -> {order['order_id']} | status={order['status']!r}")
    return order


def search_products(query: str) -> list[dict]:
    """
    Search the product catalog by keyword.

    Performs a case-insensitive substring match against product name,
    category, and description. Results are sorted by rating (highest first).

    Parameters
    ──────────
    query : str
        Search term(s). E.g. "headphones", "running shoes", "Footwear".

    Returns
    ───────
    list[dict]
        Matching products sorted by rating descending.
        Empty list if no products match.
    """
    query_lower = query.lower()
    logger.info(f"[TOOL] search_products(query={query!r})")

    results = [
        product
        for product in PRODUCTS_DB.values()
        if (
            query_lower in product["name"].lower()
            or query_lower in product["category"].lower()
            or query_lower in product["description"].lower()
        )
    ]
    results.sort(key=lambda x: x["rating"], reverse=True)

    logger.info(f"[TOOL] search_products -> {len(results)} result(s) for {query!r}")
    return results


def get_product(product_id: str) -> dict:
    """
    Fetch a single product's full details by product ID.

    Parameters
    ──────────
    product_id : str
        The product identifier (e.g. "PROD-201"). Case-insensitive.

    Returns
    ───────
    dict
        Full product record on success.
        {"error": "<message>"} if the product is not found.
    """
    product_id = product_id.strip().upper()
    logger.info(f"[TOOL] get_product(product_id={product_id!r})")

    product = PRODUCTS_DB.get(product_id)
    if not product:
        logger.warning(f"[TOOL] get_product -> not found: {product_id!r}")
        return {"error": f"No product found with ID '{product_id}'."}

    logger.info(f"[TOOL] get_product -> {product['name']} | Rs. {product['price']}")
    return product
