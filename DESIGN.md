# Design Decisions — AI Store Agent

**Author:** Agentic AI Assignment Submission  
**Project:** Customer Support AI Agent for Online Store

---

## 1. Architecture Overview

The agent is designed around a **4-stage pipeline**:

```
Question → Intent Detection → Tool Planning → Tool Execution → Response Builder
```

Each stage is a separate function with a single responsibility. This makes the system easy to test, debug, and extend — for example, replacing the mock database with a real API only requires changing `store_tools.py`.

---

## 2. Intent Detection Strategy

**Decision:** Use regex + keyword matching instead of an LLM for intent detection.

**Rationale:**
- Customer questions for an online store are highly structured and predictable
- Order IDs follow a strict pattern (`ORD-XXXX`) that regex handles perfectly
- Keyword lists for intents like "cheaper", "alternative", "where is" are small and reliable
- This makes the agent **zero-dependency** (no API keys, no external calls) and **instant** — no LLM latency

**Trade-offs:**
- Won't handle very ambiguous or creative phrasings as well as an LLM would
- Mitigated by a catch-all fallback that prompts the user with example questions

---

## 3. Tool Chaining Design

**Decision:** Use a sequential `execute_plan()` function that resolves dependencies between tool calls.

**How it works for the "cheaper alternative" query:**
1. Detect order ID + "cheaper" intent
2. Call `get_order(order_id)` → get the first item's `product_id`
3. Call `get_product(product_id)` → get the product's category and price
4. Call `search_products(category)` → filter results cheaper than current product
5. Pass all results to `build_response()`

**Why not parallel calls?** The tools have data dependencies: you can't search for alternatives until you know what product was ordered. Sequential execution naturally handles this.

---

## 4. Error Handling Philosophy

**Decision:** Fail gracefully with informative, non-technical messages.

**Principles applied:**
- Never crash — every tool call is wrapped in a conditional check
- Never fabricate data — if a product or order isn't found, we say so directly
- Empty search results return a helpful message, not silence
- Error messages tell the user what to do next (re-check ID, contact support)

**Example:**
```
"No order found with ID 'ORD-9999'. Please check the order ID and try again."
```
Rather than: `KeyError: 'ORD-9999'` or worse, hallucinating an order status.

---

## 5. Response Design

**Decision:** Build a dedicated `build_response()` layer separate from tool execution.

**Rationale:** Mixing tool logic with presentation logic makes both harder to change. By keeping them separate:
- Tools return raw data (dicts/lists)
- The response builder formats them into customer-friendly text
- Changing the response format doesn't touch any tool code
- The builder uses emojis, currency formatting (₹), and stock labels to improve UX

---

## 6. Data Layer

**Decision:** Use an in-memory mock database (`ORDERS_DB`, `PRODUCTS_DB`) in `store_tools.py`.

**Rationale:**
- Allows the assignment to run standalone without any external services
- The tool function signatures (`get_order`, `search_products`, `get_product`) are the stable interface
- In production, you would only replace the internals of these functions (e.g., with a SQL query or API call), while the agent logic remains unchanged

---

## 7. Logging

**Decision:** Log every tool call and result using Python's `logging` module, writing to both console and `logs/agent.log`.

**Format:**
```
2025-06-24 10:32:01 | INFO | agent.tools | [TOOL CALL] get_order(order_id='ORD-1002')
2025-06-24 10:32:01 | INFO | agent.tools | [TOOL RESULT] Found order: ORD-1002 | Status: In Transit
```

This satisfies the bonus logging requirement and makes debugging easy — you can trace exactly which tools were called for any question.

---

## 8. What I Would Add Next (with more time)

- **LLM integration** (Claude/Gemini): replace keyword matching with LLM-based intent classification for handling freeform questions
- **Real database**: connect to PostgreSQL or Firestore instead of the in-memory dict
- **Authentication**: verify customer identity before showing order details
- **Price range filtering**: "headphones under ₹2000" (would need NLP or LLM to extract the price constraint)
- **Multi-item orders**: currently only looks at the first item for alternatives
