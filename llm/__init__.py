"""
llm — LLM abstraction package for the AI Store Agent.

Public surface:
    from llm.base import AgentPlan, LLMProvider
    from llm.base import INTENT_ORDER_STATUS, INTENT_PRODUCT_SEARCH, ...

The GeminiProvider is not imported here to avoid a hard dependency on
google-generativeai. Import it directly only when you know it is available:
    from llm.gemini_provider import GeminiProvider
"""
