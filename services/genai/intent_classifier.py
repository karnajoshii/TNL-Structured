import logging
from typing import Dict, Any
from .llm_config import vector_store, llm
from .prompt_templates import INTENT_CLASSIFIER_PROMPT, LOGISTICS_QUERY_PROMPT
from ..session.session_manager import session_context_cache, retrieve_chat_history, update_session_context,clean_old_contexts

# Set up logging
logger = logging.getLogger(__name__)

def intent_classifier(query: str, session_id: str) -> str:
    """Classify the intent of the query, returning only the intent string."""
    if len(session_context_cache) > 100:
        clean_old_contexts()

    results = vector_store.similarity_search_with_score(query, k=1)
    if results[0][1] > 0.8:  # Adjust threshold as needed
        return "csv"
    
    try:
        context = retrieve_chat_history(session_id)
        chat_history = context["messages"]
    except Exception as e:
        logger.warning(f"Failed to retrieve chat history: {e}")
        chat_history = []
        context = {
            "order_ids": set(),
            "last_order_id": None,
            "email": None,
            "last_intent": None,
            "waiting_for": None
        }
    
    valid_intents = {"csv", "mysql", "reschedule_delivery", "address_change", "general", "small_talks", "capabilities", "frustration", "vip"}
    
    try:
        kwargs = {
            "query": query,
            "order_ids": ", ".join(context["order_ids"]) if context["order_ids"] else "None",
            "last_intent": context["last_intent"] or "None",
            "waiting_for": context.get("waiting_for", "None")
        }
        logger.debug(f"Prompt kwargs: {kwargs}")
        
        final_prompt = INTENT_CLASSIFIER_PROMPT.format(**kwargs)
        logger.debug(f"Formatted prompt: {final_prompt}")
        
        response = llm.invoke(final_prompt)
        intent = response.content.strip()
        logger.debug(f"Raw LLM response for intent classification: {intent}")
        
        if intent not in valid_intents:
            logger.warning(f"Invalid intent returned: {intent}")
            return "general"
        
        update_session_context(session_id, intent, query)
        logger.info(f"Classified intent: {intent} for query: {query}")
        return intent
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "general"

def is_logistics_query(query: str) -> bool:
    """Check if the query is related to transport, logistics, or orders."""
    try:
        final_prompt = LOGISTICS_QUERY_PROMPT.format(query=query)
        response = llm.invoke(final_prompt)
        return response.content.strip().lower() == "relevant"
    except Exception as e:
        logger.error(f"Error checking query relevance: {e}")
        return False