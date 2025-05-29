import uuid
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from ..database.db_utils import get_db_connection, execute_query, MYSQL_QUERY_CONFIG
from ..genai.llm_config import llm
from ..genai.prompt_templates import ORDER_ID_PROMPT, EMAIL_PROMPT

logger = logging.getLogger(__name__)

# Session context cache
session_context_cache = {}

def create_session(client_id: str) -> str:
    """Create a new session."""
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    if not conn:
        raise Exception("Database connection failed")
    
    try:
        query = """
            INSERT INTO chat_sessions (id, client_id, created_at, deleted)
            VALUES (%s, %s, %s, %s)
        """
        execute_query(conn, query, (session_id, client_id, datetime.now(), False), fetch=False)
        logger.info(f"Created new session: {session_id} for client: {client_id}")
        return session_id
    finally:
        if conn and conn.is_connected():
            conn.close()

def save_chat_message(session_id: str, role: str, message: str) -> bool:
    """Save a chat message to the database."""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to save chat message: No database connection")
        return False
    
    try:
        message_id = str(uuid.uuid4())
        query = """
            INSERT INTO chat_messages (id, chat_id, role, message)
            VALUES (%s, %s, %s, %s)
        """
        execute_query(conn, query, (message_id, session_id, role, message), fetch=False)
        logger.debug(f"Saved message for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def mark_session_as_deleted(session_id: str) -> bool:
    """Mark a session as deleted."""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to mark session as deleted: No database connection")
        return False
    
    try:
        query = "UPDATE chat_sessions SET deleted = TRUE, last_order_id = NULL WHERE id = %s"
        execute_query(conn, query, (session_id,), fetch=False)
        logger.info(f"Marked session {session_id} as deleted")
        
        if session_id in session_context_cache:
            del session_context_cache[session_id]
            
        return True
    except Exception as e:
        logger.error(f"Error marking session as deleted: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def retrieve_chat_history(session_id: str) -> Dict[str, Any]:
    """Retrieve chat history and context for a session."""
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed for chat history")
        raise Exception("Database connection failed")
    
    try:
        query = """
            SELECT cm.role, cm.message, cm.timestamp, cs.client_id
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.chat_id = cs.id
            WHERE cm.chat_id = %s AND cs.deleted = FALSE
            ORDER BY cm.timestamp ASC
        """
        messages = execute_query(conn, query, (session_id,), fetch=True)
        
        if not messages and not execute_query(conn, 
            "SELECT id FROM chat_sessions WHERE id = %s AND deleted = FALSE", 
            (session_id,), fetch=True):
            logger.warning(f"Session not found: {session_id}")
            raise Exception("Session not found or deleted")
        
        formatted_messages = [
            HumanMessage(content=msg["message"]) if msg["role"] == "user"
            else AIMessage(content=msg["message"])
            for msg in messages
        ]
        
        if session_id not in session_context_cache:
            session_context_cache[session_id] = {
                "order_ids": set(),
                "last_order_id": None,
                "email": None,
                "last_query_time": datetime.now(),
                "last_intent": None
            }
        
        logger.info(f"Retrieved chat history for session {session_id}")
        return {
            "messages": formatted_messages,
            "order_ids": session_context_cache[session_id]["order_ids"],
            "last_order_id": session_context_cache[session_id]["last_order_id"],
            "email": session_context_cache[session_id]["email"],
            "last_intent": session_context_cache[session_id]["last_intent"],
            "client_id": messages[0]["client_id"] if messages else None
        }
    except Exception as e:
        logger.error(f"Chat history retrieval error: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def format_chat_history_and_extract_order_id(session_id: str, query: str) -> Tuple[str, str]:
    """Format chat history and extract order ID using LLM."""
    try:
        context = retrieve_chat_history(session_id)
        messages = context["messages"]
        formatted_history = ""
        for msg in messages[-5:]:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\n"
        last_order_id = ""
        with get_db_connection() as conn:
            sql_query = """
                SELECT last_order_id FROM chat_sessions WHERE id = %s;
            """
            print(f"{conn}, {sql_query}, {(session_id,)}")
            result = execute_query(conn, sql_query, (session_id,), fetch=True)
            print(f"EXE : {result}")
            if result:
                last_order_id = result[0]['last_order_id'] or ""
        final_prompt = ORDER_ID_PROMPT.format(query=query, order_id=last_order_id)
        response = llm.invoke(final_prompt)
        order_id = response.content.strip()
        return formatted_history, order_id if order_id.startswith("ORD") else ""
    except Exception as e:
        logger.error(f"Error formatting history or extracting order ID: {e}")
        return "", ""

def update_session_context(session_id: str, intent: str, query: str, order_id: Optional[str] = None, email: Optional[str] = None, waiting_for: Optional[str] = None) -> None:
    """Update the session context."""
    if session_id not in session_context_cache:
        session_context_cache[session_id] = {
            "order_ids": set(),
            "last_order_id": None,
            "email": None,
            "last_query_time": datetime.now(),
            "last_intent": None,
            "waiting_for": None
        }
    
    context = session_context_cache[session_id]
    
    if order_id:
        context["order_ids"].add(order_id)
        context["last_order_id"] = order_id
    
    if email:
        context["email"] = email
    else:
        final_prompt = EMAIL_PROMPT.format(query=query)
        response = llm.invoke(final_prompt)
        email = response.content.strip()
        if email and "@" in email:
            context["email"] = email
    
    context["last_query_time"] = datetime.now()
    context["last_intent"] = intent
    context["waiting_for"] = waiting_for
    
    if context["last_order_id"]:
        conn = get_db_connection()
        if conn:
            try:
                query = "UPDATE chat_sessions SET last_order_id = %s WHERE id = %s"
                execute_query(conn, query, (context["last_order_id"], session_id), fetch=False)
                logger.debug(f"Updated last_order_id to {context['last_order_id']} for session {session_id}")
            except Exception as e:
                logger.error(f"Error updating last_order_id: {e}")
            finally:
                if conn and conn.is_connected():
                    conn.close()

def clean_old_contexts() -> None:
    """Clean up old session contexts."""
    now = datetime.now()
    expired_sessions = [
        session_id for session_id, context in session_context_cache.items()
        if now - context["last_query_time"] > timedelta(hours=2)
    ]
    
    for session_id in expired_sessions:
        del session_context_cache[session_id]
        logger.debug(f"Cleaned expired session context: {session_id}")