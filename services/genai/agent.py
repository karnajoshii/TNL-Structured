import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage, HumanMessage
from .llm_config import llm, vector_store
from .prompt_templates import ORDER_ID_PROMPT, CSV_QUERY_PROMPT, MYSQL_QUERY_PROMPT, MYSQL_RESPONSE_PROMPT, DELIVERY_DATE_PROMPT, DELIVERY_ADDRESS_PROMPT, SMALL_TALK_PROMPT, CONTINUING_QUERY_PROMPT
from ..database.db_utils import get_db_connection, execute_query,MYSQL_QUERY_CONFIG
from ..session.session_manager import format_chat_history_and_extract_order_id, update_session_context, session_context_cache, retrieve_chat_history, save_chat_message
from langchain_community.utilities import sql_database
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import FAISS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../../Uploads")
FAISS_PATH = os.path.join(BASE_DIR, "data/faiss_index")
for directory in [UPLOAD_FOLDER, FAISS_PATH]:
    os.makedirs(directory, exist_ok=True)

logger = logging.getLogger(__name__)

def is_continuing_query(session_id: str, intent: str, query: str) -> bool:
    """Determine if the query continues an existing conversation."""
    if session_id not in session_context_cache:
        return False
    
    context = session_context_cache[session_id]
    last_intent = context.get("last_intent")
    
    if not last_intent:
        return False
    
    try:
        waiting_for = context.get("waiting_for", "")
        final_prompt = CONTINUING_QUERY_PROMPT.format(
            last_intent=last_intent,
            current_intent=intent,
            query=query,
            order_ids=", ".join(context["order_ids"]) if context["order_ids"] else "None",
            waiting_for=waiting_for or "None"
        )
        response = llm.invoke(final_prompt)
        final_response = response.content.strip().lower()
        print(f"FINAL : {final_response}")

        conn = get_db_connection(MYSQL_QUERY_CONFIG)
        if not conn:
            logger.error("Database connection failed for reschedule eligibility check")
            return {"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}
        
        try:
            if final_response == "false":
                query = "UPDATE chat_sessions SET last_order_id = NULL WHERE id = %s;"
                result = execute_query(conn, query, (session_id,), fetch=False)
                print(f"RES : {result}")
                conn.close()
                logger.info(f"last order is {session_id} removed")

        except Exception as e:
            logger.error(f"Error marking session as deleted: {e}")
            return response.content.strip().lower() == "true"            
    
    except Exception as e:
        logger.error(f"Error determining query type: {e}")
        return False

def chat_with_csv(session_id: str, query: str) -> Dict[str, Any]:
    """Handle CSV-based FAQ queries."""
    try:
        from .llm_config import llm, vector_store
        if vector_store is None:
            if os.path.exists(FAISS_PATH):
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
                vector_store = FAISS.load_local(
                    FAISS_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS index")
            else:
                logger.warning("No CSV data available")
                return {"error": "No CSV data uploaded", "error_code": "NO_DATA"}
        
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
        faiss_index = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        def retrieve_documents(query, k=5):
            """Retrieves top-k most relevant documents from FAISS."""
            docs = faiss_index.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant data found."
        
        context = retrieve_documents(query)
        final_prompt = CSV_QUERY_PROMPT.format(context=context, query=query)
        response = llm.invoke(final_prompt)
        response_text = response.content.strip()
        
        if not response_text:
            response_text = "I don't have enough information to answer that. Please provide more details or ask about something else."
        
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "csv", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"CSV query error: {e}")
        response = "An error occurred while processing your FAQ query. Please try again."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        return {"response": response}

def chat_with_mysql(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle MySQL database queries."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with order-specific queries.")]
    
    formatted_history, order_id = format_chat_history_and_extract_order_id(session_id, query)

    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "mysql", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        db_uri = f"mysql+mysqlconnector://{MYSQL_QUERY_CONFIG['user']}:{MYSQL_QUERY_CONFIG['password']}@{MYSQL_QUERY_CONFIG['host']}:{MYSQL_QUERY_CONFIG['port']}/{MYSQL_QUERY_CONFIG['database']}"
        db = sql_database.SQLDatabase.from_uri(db_uri)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return {"error": f"Database connection failed: {str(e)}", "error_code": "DB_CONNECTION_FAILED"}
    
    def get_sql(schema: str, formatted_history: str, query: str, order_id: Optional[str]) -> str:
        """Generate SQL query."""
        context = session_context_cache.get(session_id, {})
        email = context.get("email")
        
        if not order_id and not email:
            return "Could you please share your valid order ID, so I can check the details for you?"
        
        context_info = f"Order ID: {order_id}\n" if order_id else f"Email: {email}\n"
        try:
            final_prompt = MYSQL_QUERY_PROMPT.format(
                query=query,
                schema=schema,
                history=formatted_history,
                context_info=context_info
            )
            response = llm.invoke(final_prompt)
            response_sql = response.content.strip()
            if "invoice" in response_sql:
                return f"SELECT invoice_url FROM orders WHERE order_id = '{order_id}';"
            elif "shipment":
                return f"SELECT customer_name, email, shipment_status, expected_delivery, delivery_address FROM orders WHERE order_id = '{order_id}';"
            else:
                return f"SELECT * FROM orders WHERE order_id = '{order_id}';"
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return f"Error generating SQL query: {str(e)}"
    
    def get_response(schema: str, formatted_history: str, query: str, sql_query: str, sql_response: str) -> str:
        """Generate natural language response."""
        try:
            final_prompt = MYSQL_RESPONSE_PROMPT.format(
                query=query,
                schema=schema,
                history=formatted_history,
                sql_query=sql_query,
                sql_response=sql_response
            )
            response = llm.invoke(final_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "An error occurred while generating the response."
    
    try:
        chat_history.append(HumanMessage(content=query))
        sql_query = get_sql(db.get_table_info(), formatted_history, query, order_id)
        
        if sql_query.startswith("Please provide"):
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', sql_query)
            update_session_context(session_id, "mysql", query, order_id)
            return {"response": sql_query}
        
        try:
            sql_response = db.run(sql_query)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            response = "Sorry, I encountered an error. Please try again or refine your question."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response, "sql_query": sql_query, "sql_response": str(e)}
        
        natural_language_response = get_response(db.get_table_info(), formatted_history, query, sql_query, sql_response)
        
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', natural_language_response)
        update_session_context(session_id, "mysql", query, order_id)
        
        return {
            "response": natural_language_response,
            "sql_query": sql_query,
            "sql_response": sql_response
        }
    except Exception as e:
        logger.error(f"Unexpected error in MySQL query: {e}")
        response = "An unexpected error occurred. Please try again or contact support."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        return {"response": response}

def extract_delivery_date(session_id: str, query: str) -> str:
    """Extract delivery date from current query or recent chat history using LLM."""
    try:
        current_time = datetime.now()
        current_year = current_time.year
        today_str = current_time.strftime('%Y-%m-%d')
        tomorrow_str = (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
        
        final_prompt = DELIVERY_DATE_PROMPT.format(
            query=query,
            current_year=current_year,
            today=today_str,
            tomorrow=tomorrow_str
        )
        response = llm.invoke(final_prompt)
        date_str = response.content.strip()
        
        return date_str
    except Exception as e:
        logger.error(f"Error extracting delivery date: {e}")
        return ""

def handle_reschedule_delivery(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle delivery rescheduling requests."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with rescheduling your delivery.")]
    
    chat_history, order_id = format_chat_history_and_extract_order_id(session_id, query)
    
    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "reschedule_delivery", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        conn = get_db_connection(MYSQL_QUERY_CONFIG)
        if not conn:
            logger.error("Database connection failed for reschedule eligibility check")
            return {"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}
        
        query_check = "SELECT reschedule_eligible, expected_delivery, shipment_status FROM orders WHERE order_id = %s"
        result = execute_query(conn, query_check, (order_id,), fetch=True)
        
        if not result:
            response = f"Order {order_id} not found. Please verify the order ID and try again."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response}
        
        order_details = result[0]
        
        if not order_details['reschedule_eligible']:
            response = f"Order {order_id} can no longer be rescheduled.\n If you need further assistance, please contact our support team."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id)
            return {"response": response}
        
        date_str = extract_delivery_date(session_id, query)
        
        if not date_str or date_str == '""':
            current_date = order_details['expected_delivery']
            response = f"Please provide the new delivery date for order {order_id} (current date: {current_date})."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
            return {"response": response}
        
        try:
            new_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if new_date <= datetime.now().date():
                response = "I’m sorry, but rescheduling is only possible for future dates. Could you please provide a valid future date?"
                # save_chat_message(session_id, 'user', query)
                save_chat_message(session_id, 'assistant', response)
                update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
                return {"response": response}
            
            if (new_date - datetime.now().date()).days > 30:
                response = "To ensure timely processing, rescheduling is limited to dates within the next 30 days. Please choose a date within that range."
                # save_chat_message(session_id, 'user', query)
                save_chat_message(session_id, 'assistant', response)
                update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
                return {"response": response}
            
            update_query = "UPDATE orders SET expected_delivery = %s WHERE order_id = %s"
            execute_query(conn, update_query, (new_date, order_id), fetch=False)
            
            response = f"The delivery for Order {order_id} has been rescheduled to {new_date}. \n Is there anything else I can help you with?"
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for=None)
            return {"response": response}
        except ValueError:
            response = f"Please provide the new delivery date for order {order_id} in a valid format (e.g., '2025-05-20' or 'tomorrow')."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
            return {"response": response}
    except Exception as e:
        logger.error(f"Error in reschedule delivery: {e}")
        response = "An error occurred while processing your request. Please try again or contact support."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "reschedule_delivery", query, order_id)
        return {"response": response}
    finally:
        if conn and conn.is_connected():
            conn.close()

def extract_delivery_address(session_id: str, query: str) -> str:
    """Extract delivery address from current query or recent chat history using LLM."""
    try:
        final_prompt = DELIVERY_ADDRESS_PROMPT.format(query=query)
        response = llm.invoke(final_prompt)
        address = response.content.strip()
        
        if address and (len(address) < 10 or not any(char.isdigit() for char in address)):
            address = ""
        
        return address
    except Exception as e:
        logger.error(f"Error extracting delivery address: {e}")
        return ""

def handle_address_change(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle address change requests."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with changing your delivery address.")]
    
    chat_history, order_id = format_chat_history_and_extract_order_id(session_id, query)
    
    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        conn = get_db_connection(MYSQL_QUERY_CONFIG)
        if not conn:
            logger.error("Database connection failed for address change eligibility check")
            return {"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}
        
        query_check = "SELECT address_change_eligible, delivery_address, shipment_status FROM orders WHERE order_id = %s"
        result = execute_query(conn, query_check, (order_id,), fetch=True)
        
        if not result:
            response = f"Order {order_id} not found. Please verify the order ID and try again."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response}
        
        order_details = result[0]
        
        if not order_details['address_change_eligible']:
            response = f"Order {order_id} isn’t eligible for an address change at this stage.\n If you need further assistance, please contact our support team."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "address_change", query, order_id)
            return {"response": response}
        
        new_address = extract_delivery_address(session_id, query)
        
        if not new_address:
            current_address = order_details['delivery_address']
            response = f"Please share the new delivery address for Order {order_id}. The current address on record is: {current_address}."
            # save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "address_change", query, order_id, waiting_for="address")
            return {"response": response}
        
        update_query = "UPDATE orders SET delivery_address = %s WHERE order_id = %s"
        execute_query(conn, update_query, (new_address, order_id), fetch=False)
        
        response = f"The address for {order_id} has been updated to:\n  {new_address}. \n Is there anything else I can help you with?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, order_id, waiting_for=None)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in address change: {e}")
        response = "An error occurred while processing your request. Please try again or contact support."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, order_id)
        return {"response": response}
    finally:
        if conn and conn.is_connected():
            conn.close()

def handle_general_query(session_id: str, query: str) -> Dict[str, str]:
    """Handle general greetings or unrelated queries."""
    normalized_query = query.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you", "who are you"]
    
    if any(greeting in normalized_query for greeting in greetings):
        response = """
Hi! I’m **AIRA**.\n
I can help you with the following:\n
- Track your shipment\n
- Reschedule a delivery\n
- Update your address\n
- Get your invoice\n
- Answer common questions\n
\n
Just tell me what you’d like help with!
"""
    else:
        response = """
Hi! I’m **AIRA**.\n
I can help you with the following:\n
- Track your shipment\n
- Reschedule a delivery\n
- Update your address\n
- Get your invoice\n
- Answer common questions\n
\n
Just tell me what you’d like help with!
"""
    
    # save_chat_message(session_id, 'user', query)
    save_chat_message(session_id, 'assistant', response)
    update_session_context(session_id, "general", query)
    return {"response": response}

def handle_capabilities_query(session_id: str, query: str) -> Dict[str, str]:
    """Handle queries about assistant capabilities."""
    response = """
I can help you with the following:\n
- Track your shipment\n
- Reschedule a delivery\n
- Update your address\n
- Get your invoice\n
- Answer common questions\n
\n
Just tell me what you’d like help with!
"""
    
    # save_chat_message(session_id, 'user', query)
    save_chat_message(session_id, 'assistant', response)
    update_session_context(session_id, "capabilities", query)
    return {"response": response}

def handle_small_talks(session_id: str, query: str) -> Dict[str, str]:
    """Handle small talk queries like 'How are you', 'Great', 'Thanks', 'Good morning' using LLM."""
    try:
        final_prompt = SMALL_TALK_PROMPT.format(query=query)
        response = llm.invoke(final_prompt)
        response_text = response.content.strip()

        if not response_text:
            response_text = "Nice to chat! How can I assist with your logistics needs?"

        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "small_talks", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error handling small talk query: {e}")
        response_text = "Nice to chat! How can I assist with your logistics needs?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "small_talks", query)
        return {"response": response_text}

def handle_frustration(session_id: str, query: str) -> Dict[str, str]:
    """Handle frustration queries."""
    try:
        response_text = "I’m really sorry you're facing this. I completely understand how frustrating it can be.\nLet me help by creating a support ticket so our team can review and get back to you as soon as possible."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "frustration", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error handling frustration query: {e}")
        response_text = "Nice to chat! How can I assist with your logistics needs?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "frustration", query)
        return {"response": response_text}

def handle_vip(session_id: str, query: str) -> Dict[str, str]:
    """Handle VIP queries."""
    try:
        response_text = "Thank you for your interest in shipping with us.\nI’ve flagged this as a priority inquiry. Our sales team will connect with you shortly to help you explore the best options."
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "vip", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error handling vip query: {e}")
        response_text = "Nice to chat! How can I assist with your logistics needs?"
        # save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "vip", query)
        return {"response": response_text}