from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import re
import os
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
from ..genai.intent_classifier import intent_classifier, is_logistics_query
from ..genai.agent import chat_with_csv, chat_with_mysql, is_continuing_query,handle_reschedule_delivery, handle_address_change, handle_general_query, handle_capabilities_query, handle_small_talks, handle_frustration, handle_vip
from ..session.session_manager import create_session, retrieve_chat_history, mark_session_as_deleted, save_chat_message
from ..data_processing.csv_processor import process_csv, UPLOAD_FOLDER, FAISS_PATH
from ..database.db_utils import get_db_connection, execute_query, DB_CONFIG, MYSQL_QUERY_CONFIG
from ..crm_api.hubspot_adapter import create_hubspot_ticket
from .models.genai_query import QueryRequest, SessionRequest, ClearSessionRequest, TicketRequest

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)

@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        data = request.get_json()
        session_request = SessionRequest(**data)
        session_id = create_session(session_request.client_id)
        return jsonify({"status": "success", "session_id": session_id}), 201
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({"error": str(e), "error_code": "SESSION_CREATION_FAILED"}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload and process CSV file."""
    session_id = request.form.get('session_id')
    if not session_id:
        logger.warning("Missing session ID in CSV upload")
        return jsonify({"error": "Session ID is required", "error_code": "MISSING_SESSION_ID"}), 400
    
    if 'file' not in request.files:
        logger.warning("No file part in CSV upload request")
        return jsonify({"error": "No file part in the request", "error_code": "NO_FILE"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected in CSV upload")
        return jsonify({"error": "No file selected", "error_code": "NO_FILE_SELECTED"}), 400
    
    if file and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(file_path)
            process_csv(file_path)
            logger.info(f"Successfully processed CSV: {filename}")
            return jsonify({"message": "CSV processed successfully"}), 200
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return jsonify({"error": str(e), "error_code": "PROCESSING_FAILED"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        logger.warning("Invalid file format in CSV upload")
        return jsonify({"error": "Please upload a CSV file", "error_code": "INVALID_FILE"}), 400

@app.route('/query', methods=['POST'])
def query_data():
    try:
        data = request.get_json()
        query_request = QueryRequest(**data)
        session_id = query_request.session_id
        client_id = query_request.client_id
        user_input = query_request.query.strip()
        order_id = query_request.order_id

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}), 500
        try:
            query = "SELECT client_id FROM chat_sessions WHERE id = %s AND deleted = FALSE"
            result = execute_query(conn, query, (session_id,), fetch=True)
            if not result or result[0]['client_id'] != client_id:
                logger.warning(f"Invalid client_id for session: {session_id}")
                return jsonify({"error": "Invalid session or client ID", "error_code": "INVALID_SESSION"}), 400
        finally:
            if conn and conn.is_connected():
                conn.close()

        if not user_input:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty", "error_code": "EMPTY_QUERY"}), 400
        
        logger.info(f"Processing query: '{user_input}' for session: {session_id}")
        
        save_chat_message(session_id, 'user', user_input)
        intent = intent_classifier(user_input, session_id)
        logger.debug(f"Classified intent: {intent} for query: {user_input}")

        # if not is_logistics_query(user_input):
        #     result = {"response": "I'm sorry, I can only assist with transport and logistics queries. Please ask about orders or shipments."}
        # else:
        is_continuing_query(session_id, intent, user_input)

        chat_history = retrieve_chat_history(session_id)["messages"]
        if intent == "csv":
            result = chat_with_csv(session_id, user_input)
        elif intent == "mysql":
            result = chat_with_mysql(session_id, user_input, chat_history)
        elif intent == "reschedule_delivery":
            result = handle_reschedule_delivery(session_id, user_input, chat_history)
        elif intent == "address_change":
            result = handle_address_change(session_id, user_input, chat_history)
        elif intent == "general":
            result = handle_general_query(session_id, user_input)
        elif intent == "capabilities":
            result = handle_capabilities_query(session_id, user_input)
        elif intent == "small_talks":
            result = handle_small_talks(session_id, user_input)
        elif intent == "frustration":
            result = handle_frustration(session_id, user_input)
        elif intent == "vip":
            result = handle_vip(session_id, user_input)
        else:
            logger.warning(f"Unknown intent: {intent}")
            result = {"response": "I'm not sure how to handle that request. Please ask about orders or logistics."}
        
        if "error" in result:
            logger.error(f"Query processing error: {result['error']}")
            return jsonify({"error": result["error"], "error_code": result["error_code"]}), 500
        
        response = {"response": result["response"]}
        if "sql_query" in result:
            response.update({
                "sql_query": result["sql_query"],
                "sql_response": result["sql_response"]
            })
        if order_id:
            response["order_id"] = order_id
        
        logger.info(f"Query processed successfully for session {session_id}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "error_code": "UNEXPECTED_ERROR"}), 500

@app.route('/chat_history/<session_id>', methods=['GET'])
def get_chat_history_endpoint(session_id: str):
    """Retrieve chat history for a session."""
    try:
        history_data = retrieve_chat_history(session_id)
        formatted_messages = [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if hasattr(msg, 'created_at') else datetime.now().isoformat()
            }
            for msg in history_data["messages"]
        ]
        return jsonify({"messages": formatted_messages}), 200
    except Exception as e:
        logger.error(f"Chat history retrieval error: {e}")
        if str(e) == "Session not found or deleted":
            return jsonify({"error": "Session not found or deleted", "error_code": "INVALID_SESSION"}), 404
        return jsonify({"error": str(e), "error_code": "HISTORY_RETRIEVAL_FAILED"}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear chat history for a session."""
    try:
        data = request.get_json()
        clear_session_request = ClearSessionRequest(**data)
        if not mark_session_as_deleted(clear_session_request.session_id):
            logger.error(f"Failed to clear session: {clear_session_request.session_id}")
            return jsonify({"error": "Failed to clear session", "error_code": "SESSION_CLEAR_FAILED"}), 500
        
        logger.info(f"Session cleared: {clear_session_request.session_id}")
        return jsonify({"message": "Session cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        return jsonify({"error": str(e), "error_code": "SESSION_CLEAR_FAILED"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        db_status = False
        mysql_db_status = False
        
        conn = get_db_connection()
        if conn and conn.is_connected():
            db_status = True
            conn.close()
        
        mysql_conn = get_db_connection(config=MYSQL_QUERY_CONFIG)
        if mysql_conn and mysql_conn.is_connected():
            mysql_db_status = True
            mysql_conn.close()
        
        logger.info("Health check performed")
        return jsonify({
            "status": "healthy",
            "session_database": "connected" if db_status else "disconnected",
            "mysql_query_database": "connected" if mysql_db_status else "disconnected"
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": str(e), "error_code": "HEALTH_CHECK_FAILED"}), 500

@app.route('/api/create-ticket', methods=['POST'])
def create_ticket_endpoint():
    """
    API endpoint to create a HubSpot ticket.
    Expects JSON payload with 'email', 'conversation_history', and 'query'.
    """
    try:
        data = request.get_json()
        ticket_request = TicketRequest(**data)
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', ticket_request.email):
            return jsonify({
                "status": "error",
                "message": "Invalid email format"
            }), 400

        success = create_hubspot_ticket(
            ticket_request.email, 
            ticket_request.conversation_history, 
            ticket_request.query, 
            ticket_request.type
        )

        if success:
            return jsonify({
                "status": "success",
                "message": "Ticket created successfully in HubSpot"
            }), 201
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create ticket in HubSpot"
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500