import os
import requests
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
HUBSPOT_API_URL = "https://api.hubapi.com/crm/v3/objects/tickets"

def create_hubspot_ticket(email, conversation_history, query, trigger_type):
    """Create a ticket in HubSpot with the conversation details."""
    headers = {
        "Authorization": f"Bearer {HUBSPOT_API_KEY}",
        "Content-Type": "application/json"
    }
    if trigger_type == 'frustrating':
        ticket_data = {
            "properties": {
                "subject": f"{email} : {query}",
                "content": f"User Email: {email}\n\nConversation History:\n{conversation_history}",
                "hs_pipeline": "0",
                "hs_pipeline_stage": "1",
                "hubspot_owner_id": "79298222",
                "hs_ticket_priority": "URGENT",
                "hs_ticket_category": "PRODUCT_ISSUE"
            }
        }
    else:
        ticket_data = {
            "properties": {
                "subject": f"{email} : {query}",
                "content": f"User Email: {email}\n\nConversation History:\n{conversation_history}",
                "hs_pipeline": "0",
                "hs_pipeline_stage": "1",
                "hubspot_owner_id": "79298222",
                "hs_ticket_priority": "URGENT",
                "hs_ticket_category": "BILLING_ISSUE"
            }
        }
    
    try:
        response = requests.post(HUBSPOT_API_URL, headers=headers, json=ticket_data)
        if response.status_code == 201:
            return True
        else:
            logger.error(f"HubSpot ticket creation failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error creating HubSpot ticket: {str(e)}")
        return False