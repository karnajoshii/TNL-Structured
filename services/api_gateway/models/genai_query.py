from pydantic import BaseModel

class QueryRequest(BaseModel):
    session_id: str
    client_id: str
    query: str
    order_id: str | None = None

class SessionRequest(BaseModel):
    client_id: str

class ClearSessionRequest(BaseModel):
    session_id: str

class TicketRequest(BaseModel):
    email: str
    conversation_history: str
    query: str
    type: str