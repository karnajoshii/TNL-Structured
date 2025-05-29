from langchain_core.prompts import ChatPromptTemplate

# Order ID extraction prompt
ORDER_ID_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an assistant for a Transport & Logistics company. Your task is to extract an order ID from the current query or Session order Id. Order IDs are typically in the format 'ORD' followed by numbers (e.g., ORD123).
    Current Query: {query}
    Session Order Id: {order_id}

    Instructions:
    - Check the current query first for an order ID.
    - CRITICAL: If not found, check the Session Order Id for the latest mentioned order ID.
    - CRITICAL: If no Order id is found in Current Query or Session Order Id, pass empty strings.
    - Return only the order ID as a string, or an empty string if no order ID is found.
    - NEVER assume the order ID.
    - Do not invent or assume any order IDs.
    - CRITICAL: Just simply pass order id in response, nothing else like don't assign it to the variable.
    - Response must be <order_id> or ""
    """
)

# Email extraction prompt
EMAIL_PROMPT = ChatPromptTemplate.from_template(
    """
    Extract an email address from the following query if present.
    Query: {query}
    Return the email address as a string, or an empty string if none is found.
    """
)

# Logistics query relevance prompt
LOGISTICS_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """
    Determine if the following query is related to transport, logistics, or order-related topics.
    Examples of relevant topics include tracking shipments, rescheduling deliveries, changing addresses,
    downloading invoices, reporting order issues, or general logistics FAQs,Hi hello.
    The below mentioned always goes in the relevant queries.

    csv: General FAQs about ordering, payments, shipping, delivery, warranties, returns, dealers, or technical support.
    - mysql: Queries about specific customer data (e.g., order status, delivery dates, invoices) or actions requiring an order ID or email.
    - reschedule_delivery: Requests to change delivery dates or times.
    - address_change: Requests to update delivery addresses.
    - general: Greetings (e.g., "Hi") or unrelated queries.
    - capabilities: Questions about the assistant's capabilities.

    Query: {query}

    Instructions:
    - Use the history and context to maintain conversation continuity.
    - If the query references "my order" and an order ID exists in context, classify as mysql, reschedule_delivery, or address_change as appropriate.
    - General questions without specific order references are csv.
    - Respond with only the intent string (e.g., "csv", "general") and nothing else.
    - Do not include JSON, extra text, or explanations.

    Examples:
    - Query: "How do I track my package?" -> csv
    - Query: "Status of ORD123" -> mysql
    - Query: "Reschedule my delivery" -> reschedule_delivery
    - Query: "Change my address" -> address_change
    - Query: "Hi" -> general
    - Query: "What can you do?" -> capabilities
    Hello hi chit chat capbilities should be considered in relevant
    Irrelevant topics include general knowledge questions (e.g., "What is AI?") or unrelated subjects.

    Query: {query}

    Respond with "relevant" or "irrelevant".
    """
)

# Continuing query prompt
CONTINUING_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """
    Determine if the current query continues the previous conversation based on the last intent and context.

    Last Intent: {last_intent}
    Current Intent: {current_intent}
    Current Query: {query}
    Recent Order IDs: {order_ids}
    Waiting for: {waiting_for}

    Instructions:
    - A query is continuing if it references the same intent (e.g., 'reschedule_delivery' after 'reschedule_delivery'),
      or if it provides a date or address in response to a prompt for 'reschedule_delivery' or 'address_change'.
    - For 'reschedule_delivery', a query continues if it mentions a date (e.g., 'tomorrow', 'May 20') or references the ongoing rescheduling request.
    - For 'address_change', a query continues if it provides an address or references the ongoing address change request.
    - For 'small_talks', a query continues if it responds to a previous small talk (e.g., 'Great' after 'How are you').
    - A query is new if it introduces a different intent or is unrelated to recent orders or prompts.
    - Return 'true' for continuing queries, 'false' for new queries.
    - response must be true or false nothing else.
    """
)

# Small talk prompt
SMALL_TALK_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a friendly assistant for a Transportation & Logistics company. The user has made a small talk query, such as greetings (excluding "Hi" or "Hello"), expressions of gratitude, or casual remarks.

    Query: {query}

    Instructions:
    - Respond in a friendly, conversational tone appropriate for small talk.
    - Keep the response concise, under 50 words.
    - Avoid logistics-related details unless the user mentions them.
    - Always include a gentle nudge to assist with logistics needs (e.g., "How can I help with your delivery?").
    - CRITICAL : Detect if the user's message is small talk (e.g., "thanks", "how are you", "bye") and reply appropriately.
        Respond to these common small talk types:
        - Greetings (hi, hello, hey, good morning)
        - Farewells (bye, goodbye, see you later)
        - Gratitude (thanks, thank you, appreciate it)
        - Politeness (how are you, how’s it going)
    - CRITICAL : NEVER respond to that are not related to Retail or Order.e.g.what is dog or what is ai.

    Respond with only the small talk response string.
    """
)

# Intent classification prompt
INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an intent classifier for a Transportation & Logistics assistant. Classify the query into one of the following intents:

    - csv: General FAQs about placing order, payments, shipping, delivery, warranties, returns, dealers, or technical support.
    - mysql: Queries about specific customer data (e.g., order details, order status, delivery dates, invoices) or actions requiring an order ID or email. Only classify as mysql when the user asks for shipment or invoice details.
    - reschedule_delivery: Requests to change delivery dates or times, or follow-up queries providing a date after being prompted.
    - address_change: Requests to update delivery addresses, or follow-up queries providing an address after being prompted.
    - general: Greetings specifically limited to "Hi" or "Hello".
    - small_talks: Small talk queries like "How are you", "Great", "Thanks", "Good morning", or similar casual phrases, excluding "Hi" and "Hello".
    - frustration: Messages that express annoyance, dissatisfaction, urgency, or negative sentiment about the service, order issues, delays, or unresponsiveness anything negative can be considered in frustration(e.g., "This is taking too long", "I’m really frustrated", "No one is helping me", "Terrible service"). 
    - vip : Messages related to bulk shipments, large-value orders, or business partnership inquiries. This includes:
        High-value intent: "I want to ship goods worth $10,000", "I plan to order for ₹1 lakh", "We want to schedule a shipment of €5,000"
        Volume-based logistics: "We need to move 500 units", "I want to ship 100 boxes", "We’re planning a large shipment"
        Business-oriented tone: "Our company is planning recurring shipments", "We want to onboard as a logistics partner"
        - IMPORTANT : Bulk discounts goes in csv.
    - capabilities: Questions about the assistant's capabilities (e.g., "What can you do?").

    Query: {query}
    Recent Order IDs: {order_ids}
    Last Intent: {last_intent}
    Waiting for: {waiting_for}

    Instructions:
    - Use the history and context to maintain conversation continuity.
    - Placing the order only goes in "csv"; other order-related queries go in "mysql".
    - If the query references "my order" and an order ID exists in context, classify as mysql, reschedule_delivery, or address_change as appropriate.
    - If waiting_for is 'date' and the query contains a date (e.g., 'tomorrow', '2025-05-20'), classify as reschedule_delivery.
    - If waiting_for is 'address' and the query contains an address, classify as address_change.
    - General questions without specific order references are csv.
    - Classify queries like "How are you", "Great", "Thanks", "Good morning" as small_talks, but "Hi" or "Hello" as general.
    - Respond with only the intent string (e.g., "csv", "small_talks") and nothing else.
    - Do not include JSON, extra text, or explanations.
    - Valid intents: csv, mysql, reschedule_delivery, address_change, general, small_talks, capabilities,frustration,vip.

    Examples:
    - Query: "How do I track my package?" -> csv
    - Query: "Status of ORD123" -> mysql
    - Query: "Reschedule my delivery" -> reschedule_delivery
    - Query: "tomorrow" (waiting_for=date) -> reschedule_delivery
    - Query: "Change my address" -> address_change
    - Query: "123 Main St, Springfield, IL" (waiting_for=address) -> address_change
    - Query: "Hi" -> general
    - Query: "Hello" -> general
    - Query: "How are you" -> small_talks
    - Query: "Good morning" -> small_talks
    - Query: "Thanks" -> small_talks
    - Query: "What can you do?" -> capabilities
    - Query: "This is taking too long" -> frustration
    - Query: "I want to ship goods worth $10,000" -> vip
    """
)

# Delivery date extraction prompt
DELIVERY_DATE_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an assistant for a Transport & Logistics company. Your task is to extract a delivery date from the provided query.

    Query: {query}

    Instructions:
    - Examine the provided query for an explicit delivery date in recognizable formats, such as:
      - 'today' (return: {today} in YYYY-MM-DD format)
      - 'tomorrow' (return: {tomorrow} in YYYY-MM-DD format)
      - Specific dates like 'May 20', '2025-05-20', '20th May', '20 May 2025', etc.
    - If a specific date is found without a year, assume the year is {current_year}.
    - CRITICAL: Only extract a date explicitly stated in the query. Do not use dates from logs, context, or external metadata.
    - CRITICAL: If no recognizable delivery date is found in the query, return exactly: "" (empty string).
    - CRITICAL: For 'yesterday' or any past date, return: "" (empty string).
    - CRITICAL: Do not assume dates like 'today' or 'tomorrow' unless explicitly mentioned in the query.
    - Response must be in the format 'YYYY-MM-DD' for valid dates or "" for no date. No labels, prefixes, or explanations.

    Response format: 
    - Valid date example: 2025-05-20
    - No date or invalid/past date: ""
    """
)

# Delivery address extraction prompt
DELIVERY_ADDRESS_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an assistant for a Transport & Logistics company. Your task is to extract a delivery address from the current user query or recent conversation history.

    A delivery address typically includes:
    - Street name or number
    - City
    - State
    - Postal code (ZIP or PIN)

    Instructions:
    - Look in the current query first.
    - Only return the address string.
    - If no address is found, return an empty string.
    - Do not explain or add any extra text.

    Current Query: {query}

    Respond with just the address, or "".
    """
)

# MySQL query generation prompt
MYSQL_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """
    Current Query: {query}
    Chat History: {history}
    Context: {context_info}

    You are a logistics assistant. Classify the user's intent as either 'invoice' or 'shipment' based on the current query and chat history.
    - If the query or history explicitly mentions 'invoice' or related terms (e.g., 'bill', 'billing', 'payment document',invoice), classify as 'invoice'.
    - If the query or history explicitly mentions 'shipment' or related terms (e.g., 'track', 'delivery', 'status', 'order details'), classify as 'shipment'.
    - If the query is about general order details without specific mention of 'invoice', classify as 'shipment'.
    - If unable to determine intent or no clear mention of 'invoice' or 'shipment', default to 'shipment'.
    - Response must be exactly 'invoice' or 'shipment' (lowercase).

    Examples:
    - Query: "ORD123", History: "I want the invoice details" -> invoice
    - Query: "Where's my order?", History: "User asked about delivery status" -> shipment
    - Query: "Order details", History: "" -> shipment
    """
)

# MySQL response generation prompt
MYSQL_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a logistics assistant. provide a natural language response.
    Query: {query}
    Schema: {schema}
    History: {history}
    SQL Query: {sql_query}
    SQL Response: {sql_response}
    Use 'according to my knowledge' if appropriate. If no data, suggest providing more details.
    Format the data correctly and human readable with stars and bullet points.
    """
)

# CSV query prompt
CSV_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a logistics assistant answering FAQs based on provided data.
    Context: {context}
    Query: {query}
    Provide a concise answer based only on the context. If the answer is not in the context, say so politely.
    Never reveal raw data or mention the source.
    """
)