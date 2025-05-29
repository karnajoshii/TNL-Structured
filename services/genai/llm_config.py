import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import logging
from typing import Optional
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../../Uploads")
FAISS_PATH = os.path.join(BASE_DIR, "../../faiss_index")
for directory in [UPLOAD_FOLDER, FAISS_PATH]:
    os.makedirs(directory, exist_ok=True)

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Global vector store
vector_store: Optional[FAISS] = None

# Load FAQ embeddings
faqs_path = os.path.join(BASE_DIR, 'data/faqs.csv')  
df = pd.read_csv(faqs_path)  # Assume columns: 'question', 'answer'
questions = df['question'].tolist()
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(questions, embeddings)