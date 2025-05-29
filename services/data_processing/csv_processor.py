import os
import uuid
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../../Uploads")
FAISS_PATH = os.path.join(BASE_DIR, "data/faiss_index")

def process_csv(file_path: str) -> Optional[FAISS]:
    """Process CSV file and create FAISS vector store."""
    global vector_store
    try:
        df = pd.read_csv(file_path)
        csv_text = df.to_string(index=False)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(csv_text)
        
        metadatas = [{"chunk_id": str(uuid.uuid4()), "source": file_path, "index": i} 
                     for i in range(len(chunks))]
        
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
        vector_store.save_local(FAISS_PATH)
        logger.info(f"Processed CSV and saved FAISS index: {file_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise