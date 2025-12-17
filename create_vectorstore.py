# create_vectorstore.py (Updated Version)

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Remove the langchain_core.documents import, as it's now in the other file

# --- NEW: Import the CUAD data loader ---
from prepare_cuad_data import load_cuad_clauses_from_csv

INDEX_PATH = "faiss_index"
# --- NEW: Set the path to your downloaded CUAD CSV file ---
CUAD_CSV_PATH = "CUAD_v1\master_clauses.csv" # <-- IMPORTANT: CHANGE THIS PATH

def create_vector_store():
    """Reads clauses from the CUAD dataset, creates embeddings, and saves them."""
    
    # --- THIS IS THE MAIN CHANGE ---
    print("Loading clauses from the CUAD dataset...")
    documents = load_cuad_clauses_from_csv(CUAD_CSV_PATH)
    # --- END OF CHANGE ---
    
    if not documents:
        print("No documents were loaded from the CUAD dataset. Aborting.")
        return

    print(f"Found {len(documents)} clauses. Creating embeddings (this may take a while)...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(documents, embeddings)
    print("Embeddings created successfully.")

    # Delete old index if it exists, to avoid conflicts
    if os.path.exists(INDEX_PATH):
        import shutil
        print(f"Removing old index at '{INDEX_PATH}'...")
        shutil.rmtree(INDEX_PATH)

    vector_store.save_local(INDEX_PATH)
    print(f"Vector store saved to '{INDEX_PATH}'. You can now run the main app.")

if __name__ == "__main__":
    create_vector_store()

# create_vectorstore.py (Updated Version)

# import os
# from dotenv import load_dotenv  # <-- 1. ADD THIS IMPORT
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from prepare_cuad_data import load_cuad_clauses_from_csv

# load_dotenv()  # <-- 2. ADD THIS LINE TO LOAD THE .env FILE

# INDEX_PATH = "faiss_index"
# CUAD_CSV_PATH = "CUAD_v1/master_clauses.csv"

# def create_vector_store():
#     """Reads clauses from the CUAD dataset, creates embeddings, and saves them."""
    
#     print("Loading clauses from the CUAD dataset...")
#     documents = load_cuad_clauses_from_csv(CUAD_CSV_PATH)
    
#     if not documents:
#         print("No documents were loaded from the CUAD dataset. Aborting.")
#         return

#     print(f"Found {len(documents)} clauses. Creating embeddings (this may take a while)...")

#     # This line will now work because of the environment variable we set
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     vector_store = FAISS.from_documents(documents, embeddings)
#     print("Embeddings created successfully.")

#     if os.path.exists(INDEX_PATH):
#         import shutil
#         print(f"Removing old index at '{INDEX_PATH}'...")
#         shutil.rmtree(INDEX_PATH)

#     vector_store.save_local(INDEX_PATH)
#     print(f"Vector store saved to '{INDEX_PATH}'. You can now run the main app.")

# if __name__ == "__main__":
#     create_vector_store()