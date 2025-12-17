# prepare_cuad_data.py (Updated with Text Splitting)
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # <-- ADD THIS IMPORT
import re

# This is the list of the 41 categories from the README.
CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice to Terminate Renewal", "Governing Law", "Most Favored Nation",
    "Non-Compete", "Exclusivity", "No-Solicit of Customers", "Competitive Restriction Exception",
    "No-Solicit of Employees", "Non-Disparagement", "Termination for Convenience",
    "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)", "Change of Control",
    "Anti-Assignment", "Revenue/Profit Sharing", "Price Restriction", "Minimum Commitment",
    "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership", "License Grant",
    "Non-Transferable License", "Affiliate IP License-Licensor", "Affiliate IP License-Licensee",
    "Unlimited/All-You-Can-Eat License", "Irrevocable or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap on Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not to Sue", "Third Party Beneficiary"
]

def clean_column_name(col_name):
    return re.sub(r'\s+', ' ', col_name).strip()

def load_cuad_clauses_from_csv(csv_path="path/to/your/CUAD_v1/master_clauses.csv"):
    """
    Parses the master CUAD CSV, splits long clauses, and returns a list of LangChain Document objects.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {csv_path}")
        return []

    contract_name_col = df.columns[0]
    documents = []
    
    # --- NEW: Initialize a text splitter ---
    # We use a smaller chunk size to ensure it fits within the embedding model's limits.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # --- END OF NEW ---

    df.columns = [clean_column_name(col) for col in df.columns]
    available_categories = [cat for cat in CUAD_CATEGORIES if cat in df.columns]
    print(f"Found {len(available_categories)} matching category columns in the CSV.")

    for _, row in df.iterrows():
        contract_name = row[contract_name_col]
        for category in available_categories:
            clause_text = str(row[category])
            
            if pd.notna(row[category]) and len(clause_text) > 50: 
                # --- THIS IS THE MAIN CHANGE ---
                # Instead of creating one big document, we split it into smaller ones.
                
                # Create initial metadata that will be shared by all chunks
                base_metadata = {
                    "source_contract": contract_name,
                    "clause_category": category
                }
                
                # Split the clause text and create a Document for each chunk
                chunks = text_splitter.split_text(clause_text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata=base_metadata.copy() # Use a copy to avoid aliasing issues
                    )
                    documents.append(doc)
                # --- END OF CHANGE ---

    print(f"Successfully created {len(documents)} clause documents (after splitting).")
    return documents

if __name__ == '__main__':
    docs = load_cuad_clauses_from_csv("CUAD_v1/master_clauses.csv") 
    if docs:
        print("\n--- Example Document (after splitting) ---")
        print(docs[0].page_content)
        print(f"\nMetadata: {docs[0].metadata}")
        
# # prepare_cuad_data.py
# import pandas as pd
# from langchain_core.documents import Document
# import re

# # This is the list of the 41 categories from the README.
# # We need to match these to the column names in the CSV.
# CUAD_CATEGORIES = [
#     "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
#     "Renewal Term", "Notice to Terminate Renewal", "Governing Law", "Most Favored Nation",
#     "Non-Compete", "Exclusivity", "No-Solicit of Customers", "Competitive Restriction Exception",
#     "No-Solicit of Employees", "Non-Disparagement", "Termination for Convenience",
#     "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)", "Change of Control",
#     "Anti-Assignment", "Revenue/Profit Sharing", "Price Restriction", "Minimum Commitment",
#     "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership", "License Grant",
#     "Non-Transferable License", "Affiliate IP License-Licensor", "Affiliate IP License-Licensee",
#     "Unlimited/All-You-Can-Eat License", "Irrevocable or Perpetual License",
#     "Source Code Escrow", "Post-Termination Services", "Audit Rights", "Uncapped Liability",
#     "Cap on Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
#     "Covenant Not to Sue", "Third Party Beneficiary"
# ]

# def clean_column_name(col_name):
#     # The CSV might have slightly different column names (e.g. extra spaces)
#     return re.sub(r'\s+', ' ', col_name).strip()

# def load_cuad_clauses_from_csv(csv_path="path/to/your/CUAD_v1/master_clauses.csv"):
#     """
#     Parses the master CUAD CSV and returns a list of LangChain Document objects.
#     Each document represents one clause.
#     """
#     print(f"Loading data from {csv_path}...")
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"ERROR: The file was not found at {csv_path}")
#         print("Please download the CUAD dataset and provide the correct path to 'master_clauses.csv'")
#         return []

#     # The first column is the contract's filename.
#     contract_name_col = df.columns[0]
    
#     documents = []
    
#     # Clean up the dataframe's column names to match our list
#     df.columns = [clean_column_name(col) for col in df.columns]
    
#     # Get the actual columns that exist in the CSV and match our list
#     available_categories = [cat for cat in CUAD_CATEGORIES if cat in df.columns]
#     print(f"Found {len(available_categories)} matching category columns in the CSV.")

#     for _, row in df.iterrows():
#         contract_name = row[contract_name_col]
#         for category in available_categories:
#             # The cell may contain NaN or other non-string values
#             clause_text = str(row[category])
            
#             # We only care about cells that actually contain clause text.
#             # The README implies a complex structure, but for RAG, let's just grab non-empty text.
#             # We filter out short/empty/NaN values.
#             if pd.notna(row[category]) and len(clause_text) > 50: 
#                 doc = Document(
#                     page_content=clause_text,
#                     metadata={
#                         "source_contract": contract_name,
#                         "clause_category": category
#                     }
#                 )
#                 documents.append(doc)

#     print(f"Successfully created {len(documents)} clause documents.")
#     return documents

# if __name__ == '__main__':
#     # Example of how to run this script directly
#     # IMPORTANT: Replace the path with the actual path to your downloaded CSV file.
#     docs = load_cuad_clauses_from_csv("CUAD_v1\master_clauses.csv") 
#     if docs:
#         print("\n--- Example Document ---")
#         print(docs[0].page_content)
#         print(f"\nMetadata: {docs[0].metadata}")