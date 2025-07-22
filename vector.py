from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

df = pd.read_csv("restaurant_review.csv")
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or "nomic-embed-text"
# ...rest of your code...
db_location = "./chrome_langchain_db"
os.path.exists(db_location)
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

if __name__ == "__main__":
    print("Testing embedding model...")
    emb = embeddings.embed_query("test embedding")
    print("Embedding result:", emb)
