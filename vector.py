# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import pandas as pd
# import os

# df = pd.read_csv("restaurant_review.csv")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or "nomic-embed-text"
# # ...rest of your code...
# db_location = "./chrome_langchain_db"
# os.path.exists(db_location)
# add_documents = not os.path.exists(db_location)

# if add_documents:
#     documents = []
#     ids = []
    
    
#     for i, row in df.iterrows():
#         document = Document(
#             page_content=row["Title"] + " " + row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#             id=str(i)
#         )
#         ids.append(str(i))
#         documents.append(document)
        
# vector_store = Chroma(
#     collection_name="restaurant_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )

# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)
    
# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 5}
# )

# if __name__ == "__main__":
#     print("Testing embedding model...")
#     emb = embeddings.embed_query("test embedding")
#     print("Embedding result:", emb)

# # import logging
# # logging.basicConfig(level=logging.INFO)

# # from langchain_ollama import OllamaLLM, OllamaEmbeddings
# # from langchain_chroma import Chroma
# # from langchain_core.documents import Document
# # import pandas as pd
# # import os

# # df = pd.read_csv("restaurant_review.csv")
# # embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or "nomic-embed-text"
# # # ...rest of your code...
# # db_location = "./chrome_langchain_db"
# # os.path.exists(db_location)
# # add_documents = not os.path.exists(db_location)

# # if add_documents:
# #     documents = []
# #     ids = []
# #     for i, row in df.iterrows():
# #         document = Document(
# #             page_content=row["Title"] + " " + row["Review"],
# #             metadata={"rating": row["Rating"], "date": row["Date"]},
# #             id=str(i)
# #         )
# #         ids.append(str(i))
# #         documents.append(document)
        
# # vector_store = Chroma(
# #     collection_name="restaurant_reviews",
# #     persist_directory=db_location,
# #     embedding_function=embeddings
# # )

# # if add_documents:
# #     vector_store.add_documents(documents=documents, ids=ids)
    
# # retriever = vector_store.as_retriever(
# #     search_kwargs={"k": 5}
# # )

# # if __name__ == "__main__":
# #     print("Testing embedding model...")
# #     emb = embeddings.embed_query("test embedding")
# #     print("Embedding result:", emb)
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# import pandas as pd
# import os
# import time
# import logging
# from typing import List, Dict, Any

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class RestaurantReviewAnalyzer:
#     def __init__(self, csv_path: str = "restaurant_review.csv", db_location: str = "./chroma_langchain_db"):
#         """
#         Initialize the restaurant review analyzer.
        
#         Args:
#             csv_path: Path to the CSV file containing reviews
#             db_location: Directory to store/load the Chroma vector database
#         """
#         self.csv_path = csv_path
#         self.db_location = db_location
#         self.embeddings = None
#         self.vector_store = None
#         self.retriever = None
#         self.llm = None
        
#         self._validate_files()
#         self._initialize_components()
        
#     def _validate_files(self):
#         """Validate that required files exist."""
#         if not os.path.exists(self.csv_path):
#             raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
#         logger.info(f"Found CSV file at {self.csv_path}")
        
#     def _initialize_components(self):
#         """Initialize all the components needed for the analyzer."""
#         start_time = time.time()
        
#         # Load and prepare data
#         self.df = pd.read_csv(self.csv_path)
#         logger.info(f"Loaded {len(self.df)} reviews from CSV.")
        
#         # Initialize embeddings
#         self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
#         logger.info("Initialized Ollama Embeddings model.")
        
#         # Initialize LLM
#         self.llm = OllamaLLM(model="llama3")
#         logger.info("Initialized Ollama LLM.")
        
#         # Create or load vector store
#         self._initialize_vector_store()
        
#         # Create retriever
#         self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
#         logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
        
#     def _initialize_vector_store(self):
#         """Initialize the Chroma vector store, adding documents if needed."""
#         add_documents = not os.path.exists(self.db_location)
        
#         # Prepare documents if needed
#         documents = []
#         ids = []
        
#         if add_documents:
#             logger.info("Preparing documents for vector store...")
#             for i, row in self.df.iterrows():
#                 document = Document(
#                     page_content=row["Title"] + " " + row["Review"],
#                     metadata={
#                         "rating": row["Rating"],
#                         "date": row["Date"],
#                         "source": "restaurant_reviews"
#                     }
#                 )
#                 ids.append(str(i))
#                 documents.append(document)
        
#         # Create vector store
#         self.vector_store = Chroma(
#             collection_name="restaurant_reviews",
#             persist_directory=self.db_location,
#             embedding_function=self.embeddings
#         )
        
#         # Add documents if this is the first run
#         if add_documents:
#             logger.info("Adding documents to vector store...")
#             self.vector_store.add_documents(documents=documents, ids=ids)
#             logger.info(f"Added {len(documents)} documents to vector store.")
    
#     def search_reviews(self, query: str, k: int = 5) -> List[Document]:
#         """
#         Search for similar reviews based on the query.
        
#         Args:
#             query: The search query
#             k: Number of results to return
            
#         Returns:
#             List of matching documents
#         """
#         logger.info(f"Searching for reviews similar to: '{query}'")
#         return self.retriever.invoke(query)
    
#     def analyze_sentiment(self, query: str) -> Dict[str, Any]:
#         """
#         Analyze the sentiment of reviews related to the query.
        
#         Args:
#             query: The topic to analyze
            
#         Returns:
#             Dictionary containing analysis results
#         """
#         logger.info(f"Analyzing sentiment for: '{query}'")
        
#         # Retrieve relevant reviews
#         relevant_reviews = self.search_reviews(query)
#         if not relevant_reviews:
#             return {"error": "No relevant reviews found"}
        
#         # Prepare prompt
#         prompt_template = """
#         Analyze the sentiment of these restaurant reviews about {topic}:
        
#         {reviews}
        
#         Provide:
#         1. Overall sentiment (positive, neutral, negative)
#         2. Key positive aspects mentioned
#         3. Key negative aspects mentioned
#         4. Suggestions for improvement (if any negative aspects)
#         """
        
#         reviews_text = "\n\n".join([doc.page_content for doc in relevant_reviews])
#         prompt = ChatPromptTemplate.from_template(prompt_template)
        
#         # Create chain
#         chain = (
#             {"topic": RunnablePassthrough(), "reviews": RunnablePassthrough()}
#             | prompt
#             | self.llm
#             | StrOutputParser()
#         )
        
#         # Get analysis
#         analysis = chain.invoke({
#             "topic": query,
#             "reviews": reviews_text
#         })
        
#         return {
#             "query": query,
#             "relevant_reviews": [doc.page_content for doc in relevant_reviews],
#             "analysis": analysis
#         }
    
#     def generate_summary_report(self) -> str:
#         """
#         Generate a summary report of all reviews.
        
#         Returns:
#             The generated report as a string
#         """
#         logger.info("Generating summary report...")
        
#         prompt_template = """
#         Analyze these restaurant reviews and generate a comprehensive summary:
        
#         {reviews}
        
#         Include in your report:
#         1. Overall sentiment distribution
#         2. Most common positive themes
#         3. Most common complaints
#         4. Suggestions for the restaurant
#         5. Any notable trends over time
#         """
        
#         # Get all documents (in a real app, you might sample or chunk this)
#         all_docs = self.vector_store.get()['documents']
#         reviews_text = "\n\n".join(all_docs)
        
#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         chain = prompt | self.llm | StrOutputParser()
        
#         return chain.invoke({"reviews": reviews_text})

# if __name__ == "__main__":
#     try:
#         # Initialize the analyzer
#         analyzer = RestaurantReviewAnalyzer()
        
#         # Test embedding model
#         logger.info("Testing embedding model...")
#         emb = analyzer.embeddings.embed_query("test embedding")
#         logger.info(f"Embedding result length: {len(emb)}")
        
#         # Example usage
#         query = "burger quality"
#         logger.info(f"\n===== Analyzing reviews about: {query} =====")
#         results = analyzer.analyze_sentiment(query)
        
#         print("\n=== Analysis Results ===")
#         print(results["analysis"])
        
#         # Generate full report
#         print("\n=== Generating Full Report ===")
#         report = analyzer.generate_summary_report()
#         print(report)
        
#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RestaurantReviewVectorStore:
    def __init__(self, csv_path: str = "restaurant_review.csv", db_location: str = "./chroma_langchain_db"):
        """
        Initialize the restaurant review vector store.
        
        Args:
            csv_path: Path to the CSV file containing reviews
            db_location: Directory to store/load the Chroma vector database
        """
        self.csv_path = csv_path
        self.db_location = db_location
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        
        self._validate_files()
        self._initialize_components()
        
    def _validate_files(self):
        """Validate that required files exist."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
        logger.info(f"Found CSV file at {self.csv_path}")
        
    def _initialize_components(self):
        """Initialize all the components needed for the vector store."""
        start_time = time.time()
        
        # Load and prepare data
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} reviews from CSV.")
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        logger.info("Initialized Ollama Embeddings model.")
        
        # Create or load vector store
        self._initialize_vector_store()
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        logger.info(f"Vector store initialization completed in {time.time() - start_time:.2f} seconds")
        
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store, adding documents if needed."""
        add_documents = not os.path.exists(self.db_location)
        
        # Prepare documents if needed
        documents = []
        ids = []
        
        if add_documents:
            logger.info("Preparing documents for vector store...")
            for i, row in self.df.iterrows():
                document = Document(
                    page_content=row["Title"] + " " + row["Review"],
                    metadata={
                        "rating": row["Rating"],
                        "date": row["Date"],
                        "source": "restaurant_reviews"
                    }
                )
                ids.append(str(i))
                documents.append(document)
        
        # Create vector store
        self.vector_store = Chroma(
            collection_name="restaurant_reviews",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        
        # Add documents if this is the first run
        if add_documents:
            logger.info("Adding documents to vector store...")
            self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to vector store.")

# Initialize the vector store and retriever when this module is imported
vector_store = RestaurantReviewVectorStore()
retriever = vector_store.retriever