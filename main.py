# # # from langchain_community.llms import Ollama
# # from langchain_ollama import OllamaLLM
# # from langchain_core.prompts import ChatPromptTemplate
# # from vector import retriever

# # # Use the correct model name listed in `ollama list`
# # # model = Ollama(model="llama3.2")
# # model = OllamaLLM(model="llama3.2")

# # template = """You are an expert in answering questions about restaurant reviews of books.
# # Here are some relevant reviews: {reviews}

# # Here is the question to answer: {question}"""

# # prompt = ChatPromptTemplate.from_template(template)

# # chain = prompt | model
# # while True:
# #     print("\n\n-------------------------------")
# #     question = input("Ask your question (q to quit): ")
# #     print("\n\n")
# #     if question == "q":
# #         break

# #     print("Invoking retriever...")
# #     reviews = retriever.invoke(question)
# #     print("Reviews received:", reviews)
# #     formatted_reviews = "\n\n".join([doc.page_content for doc in reviews])
# #     print("Formatted reviews:", formatted_reviews)
# #     print("Invoking LLM chain...")
# #     result = chain.invoke({"reviews": formatted_reviews, "question": question})
# #     print("Result received:")
# #     print(result)


# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def initialize_llm(model_name: str = "llama3"):
#     """Initialize the Ollama LLM model."""
#     try:
#         logger.info(f"Initializing LLM with model: {model_name}")
#         return OllamaLLM(model=model_name)
#     except Exception as e:
#         logger.error(f"Failed to initialize LLM: {str(e)}")
#         raise

# def create_chain(model):
#     """Create the question-answering chain."""
#     template = """You are an expert in answering questions about restaurant reviews.
#     Here are some relevant reviews: {reviews}

#     Here is the question to answer: {question}"""

#     prompt = ChatPromptTemplate.from_template(template)
#     return prompt | model

# def main():
#     try:
#         # Initialize components
#         model = initialize_llm("llama3")
#         chain = create_chain(model)

#         # Interactive Q&A loop
#         while True:
#             print("\n\n-------------------------------")
#             question = input("Ask your question about restaurant reviews (q to quit): ").strip()
#             print("\n\n")
            
#             if question.lower() == 'q':
#                 logger.info("Exiting...")
#                 break

#             if not question:
#                 print("Please enter a valid question.")
#                 continue

#             try:
#                 logger.info(f"Processing question: {question}")
                
#                 # Retrieve relevant reviews
#                 logger.info("Retrieving relevant reviews...")
#                 reviews = retriever.invoke(question)
#                 formatted_reviews = "\n\n".join([doc.page_content for doc in reviews])
                
#                 # Generate answer
#                 logger.info("Generating answer...")
#                 result = chain.invoke({
#                     "reviews": formatted_reviews,
#                     "question": question
#                 })
                
#                 # Display results
#                 print("\n=== Relevant Reviews ===")
#                 for i, doc in enumerate(reviews, 1):
#                     print(f"\nReview {i}:")
#                     print(doc.page_content)
#                     print(f"Rating: {doc.metadata.get('rating', 'N/A')}")
#                     print(f"Date: {doc.metadata.get('date', 'N/A')}")
                
#                 print("\n=== Answer ===")
#                 print(result)
                
#             except Exception as e:
#                 logger.error(f"Error processing question: {str(e)}")
#                 print("Sorry, an error occurred while processing your question. Please try again.")

#     except Exception as e:
#         logger.error(f"Application error: {str(e)}")
#         print("A critical error occurred. Please check the logs.")

# if __name__ == "__main__":
#     main()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_llm(model_name: str = "llama3"):
    """Initialize the Ollama LLM model."""
    try:
        logger.info(f"Initializing LLM with model: {model_name}")
        # Adding timeout and other safety parameters
        return OllamaLLM(
            model=model_name,
            temperature=0.7,
            timeout=60,  # 60 seconds timeout
            keep_alive="5m"  # keep model loaded for 5 minutes
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_chain(model):
    """Create the question-answering chain."""
    template = """You are an expert in answering questions about restaurant reviews.
    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}

    Provide a detailed answer based on the reviews, including both positive and negative aspects if mentioned.
    If no relevant reviews are found, say "I couldn't find relevant reviews about this topic."""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

def format_reviews(reviews):
    """Format reviews for display."""
    if not reviews:
        return "No relevant reviews found."
    
    formatted = []
    for i, doc in enumerate(reviews, 1):
        formatted.append(
            f"\nReview {i} (Rating: {doc.metadata.get('rating', 'N/A')}, Date: {doc.metadata.get('date', 'N/A')}):\n"
            f"{doc.page_content}"
        )
    return "\n".join(formatted)

def main():
    try:
        # Initialize components
        model = initialize_llm("llama3")
        chain = create_chain(model)

        logger.info("Starting interactive Q&A session...")
        print("\nRestaurant Review Q&A System")
        print("Type 'q' to quit\n")

        # Interactive Q&A loop
        while True:
            try:
                question = input("\nAsk your question about restaurant reviews: ").strip()
                
                if question.lower() == 'q':
                    logger.info("Exiting...")
                    break

                if not question:
                    print("Please enter a valid question.")
                    continue

                logger.info(f"Processing question: {question}")
                
                # Retrieve relevant reviews
                logger.info("Retrieving relevant reviews...")
                reviews = retriever.invoke(question)
                formatted_reviews = "\n\n".join([doc.page_content for doc in reviews])
                
                if not reviews:
                    print("\nNo relevant reviews found for your question.")
                    continue
                
                # Generate answer
                logger.info("Generating answer...")
                result = chain.invoke({
                    "reviews": formatted_reviews,
                    "question": question
                })
                
                # Display results
                print("\n=== Relevant Reviews ===")
                print(format_reviews(reviews))
                
                print("\n=== Answer ===")
                print(result)
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                logger.error(traceback.format_exc())
                print("\nSorry, an error occurred while processing your question. Please try again.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())
        print("A critical error occurred. Please check the logs.")

if __name__ == "__main__":
    main()