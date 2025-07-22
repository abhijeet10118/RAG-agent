# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Use the correct model name listed in `ollama list`
# model = Ollama(model="llama3.2")
model = OllamaLLM(model="llama3.2")

template = """You are an expert in answering questions about restaurant reviews of books.
Here are some relevant reviews: {reviews}

Here is the question to answer: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    print("Invoking retriever...")
    reviews = retriever.invoke(question)
    print("Reviews received:", reviews)
    formatted_reviews = "\n\n".join([doc.page_content for doc in reviews])
    print("Formatted reviews:", formatted_reviews)
    print("Invoking LLM chain...")
    result = chain.invoke({"reviews": formatted_reviews, "question": question})
    print("Result received:")
    print(result)
