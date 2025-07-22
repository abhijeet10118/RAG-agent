from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Use the correct model name listed in `ollama list`
model = Ollama(model="llama3.2")

template = """You are an expert in answering questions about Amazon reviews of books.
Here are some relevant reviews: {reviews}

Here is the question to answer: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
while True:
    print("\n\n-----------------------------------------------")
    question=input("question or q to quit")
    print("\n\n")
    if question=="q":
        break
    result = chain.invoke({
        "reviews": [],
        "question": question
    })

    print(result)
