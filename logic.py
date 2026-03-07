from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma3:4b")

template = """
You are an expert in answering questions about environmental CO2 reduction,
awareness, and action planning.

Here is some relevant data:
{data}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def get_response_stream(question: str):
    context_docs = retriever.invoke(question)
    context_data = "\n".join(doc.page_content for doc in context_docs)
    return chain.stream({"data": context_data, "question": question})


def print_stream(response_stream):
    chunks = []
    for chunk in response_stream:
        text = str(chunk)
        print(text, end="", flush=True)
        chunks.append(text)
    print()
    return "".join(chunks)


if __name__ == "__main__":
    while True:
        print("\n ----------------------------------")
        question = input("Ask your question (q to quit): ")
        if question == "q":
            break

        stream = get_response_stream(question)
        print_stream(stream)
