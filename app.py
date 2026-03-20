from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def load_embeddings_and_db():
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Loading FAISS index...")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Done!")
    return db


def load_llm():
    print("Loading Groq LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=1024
    )
    print("LLM ready!")
    return llm


def create_rag_chain(db, llm):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    template = """Use the following context from the Dune books to answer the question.
If you don't know the answer from the context, just say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain

def create_conversational_chain(db, llm):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10,"fetch_k":30}
    )

    template = """You are a Dune encyclopedia. You have deep knowledge of all 6 Dune books by Frank Herbert.
    Use the context below to give a thorough, detailed answer.
    If the context doesn't contain enough information, say what you know from context and mention it may be incomplete.
    Do NOT invent facts not present in the context.

    Previous conversation:
    {chat_history}

    Context from Dune books:
    {context}

    Question: {question}

    Detailed answer:
    After your detailed answer, always end with:
    TL;DR: [2 line simple summary of your answer]"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(messages):
        if not messages:
            return "No previous conversation."
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"Assistant: {msg.content}\n"
        return history

    def rag_with_history(inputs):
        question = inputs["question"]
        chat_history = inputs["chat_history"]

        docs = retriever.invoke(question)
        context = format_docs(docs)
        history_text = format_history(chat_history)

        filled_prompt = prompt.format(
            chat_history=history_text,
            context=context,
            question=question
        )

        answer = llm.invoke(filled_prompt)
        return answer

    return rag_with_history


if __name__ == "__main__":
    db = load_embeddings_and_db()
    llm = load_llm()
    chain = create_conversational_chain(db, llm)

    chat_history = []

    print("\nDune RAG ready! Type your question or 'quit' to exit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "quit":
            break

        if question.strip() == "":
            continue

        answer = chain({"question": question, "chat_history": chat_history})
        answer_text = answer.content if hasattr(answer, 'content') else str(answer)
        print(f"\nAnswer: {answer_text}\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer_text))
