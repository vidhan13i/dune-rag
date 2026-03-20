from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# ── STEP 1: Load your existing FAISS index ──────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=1024
)

# ── STEP 2: Define your test questions ─────────────────────
# These are questions you know the answers to from the books
test_questions = [
    "Who is Paul Atreides?",
    "What is the spice melange?",
    "Who are the Fremen?",
    "What is the Golden Path?",
    "Who is Duncan Idaho?",
    "What is the Bene Gesserit?",
    "What is a Mentat?",
    "Who is Stilgar?",
    "What happened to House Atreides?",
    "What is a sandworm?",
]

# ── STEP 3: Generate answers + retrieve contexts ────────────
print("Generating answers for test questions...")

questions = []
answers = []
contexts = []
ground_truths = []

for question in test_questions:
    print(f"Processing: {question}")

    # retrieve chunks
    docs = retriever.invoke(question)
    context_texts = [doc.page_content for doc in docs]

    # generate answer using your LLM
    prompt = f"""You are a Dune encyclopedia. Use the context below to answer.
Context: {' '.join(context_texts)}
Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    answer = response.content

    questions.append(question)
    answers.append(answer)
    contexts.append(context_texts)
    ground_truths.append("")  # optional - add real answers if you have them

# ── STEP 4: Build RAGAS dataset ─────────────────────────────
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

dataset = Dataset.from_dict(data)

# ── STEP 5: Wrap LLM and embeddings for RAGAS ───────────────
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ── STEP 6: Run evaluation ───────────────────────────────────
print("\nRunning RAGAS evaluation...")

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n── RAGAS Results ──────────────────────────────")
print(f"Faithfulness:       {result['faithfulness']:.2f}")
print(f"Answer Relevancy:   {result['answer_relevancy']:.2f}")
print(f"Context Precision:  {result['context_precision']:.2f}")
print("───────────────────────────────────────────────")