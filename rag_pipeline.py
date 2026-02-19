import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Step 1: Load PDFs from a folder
def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

docs = load_docs("data")
print("PDF Pages Loaded:", len(docs))

# Step 2: Split PDFs into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = text_splitter.split_documents(docs)
print("Chunks Created:", len(chunks))
# Step 3: Embeddings & Vector Store
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save texts into Chroma vector DB
print("Creating vector database...")
texts = [c.page_content for c in chunks]
db = FAISS.from_texts(texts, embedding_model)

# Retriever - will search for top 3 most similar chunks
retriever = db.as_retriever(search_kwargs={"k": 3})
print("Vector database ready!")
# Step 4: The Brain - Local LLM
print("Loading LLM model...")
llm = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150
)
print("LLM ready!")
# Step 5: The Agent Brain - Decision Maker
def agent_controller(query):
    """
    This decides whether to search documents or answer directly
    """
    q = query.lower()
    # Keywords that indicate user wants to search documents
    if any(word in q for word in ["pdf", "document", "data", "summarize", "information", "find"]):
        return "search"
    return "direct"
    # Step 6: RAG Execution Loop
def rag_answer(query):
    """
    Main function that processes queries using the agent
    """
    action = agent_controller(query)

    if action == "search":
        print(f"🕵️ Agent decided to SEARCH document for: '{query}'")
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        final_prompt = f"Use this context:\n{context}\n\nAnswer:\n{query}"
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query

    response = llm(final_prompt)[0]["generated_text"]
    return response

# Test 1: A document-specific question
print("\n" + "="*50)
print("TEST 1: Document-specific question")
print("="*50)
query1 = "Give me a 5-point summary from the PDF"
answer1 = rag_answer(query1)
print(f"\nAnswer: {answer1}\n")

print("-" * 50)

# Test 2: A general knowledge question
print("\n" + "="*50)
print("TEST 2: General knowledge question")
print("="*50)
query2 = "What is an Ideal Resume Format? Explain in 50 words."
answer2 = rag_answer(query2)
print(f"\nAnswer: {answer2}\n")