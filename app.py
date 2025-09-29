from fastapi import FastAPI, Query
from pydantic import BaseModel
from agent import (
    setup_environment,
    create_synthetic_medical_notes,
    initialize_models,
    create_vector_store,
    build_rag_chain,
)

app = FastAPI(title="Medical RAG Agent API")

# Initialize RAG pipeline at startup
api_key = setup_environment()
patient_notes = create_synthetic_medical_notes()
llm, embeddings = initialize_models(api_key)
vectorstore = create_vector_store(patient_notes, embeddings)
rag_chain = build_rag_chain(llm, vectorstore)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Medical RAG Agent API is running."}