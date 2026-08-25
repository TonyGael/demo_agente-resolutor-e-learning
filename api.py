import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "libro_ia"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:1b"

app = FastAPI(title="API Resolutor de e-learning")

print("Cargando conexión a ChromaDB y Ollama...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

system_prompt = """Sos un asistente técnico y educativo experto. 
Tu tarea es responder la pregunta del usuario utilizando ÚNICAMENTE el contexto proporcionado.
Si la respuesta no está en el contexto, indicá que no tenés esa información. No inventes.

Contexto extraído del material de estudio:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

class QueryRequest(BaseModel):
    pregunta: str

class QueryResponse(BaseModel):
    respuesta: str
    fuentes: list[int]

@app.post("/ask", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    try:
        docs = db.similarity_search(request.pregunta, k=3)
        
        if not docs:
            return QueryResponse(
                respuesta="No encontré información relevante en el material de estudio.", 
                fuentes=[]
            )
        
        contexto_texto = "\n\n".join([doc.page_content for doc in docs])
        paginas_fuente = [doc.metadata.get("page", 0) for doc in docs]
        
        chain = prompt_template | llm
        respuesta_llm = chain.invoke(
            {
                "context": contexto_texto,
                "question": request.pregunta
            }
        )
        
        return QueryResponse(
            respuesta=respuesta_llm.content,
            fuentes=list(set(paginas_fuente))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)