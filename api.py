import time
import uvicorn
from typing import List, Literal, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Importaciones de LangGraph para la orquestación
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. CONFIGURACIÓN GLOBAL Y MODELOS
# ==========================================
CHROMA_PATH: str = "chroma_db"
COLLECTION_NAME: str = "libro_ia"
EMBEDDING_MODEL: str = "nomic-embed-text"
LLM_MODEL: str = "gemma3:4b"  # Utilizamos 4b para mejor razonamiento en el enrutamiento
DOCUMENTO_ORIGEN: str = "fundamentos-de-la-inteligencia-artificial-una-vision-introductoria.pdf"

app = FastAPI(title="API Resolutor de e-learning")

print("Cargando conexión a ChromaDB y Ollama...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# Instanciamos el LLM con baja temperatura para respuestas deterministas
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

# ==========================================
# 2. DEFINICIÓN DEL ESTADO DEL GRAFO
# ==========================================
# TypedDict define la estructura de memoria que fluye entre los nodos.
class AgentState(TypedDict):
    pregunta: str
    contexto: str
    respuesta: str
    fuentes: List[int]

# ==========================================
# 3. DEFINICIÓN DE NODOS DEL GRAFO
# ==========================================

def nodo_recuperador(state: AgentState) -> AgentState:
    """
    Busca documentos en la base de datos vectorial y los inyecta en el estado.
    """
    pregunta: str = state["pregunta"]
    
    # Redujimos k=2 para optimizar el tiempo de lectura (Prompt Processing) en hardware limitado
    docs: List[Document] = db.similarity_search(pregunta, k=2)
    
    if not docs:
        return {
            "pregunta": pregunta,
            "contexto": "No se encontró información relevante.",
            "respuesta": "",
            "fuentes": []
        }
    
    # Extraemos el texto y los metadatos
    contexto_texto: str = "\n\n".join([doc.page_content for doc in docs])
    paginas_fuente: List[int] = list(set([doc.metadata.get("page", 0) for doc in docs]))
    
    # Retornamos las variables actualizadas para que el siguiente nodo las consuma
    return {
        "pregunta": pregunta,
        "contexto": contexto_texto,
        "respuesta": "",
        "fuentes": paginas_fuente
    }

def nodo_generador_rag(state: AgentState) -> AgentState:
    """
    Genera una respuesta utilizando el contexto extraído de la base de datos.
    """
    system_prompt: str = """Sos un asistente técnico y educativo experto. 
    Tu tarea es responder la pregunta del usuario utilizando ÚNICAMENTE el contexto proporcionado.
    Si la respuesta no está en el contexto, indicá que no tenés esa información. No inventes.

    Contexto extraído del material de estudio:
    {context}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    cadena = prompt_template | llm
    respuesta_llm = cadena.invoke({
        "context": state["contexto"],
        "question": state["pregunta"]
    })
    
    # Actualizamos el estado únicamente con la respuesta generada
    return {
        "pregunta": state["pregunta"],
        "contexto": state["contexto"],
        "respuesta": respuesta_llm.content,
        "fuentes": state["fuentes"]
    }

def nodo_generador_directo(state: AgentState) -> AgentState:
    """
    Genera una respuesta directa para saludos o preguntas fuera de dominio (charla general).
    """
    system_prompt: str = """Sos un asistente educativo. 
    El usuario te está haciendo una pregunta general, saludando o despidiéndose. 
    Responde de forma amable, cortés y muy breve. No intentes dar explicaciones técnicas aquí.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    cadena = prompt_template | llm
    respuesta_llm = cadena.invoke({"question": state["pregunta"]})
    
    return {
        "pregunta": state["pregunta"],
        "contexto": "N/A",
        "respuesta": respuesta_llm.content,
        "fuentes": []
    }

# ==========================================
# 4. LÓGICA DE ENRUTAMIENTO CONDICIONAL
# ==========================================

def enrutador_condicional(state: AgentState) -> Literal["rag", "directo"]:
    """
    Evalúa la pregunta y decide qué camino debe tomar el grafo.
    Retorna el nombre del siguiente nodo a ejecutar.
    """
    prompt_clasificacion: str = """Clasifica la siguiente intención del usuario.
    Si la consulta busca información técnica, conceptos, o parece una pregunta sobre un tema de estudio, responde estrictamente la palabra: RAG
    Si la consulta es un saludo (hola, buen día), agradecimiento o charla general trivial, responde estrictamente la palabra: DIRECTO

    Pregunta del usuario: {question}
    Clasificación:"""
    
    prompt_template = ChatPromptTemplate.from_template(prompt_clasificacion)
    cadena = prompt_template | llm
    
    decision = cadena.invoke({"question": state["pregunta"]}).content.strip().upper()
    
    if "RAG" in decision:
        print("-> Enrutando a RAG")
        return "rag"
    else:
        print("-> Enrutando a Respuesta Directa")
        return "directo"

# ==========================================
# 5. CONSTRUCCIÓN DEL GRAFO (DAG)
# ==========================================
constructor_grafo = StateGraph(AgentState)

# Agregamos los nodos al grafo
constructor_grafo.add_node("recuperador", nodo_recuperador)
constructor_grafo.add_node("generador_rag", nodo_generador_rag)
constructor_grafo.add_node("generador_directo", nodo_generador_directo)

# Definimos el flujo lógico (Edges)
# Desde el inicio, pasamos por la función condicional para decidir la ruta
constructor_grafo.add_conditional_edges(
    START,
    enrutador_condicional,
    {
        "rag": "recuperador",
        "directo": "generador_directo"
    }
)

# Si fue por la ruta RAG, después de recuperar debe generar
constructor_grafo.add_edge("recuperador", "generador_rag")
# Después de generar (por cualquiera de las dos vías), termina el flujo
constructor_grafo.add_edge("generador_rag", END)
constructor_grafo.add_edge("generador_directo", END)

# Compilamos el grafo para convertirlo en un ejecutable
agente_orquestado = constructor_grafo.compile()


# ==========================================
# 6. ESQUEMAS PYDANTIC Y ENDPOINTS REST
# ==========================================
class QueryRequest(BaseModel):
    pregunta: str

class QueryResponse(BaseModel):
    respuesta: str
    fuentes: List[int]
    tiempo_segundos: float

@app.get("/api/info")
def get_system_info():
    try:
        total_chunks: int = db._collection.count()
        return {
            "documento": DOCUMENTO_ORIGEN,
            "chunks_totales": total_chunks,
            "modelo_embeddings": EMBEDDING_MODEL,
            "modelo_generacion": LLM_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask", response_model=QueryResponse)
def ask_agent(request: QueryRequest) -> QueryResponse:
    start_time: float = time.time()
    try:
        # Definimos el estado inicial requerido por el grafo
        estado_inicial: AgentState = {
            "pregunta": request.pregunta,
            "contexto": "",
            "respuesta": "",
            "fuentes": []
        }
        
        # Ejecutamos el grafo completo
        # invoke() recorre todos los nodos definidos y devuelve el estado final
        estado_final: dict = agente_orquestado.invoke(estado_inicial)
        
        end_time: float = time.time()
        
        return QueryResponse(
            respuesta=estado_final["respuesta"],
            fuentes=estado_final["fuentes"],
            tiempo_segundos=round(end_time - start_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)