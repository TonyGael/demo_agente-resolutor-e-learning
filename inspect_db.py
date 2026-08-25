import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text" 
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "libro_ia"

def main():
    print("Conectando a la base de datos vectorial local...")
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # Agregamos el collection_name explícito aquí también
        db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"Error al conectar con Chroma o Ollama: {e}")
        sys.exit(1)

    collection_count = db._collection.count()
    print(f"\nTotal de fragmentos (chunks) en la base de datos: {collection_count}")

    if collection_count == 0:
        print("La base de datos sigue vacía.")
        sys.exit(0)

    print("\n--- Inspeccionando los primeros 2 registros guardados ---")
    results = db.get(limit=2)
    
    for i in range(len(results['ids'])):
        print(f"\nID del Chunk: {results['ids'][i]}")
        print(f"Metadatos: {results['metadatas'][i]}") 
        print(f"Texto guardado: {results['documents'][i][:200]}...")

    query = "¿Qué es una red neuronal?"
    print(f"\n--- Probando búsqueda por similitud con la pregunta: '{query}' ---")
    
    docs_relevantes = db.similarity_search(query, k=2)
    
    for idx, doc in enumerate(docs_relevantes):
        print(f"\nResultado {idx + 1}:")
        print(f"Fuente: Página {doc.metadata.get('page', 'Desconocida')}")
        print(f"Fragmento: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()