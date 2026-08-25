import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

FILE_PATH = "fundamentos-de-la-inteligencia-artificial-una-vision-introductoria.pdf"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
# Definimos un nombre de colección fijo para evitar desajustes
COLLECTION_NAME = "libro_ia" 

def main():
    print(f"Iniciando el pipeline de ingesta con {EMBEDDING_MODEL}...")

    try:
        loader = PyPDFLoader(FILE_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"Error crítico al cargar el PDF: {e}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"Libro dividido en {total_chunks} fragmentos.")

    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error al configurar OllamaEmbeddings: {e}")
        sys.exit(1)

    print("Inicializando ChromaDB y guardando por lotes para evitar saturación...")
    
    try:
        # 1. En vez de from_documents, inicializamos la BD vacía apuntando al directorio
        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )
        
        # 2. Lógica de Batching (Lotes)
        # Procesamos de a 50 fragmentos para darle respiro a la GPU/CPU local
        batch_size = 50
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            # add_documents procesa los vectores y los persiste automáticamente
            db.add_documents(batch)
            print(f"Lote insertado: {min(i + batch_size, total_chunks)} / {total_chunks}")
            
        print(f"¡Éxito! Todos los vectores guardados en '{CHROMA_PATH}'.")
    except Exception as e:
        print(f"Error al escribir en la base de datos vectorial: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()