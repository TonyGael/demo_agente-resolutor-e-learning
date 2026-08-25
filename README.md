# Agente Resolutor de e-Learning

> Te invito a conectar en LinkedIn: [Tony Gael](https://www.linkedin.com/in/TonyGael/).

Un agente conversacional RAG end-to-end que permite a los alumnos consultar material de estudio de forma interactiva. Combina FastAPI para el backend y Streamlit para el frontend, y utiliza ChromaDB junto con modelos locales de la familia Gemma 3 gestionados mediante Ollama.

## Motivación
El proyecto resuelve la necesidad de integrar pipelines de inteligencia artificial en experiencias educativas, priorizando la autonomía de los datos y la escalabilidad. Explora cómo diseñar una arquitectura RAG robusta bajo restricciones reales de hardware, emulando un entorno de producción donde la funcionalidad y la resiliencia priman sobre la velocidad bruta.

## Hardware utilizado
| Componente | Especificación |
|---|---|
| GPU | Asus Nvidia GTX 970 Strix, 4GB VRAM |
| CPU | Intel Core i5-7500 |
| RAM | 12 GB DDR3 |
| Sistema operativo | Ubuntu MATE 24 |

## Stack tecnológico
- **Ollama:** Motor de inferencia local para servir los modelos generativos (`gemma3:1b` y `gemma3:4b`) y el modelo de embeddings (`nomic-embed-text`).
- **LangChain:** Framework de orquestación para manejar el flujo RAG, el chunking de documentos y la conexión estructurada con la base vectorial.
- **ChromaDB:** Base de datos vectorial persistente para el almacenamiento y la búsqueda semántica local.
- **FastAPI:** Framework backend de alto rendimiento para exponer los endpoints de consulta y telemetría.
- **Streamlit:** Frontend rápido para construir la interfaz conversacional y visualizar las métricas del sistema en tiempo real.

## Estructura del proyecto
```text
demo_agente-resolutor-e-learning/
├── api.py
├── app.py
├── ingest.py
├── inspect_db.py
├── run.py
├── fundamentos-de-la-inteligencia-artificial-una-vision-introductoria.pdf
└── chroma_db/
```

## Instalación
Clonar el repositorio y preparar el entorno de Python:
```bash
git clone https://github.com/TonyGael/demo_agente-resolutor-e-learning
cd demo_agente-resolutor-e-learning
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pydantic langchain-community langchain-text-splitters langchain-ollama langchain-chroma pypdf streamlit requests
```

Descargar los modelos necesarios en Ollama:
```bash
ollama pull nomic-embed-text
ollama pull gemma3:1b
ollama pull gemma3:4b
```

## Uso
Ingestar el documento PDF inicial y poblar la base de datos vectorial (proceso por lotes):
```bash
python3 ingest.py
```

Inspeccionar la base de datos para verificar los fragmentos guardados (opcional):
```bash
python3 inspect_db.py
```

Levantar el backend y el frontend simultáneamente mediante el script orquestador:
```bash
python3 run.py
```

La interfaz gráfica queda disponible en `http://localhost:8501`.

## Estado actual
- [x] Ingesta de datos por lotes (batching) desde archivos PDF
- [x] Generación de embeddings con `nomic-embed-text` y persistencia en ChromaDB local
- [x] Backend RESTful con FastAPI y medición de latencia
- [x] Interfaz gráfica en Streamlit con telemetría del sistema y chat integrado
- [x] Orquestación de múltiples servicios (API y Web) desde un único script de ejecución

## TODO
- [ ] Implementar streaming de respuestas en la API y el frontend para mejorar el Time To First Token (TTFT)
- [ ] Integrar un grafo de enrutamiento con LangGraph para dotar al agente de capacidad de decisión
- [ ] Añadir persistencia de memoria para mantener el contexto multi-turno de la conversación
- [ ] Optimizar el tamaño de los chunks y probar estrategias de chunking semántico
- [ ] Incluir comentarios detallados inline en los diferentes bloques de código con type hints estrictos
- [ ] [COMPLETAR: Cualquier otro requerimiento que surja durante el desarrollo]