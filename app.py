import streamlit as st
import requests

st.set_page_config(page_title="Agente e-Learning | AI Engineer", page_icon="🤖", layout="wide")

API_URL = "http://localhost:8000"

st.sidebar.title("⚙️ Estado del Sistema")

try:
    info_res = requests.get(f"{API_URL}/api/info")
    if info_res.status_code == 200:
        info = info_res.json()
        st.sidebar.markdown("**Documento Base:**")
        st.sidebar.caption(info['documento'])
        
        st.sidebar.markdown("**Base Vectorial:**")
        st.sidebar.info(f"📚 {info['chunks_totales']} chunks generados")
        
        st.sidebar.markdown("**Modelo Embeddings:**")
        st.sidebar.success(f"🔍 {info['modelo_embeddings']}")
        
        st.sidebar.markdown("**Modelo Generación (LLM):**")
        st.sidebar.warning(f"🧠 {info['modelo_generacion']}")
    else:
        st.sidebar.error("Error al cargar la telemetría del backend.")
except Exception as e:
    st.sidebar.error("API desconectada. Asegurate de que FastAPI esté corriendo.")

st.title("💬 Agente Resolutor de e-Learning")
st.markdown("Consulta al material de estudio. El agente buscará en la base vectorial y responderá basándose en el documento.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre el libro de Inteligencia Artificial..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta y buscando en la base de datos..."):
            try:
                res = requests.post(f"{API_URL}/api/ask", json={"pregunta": prompt})
                
                if res.status_code == 200:
                    data = res.json()
                    respuesta = data['respuesta']
                    fuentes = data['fuentes']
                    latencia = data['tiempo_segundos']
                    
                    texto_fuentes = ", ".join(map(str, fuentes)) if fuentes else "N/A"
                    respuesta_formateada = f"{respuesta}\n\n---\n*📄 Páginas fuente: {texto_fuentes} | ⏱️ Latencia: {latencia}s*"
                    
                    st.markdown(respuesta_formateada)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_formateada})
                else:
                    st.error(f"Error del servidor: {res.status_code}")
            except Exception as e:
                st.error("Error crítico: No se pudo conectar con FastAPI. Verificá la terminal del backend.")