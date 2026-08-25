import subprocess
import sys
import time

def main():
    print("Iniciando servicios del Agente e-Learning...")
    
    print("Levantando FastAPI (Backend) en el puerto 8000...")
    api_process = subprocess.Popen([sys.executable, "api.py"])
    
    time.sleep(3)
    
    print("Levantando Streamlit (Frontend)...")
    ui_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
    
    try:
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\nApagando servicios de forma segura...")
        api_process.terminate()
        ui_process.terminate()
        api_process.wait()
        ui_process.wait()
        print("Servicios detenidos.")

if __name__ == "__main__":
    main()