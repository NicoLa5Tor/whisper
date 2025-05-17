from flask import Flask, request, jsonify
import torch
import io
import librosa
import traceback
import whisper

app = Flask(__name__)

# Configuración
MODEL_SIZE = 'large'  # 'tiny', 'base', 'small', 'medium'

# Cargar el modelo Whisper al iniciar
print(f"Cargando modelo Whisper {MODEL_SIZE}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo = whisper.load_model(MODEL_SIZE, device=device)
print(f"Modelo cargado correctamente en {device}")

def transcribir_audio(audio_bytes):
    """
    Transcribe un archivo de audio usando Whisper, a partir de los bytes.
    """
    try:
        # Cargar el audio desde los bytes utilizando librosa
        print(f"Cargando audio desde los bytes...")
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Realizar la transcripción
        print("Procesando transcripción...")
        resultado = modelo.transcribe(audio)
        return resultado["text"], None
    except Exception as e:
        print(f"Error en transcripción: {str(e)}")
        traceback.print_exc()
        return None, str(e)

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    try:
        # Verificar si hay un archivo en la solicitud
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No se encontró ningún archivo de audio'}), 400
        
        audio_file = request.files['audio_file']
        
        # Verificar si se seleccionó un archivo
        if audio_file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo de audio'}), 400
        
        try:
            # Leer los bytes del archivo
            audio_bytes = audio_file.read()
            
            # Transcribir el audio
            transcripcion, error = transcribir_audio(audio_bytes)
            
            if error:
                return jsonify({'error': f'Error al transcribir: {error}'}), 500
            
            return jsonify({'transcription': transcripcion})
        
        except Exception as e:
            print(f"Error en el procesamiento: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        print(f"Error general: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

if __name__ == '__main__':
    # Ejecutar el servidor Flask
    print("Iniciando servicio de transcripción en http://localhost:5003")
    print("- API disponible en: http://localhost:5003/api/transcribe")
    app.run(host='0.0.0.0', port=5003, debug=True)