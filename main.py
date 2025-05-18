from flask import Flask, request, jsonify
import torch
import io
import librosa
import traceback
import whisper
import os

app = Flask(__name__)

modelo = None  # Variable global para el modelo Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
model = os.environ.get("MODEL","base")
@app.route('/api/load_model', methods=['POST'])
def cargar_modelo():
    global modelo
    try:
        data = request.get_json()
        model_size = data.get('model_size', model)  # por defecto 'base'
        
        if modelo is not None:
            return jsonify({'message': 'El modelo ya está cargado. Reinicia si deseas cambiar el modelo.'}), 400

        print(f"Cargando modelo Whisper '{model_size}' en {device}...")
        modelo = whisper.load_model(model_size, device=device)
        print("Modelo cargado correctamente.")
        return jsonify({'message': f'Modelo "{model_size}" cargado exitosamente en {device}.'})
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def transcribir_audio(audio_bytes):
    global modelo
    if modelo is None:
        return None, "El modelo aún no ha sido cargado. Usa /api/load_model primero."
    try:
        print(f"Cargando audio desde los bytes...")
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
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
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No se encontró ningún archivo de audio'}), 400

        audio_file = request.files['audio_file']

        if audio_file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo de audio'}), 400

        audio_bytes = audio_file.read()
        transcripcion, error = transcribir_audio(audio_bytes)

        if error:
            return jsonify({'error': f'Error al transcribir: {error}'}), 500

        return jsonify({'transcription': transcripcion})
    except Exception as e:
        print(f"Error general: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

if __name__ == '__main__':
    print("Iniciando servicio de transcripción en http://localhost:5003")
    print("- Carga de modelo:      POST http://localhost:5003/api/load_model (JSON: {'model_size': 'large'})")
    print("- Transcripción audio:  POST http://localhost:5003/api/transcribe")
    app.run(host='0.0.0.0', port=5003, debug=True)
