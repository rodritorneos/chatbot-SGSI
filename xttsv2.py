from TTS.api import TTS

# Mostrar modelos disponibles
print(TTS().list_models())

# Descargar y usar el modelo XTTS v2 en español
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)

texto = "Hola, soy tu asistente SGSI. Estoy aquí para ayudarte en todo momento."
tts.tts_to_file(text=texto, file_path="voz_xtts.wav", speaker="female", language="es")
