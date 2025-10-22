from openvoice import openvoice
import soundfile as sf

# Inicializa el modelo
model = openvoice(model_name="openvoice_v2_spanish_female", device="cpu")

# Texto a convertir
texto = (
    "Hola, soy la voz femenina de tu asistente SGSI. "
    "Estoy aquí para ayudarte con todo lo relacionado a la seguridad de la información."
)

# Generar el audio
audio, sr = model.tts(texto)

# Guardar el resultado
sf.write("voz_openvoice.wav", audio, sr)
print("✅ Voz generada y guardada como voz_openvoice.wav")
