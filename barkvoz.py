import torch
from transformers import AutoProcessor, BarkModel
import soundfile as sf

# ======== CONFIGURACIÓN ========
MODEL_ID = "suno/bark-small"  # Usaremos Bark (voz natural femenina)
DEVICE = "cpu"  # o "cuda" si tienes GPU

# ======== CARGAR MODELO ========
print("🔄 Cargando modelo Bark (voz femenina natural)...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = BarkModel.from_pretrained(MODEL_ID).to(DEVICE)
print("✅ Modelo cargado.")

# ======== TEXTO ========
texto = (
    "Hola, soy la voz femenina de tu asistente SGSI. "
    "Estoy aquí para ayudarte con todo lo relacionado con la seguridad de la información. "
    "¿En qué puedo asistirte hoy?"
)

# ======== PROCESAR TEXTO ========
inputs = processor(text=texto, voice_preset="es_speaker_2", return_tensors="pt").to(DEVICE)

# ======== GENERAR AUDIO ========
print("🎙️ Generando voz...")
with torch.no_grad():
    audio_array = model.generate(**inputs).cpu().numpy().squeeze()

# ======== GUARDAR AUDIO ========
sf.write("voz_bark.wav", audio_array, 24000)
print("✅ Voz generada y guardada como voz_bark.wav")