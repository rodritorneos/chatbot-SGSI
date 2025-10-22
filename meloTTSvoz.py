import torch
from melo.api import TTS

# --------------------------------------------------------
# ⚙️ OPTIMIZACIÓN CPU (Windows-compatible)
# --------------------------------------------------------
torch.set_num_threads(6)           # Ajusta según tus núcleos (4-8 ideal)
torch.set_num_interop_threads(6)

# --------------------------------------------------------
# 🚀 CARGA DEL MODELO
# --------------------------------------------------------
device = "cpu"
model = TTS(language="ES", device=device)

speaker_ids = model.hps.data.spk2id
print("Voces disponibles:", speaker_ids)

# --------------------------------------------------------
# 💬 TEXTO DE PRUEBA
# --------------------------------------------------------
text = ("Hola, soy la voz femenina de tu chatbot SGSI. Qué gusto saludarte. Estoy aquí para ayudarte con todo lo relacionado a la seguridad de la información. Puedes preguntarme sobre políticas, normativas, buenas prácticas y más. Estoy disponible las 24 horas del día para asistirte. ¡Comencemos cuando quieras!")
output_path = "melo_out.wav"

# --------------------------------------------------------
# 🎧 GENERACIÓN DE AUDIO
# --------------------------------------------------------
model.tts_to_file(text, speaker_ids["ES"], output_path, speed=1.1)
print("✅ Audio generado y guardado en:", output_path)