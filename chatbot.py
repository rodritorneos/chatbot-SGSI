import requests
import logging
import re
import random
import time
import json
import os
import re

# --------------------------
# CONFIG
# --------------------------
MODELO = "gemma3:4b-it-qat"  # tu modelo local Ollama
URL_API = "http://localhost:11434/v1/chat/completions"
SESSION = requests.Session()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# UTIL: llamada al modelo
# --------------------------
def generar_respuesta(prompt: str, max_tokens: int = 250, temperature: float = 0.25, timeout: int = 60) -> str:
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        resp = SESSION.post(URL_API, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        respuesta = data["choices"][0]["message"]["content"].strip()
        respuesta = respuesta.replace("**", "*")
        return respuesta
    except Exception as e:
        logging.error(f"❌ Error en API HTTP: {e}")
        return f"❌ Error en API HTTP: {e}"

# --------------------------
# WARM-UP
# --------------------------
def warm_up_model():
    logging.info("🔥 Warm-up del modelo (sin caché real)...")
    generar_respuesta("¿Qué es un SGSI?", max_tokens=20)
    generar_respuesta("¿Qué es ISO 27001?", max_tokens=20)
    time.sleep(0.15)

# --------------------------
# PARSER ROBUSTO
# --------------------------
def parse_question_block(raw: str):
    if not raw:
        return "", {}, None
    text = raw.replace("\r", "")
    text = re.sub(r"(📘\s*Pregunta[:：]?\s*)+", "📘 Pregunta: ", text, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    question_lines = []
    options = {}
    correct = None
    opt_re = re.compile(r'^[\*\s]*([A-Ca-c])[\)\.\:]\s*(.*?)\s*(\*+)?\s*$')
    for ln in lines:
        m = opt_re.match(ln)
        if m:
            letter = m.group(1).upper()
            rest = m.group(2).strip()
            trailing_star = m.group(3)
            if trailing_star or "*" in ln or "*" in rest:
                correct = letter
            rest = rest.replace("*", "").strip()
            rest = re.sub(r'\bOpción\s*[A-Ca-c]\b', '', rest, flags=re.IGNORECASE).strip()
            rest = re.sub(r'\(\s*[A-Ca-c]\s*\)\s*$', '', rest).strip()
            options[letter] = rest
        else:
            if options:
                last = sorted(options.keys())[-1]
                options[last] = (options[last] + " " + ln).strip()
            else:
                question_lines.append(ln)
    enunciado = " ".join(question_lines).strip()
    if correct is None:
        m = re.search(r'(?:la respuesta correcta es\s*([A-Ca-c])|correcta[:\s]*([A-Ca-c])|respuesta[:\s]*([A-Ca-c]))', text, flags=re.IGNORECASE)
        if m:
            for grp in m.groups():
                if grp and re.match(r'^[A-Ca-c]$', grp):
                    correct = grp.upper()
                    break
    if correct is None:
        m = re.search(r'\b([A-Ca-c])\)\s*(?:es correcta|correcta|es la respuesta)', text, flags=re.IGNORECASE)
        if m:
            correct = m.group(1).upper()
    if correct is None and len(options) == 3:
        correct = max(options, key=lambda k: len(options[k]))
    for k in list(options.keys()):
        options[k] = options[k].replace("*", "").strip()
    return enunciado, options, correct

# --------------------------
# FALLBACK SAFE
# --------------------------
def safe_fallback_question():
    return (
        "📘 Pregunta:\n"
        "¿Qué significa SGSI en el contexto de la seguridad de la información?\n"
        "A) Sistema General de Seguridad Integral\n"
        "B) Sistema de Gestión de Seguridad de la Información *\n"
        "C) Sistema de Gestión de Servicios"
    )

# --------------------------
# GENERADORES
# --------------------------
def generate_question_for_topic(topic: str, advanced: bool = False) -> str:
    context = (
        "SGSI significa 'Sistema de Gestión de Seguridad de la Información'. "
        "Genera UNA pregunta breve y precisa sobre el tema indicado. Debe incluir tres opciones (A, B, C). "
        "Marca la opción correcta con un asterisco '*' pegado a la letra."
    )
    adv = "Incluye referencia a ISO/IEC cuando corresponda." if advanced else "Enfócate en ISO 27001."
    tag = random.randint(1000, 999999)
    prompt = (
        f"{context}\n{adv}\n\n"
        f"Tema: {topic}\nID: {tag}\n\n"
        "Formato EXACTO de salida:\n"
        "📘 Pregunta: <enunciado>\n"
        "A) <texto>\n"
        "B) <texto>\n"
        "C) <texto>\n"
    )
    salida = generar_respuesta(prompt, max_tokens=240, temperature=0.18, timeout=60)
    q_text, opts, corr = parse_question_block(salida)
    if not opts:
        salida = safe_fallback_question()
        q_text, opts, corr = parse_question_block(salida)
    textos = list(opts.values())
    textos_sin_duplicar = list(dict.fromkeys(textos))
    for i, key in enumerate(["A","B","C"]):
        if i < len(textos_sin_duplicar):
            opts[key] = textos_sin_duplicar[i]
        else:
            opts[key] = f"Opción {key}"
    if opts and corr in opts:
        letters = list(opts.keys())
        correct_text = opts[corr]
        random_letter = random.choice(letters)
        if random_letter != corr:
            opts[random_letter], opts[corr] = opts[corr], opts[random_letter]
            corr = random_letter
    clean_q = re.sub(r"^📘\s*Pregunta[:：]?\s*", "", q_text, flags=re.IGNORECASE).strip()
    lines = [f"📘 Pregunta: {clean_q}"]
    for L in ["A", "B", "C"]:
        opt_text = opts.get(L, "")
        if corr == L:
            lines.append(f"{L}) {opt_text} *")
        else:
            lines.append(f"{L}) {opt_text}")
    return "\n".join(lines)

# --------------------------
# EXPLICACIÓN BREVE
# --------------------------
def generate_brief_explanation(question_block: str, correct_letter: str, user_letter: str) -> str:
    prompt = (
        "Explica en máximo 3 líneas por qué la opción correcta es correcta. "
        "Si el usuario respondió mal, añade una frase aclaratoria breve.\n\n"
        f"Pregunta:\n{question_block}\n\n"
        f"Opción correcta: {correct_letter}\n"
        f"Respuesta del usuario: {user_letter}"
    )
    explic = generar_respuesta(prompt, max_tokens=120, temperature=0.18, timeout=40)
    return explic.replace("*", "").strip()

# --------------------------
# INTRO TEXT
# --------------------------
INTRO_MAIN = (
    "🤖 Bienvenido al Chatbot SGSI — Capacitación en Seguridad de la Información\n"
    "Este asistente te ayudará a practicar conceptos clave de SGSI y normas ISO/IEC 27001, 27002 y 27005.\n"
    "Usa chat libre para dudas, quizzes para practicar, un examen corto y casos prácticos.\n"
)
INTRO_CHAT = (
    "💬 Chat Libre — Haz preguntas abiertas sobre SGSI e ISO (máx 4 frases por respuesta). "
    "Ideal para aclarar dudas rápidas.\n"
)
INTRO_QUIZ_BASIC = (
    "📝 Quiz Básico — Preguntas de opción múltiple centradas en requisitos esenciales de ISO 27001.\n"
)
INTRO_QUIZ_ADV = (
    "⚙️ Quiz Avanzado — Preguntas más técnicas que combinan ISO 27001, 27002 y 27005.\n"
)
INTRO_EXAM = (
    "🏁 Examen Rápido — Simulación de prueba corta: 4 preguntas, 5 puntos cada una.\n"
)
INTRO_CASE = (
    "💼 Caso Práctico — Escenarios reales. Describe acciones (2-4 líneas) y recibe evaluación práctica.\n"
)

# --------------------------
# MODO CHAT LIBRE (mejorado con historial dinámico)
# --------------------------
def modo_chat_libre():
    print("\n=== 💬 CHAT LIBRE ===")
    print(INTRO_CHAT)
    print("\n💡 El chat mantiene contexto. Escribe 'limpiar' para reiniciar o 'salir' para terminar.\n")

    historial = []
    MAX_HISTORIAL = 25  # límite de contexto en memoria

    while True:
        pregunta = input("👤 Tu mensaje: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            break
        if pregunta.lower() == "limpiar":
            historial.clear()
            print("🧹 Historial limpio.\n")
            continue

        # -----------------------
        # Construir contexto dinámico
        # -----------------------
        contexto = "\n".join([h["user"] for h in historial[-MAX_HISTORIAL:]])

        # -----------------------
        # Filtro SGSI (robusto)
        # -----------------------
        filtro_prompt = (
            "Responde solo con 'Sí' o 'No'. "
            "Considera que 'SGSI' significa 'Sistema de Gestión de Seguridad de la Información'. "
            "Cualquier tema relacionado con ISO/IEC 27001, 27002 o 27005 también cuenta como Seguridad de la Información. "
            "Si el texto se refiere directa o indirectamente a esos temas, responde 'Sí'.\n\n"
            "Si es irrelevante, absurdo o trolleo, responde 'No'.\n\n"
            f"Contexto de conversación reciente:\n{contexto}\n\n"
            f"Pregunta actual:\n{pregunta}"
        )

        filtro_resp = generar_respuesta(filtro_prompt, max_tokens=10, temperature=0.0, timeout=15)
        print(f"🔍 [DEBUG filtro_resp] => {repr(filtro_resp)}")

        if not isinstance(filtro_resp, str):
            filtro_resp = str(filtro_resp or "")

        # Limpieza de texto y verificación flexible
        texto_filtro = filtro_resp.lower()
        texto_filtro = re.sub(r"[^a-záéíóúñ]", " ", texto_filtro)

        # Validación básica SGSI
        if not re.search(r"\bs[ií]\b", texto_filtro):
            print(
                "⚠️ Tu pregunta no parece estar relacionada con SGSI/ISO. "
                "Por favor, intenta reformularla enfocándote en información, políticas, controles o normas ISO.\n"
            )
            continue

        # -----------------------
        # Generación de respuesta con historial
        # -----------------------
        prompt = (
            "Eres un asistente experto en SGSI. "
            "Recuerda que 'SGSI' siempre significa 'Sistema de Gestión de Seguridad de la Información' (ISMS en inglés). "
            "Responde en español de forma clara y práctica (máx. 4 frases). "
            "Si hay siglas, explica su significado antes de responder.\n\n"
        )

        mensajes = [{"role": "system", "content": prompt}]
        for h in historial[-MAX_HISTORIAL:]:
            mensajes.append({"role": "user", "content": h["user"]})
            mensajes.append({"role": "assistant", "content": h["bot"]})
        mensajes.append({"role": "user", "content": pregunta})

        respuesta = generar_respuesta(prompt + f"Pregunta: {pregunta}", max_tokens=320, temperature=0.3)
        print("\n🤖", respuesta, "\n")

        historial.append({"user": pregunta, "bot": respuesta})

# =======================
# Modo Quiz
# =======================
def modo_quiz(basico: bool = True):
    print("\n=== 📝 QUIZ ===")
    print(INTRO_QUIZ_BASIC if basico else INTRO_QUIZ_ADV)
    temas = [
        "Requisitos ISO 27001", "Política de seguridad", "Controles de acceso",
        "Gestión de incidentes", "Auditoría interna", "Clasificación de la información", "Gestión de riesgos"
    ]
    puntaje = 0

    while True:
        # Selección de tema
        print("\nTemas disponibles:")
        for i, t in enumerate(temas, 1):
            print(f"{i}) {t}")
        sel = input("\n👉 Elige número o escribe tu propio tema (o 'salir'): ").strip()
        if sel.lower() == "salir":
            break
        tema = temas[int(sel)-1] if sel.isdigit() and 1 <= int(sel) <= len(temas) else sel

        # Bucle de preguntas sobre el mismo tema
        while True:
            raw = generate_question_for_topic(tema, advanced=not basico)
            q_text, opts, corr = parse_question_block(raw)
            if not opts:
                raw = safe_fallback_question()
                q_text, opts, corr = parse_question_block(raw)

            clean_q = re.sub(r"^📘\s*Pregunta[:：]?\s*", "", q_text, flags=re.IGNORECASE).strip()
            print(f"\n📘 {clean_q}\n")
            for L in ["A", "B", "C"]:
                print(f"{L}) {opts.get(L,'')}")

            while True:
                user = input("\n👤 Tu opción (A/B/C o 'salir'): ").strip().upper()
                if user.lower() == "salir":
                    return
                if user not in ["A","B","C"]:
                    print("❌ Opción no válida. Elige A, B o C.")
                    continue
                break

            if user == corr:
                print("\n✅ Correcto!")
            else:
                print(f"\n❌ Incorrecto. La correcta era {corr}) {opts.get(corr,'')}")

            explic = generate_brief_explanation(raw, corr, user)
            print(f"\n📖 {explic}\n")

            if user == corr:
                puntaje += 5
            print(f"🔥 Puntaje: {puntaje}\n")

            seguir = input("¿Otra pregunta sobre este tema? (sí/no): ").strip().lower()
            if seguir not in ("sí", "si", "s", "y", "yes"):
                break

# ==========================
# Modo Examen rápido
# ==========================
def modo_examen_rapido():
    print("\n=== 🏁 EXAMEN RÁPIDO ===")
    print(INTRO_EXAM)
    puntaje = 0
    used_q_texts = set()

    pool = [
        "Controles de acceso en ISO 27001",
        "Gestión de incidentes según ISO 27001",
        "Clasificación de la información (SGSI)",
        "Gestión de riesgos (ISO 27005)",
        "Política de seguridad",
        "Auditoría interna"
    ]

    for i in range(1, 5):
        candidates = [t for t in pool if t not in used_q_texts] or pool
        tema = random.choice(candidates)

        attempts = 0
        while True:
            raw = generate_question_for_topic(tema, advanced=True)
            q_text, opts, corr = parse_question_block(raw)
            clean_q = re.sub(r"^📘\s*Pregunta[:：]?\s*", "", q_text, flags=re.IGNORECASE).strip()
            attempts += 1

            if not clean_q or clean_q in used_q_texts:
                if attempts > 4:
                    raw = safe_fallback_question()
                    q_text, opts, corr = parse_question_block(raw)
                    clean_q = re.sub(r"^📘\s*Pregunta[:：]?\s*", "", q_text, flags=re.IGNORECASE).strip()
                else:
                    continue

            textos = list(opts.values())
            textos_sin_duplicar = list(dict.fromkeys(textos))
            for j, key in enumerate(["A","B","C"]):
                if j < len(textos_sin_duplicar):
                    opts[key] = textos_sin_duplicar[j]
                else:
                    opts[key] = f"Opción {key}"

            if opts and corr in opts:
                letters = list(opts.keys())
                correct_text = opts[corr]
                random_letter = random.choice(letters)
                if random_letter != corr:
                    opts[random_letter], opts[corr] = opts[corr], opts[random_letter]
                    corr = random_letter

            used_q_texts.add(clean_q)
            break

        print(f"\n📘 Pregunta {i}: {clean_q}\n")
        for L in ["A", "B", "C"]:
            print(f"{L}) {opts.get(L, '')}")

        while True:
            user = input("\n👤 Tu opción (A/B/C o 'salir'): ").strip().upper()
            if user.lower() == "salir":
                print("\n🚪 Examen interrumpido por el usuario.")
                print(f"\n🏆 Puntaje final: {puntaje}/20\n")
                return
            if user not in ["A","B","C"]:
                print("❌ Opción no válida. Elige A, B o C.")
                continue
            break

        if user == corr:
            print("\n✅ Correcto!")
            puntaje += 5
        else:
            print(f"\n❌ Incorrecto. La correcta era {corr}) {opts.get(corr,'')}")

        explic = generate_brief_explanation(raw, corr, user)
        print(f"\n📖 Explicación breve:\n{explic}\n")
        time.sleep(0.3)

    print(f"\n🏆 Puntaje final: {puntaje}/20\n")

# ==========================
# Modo Caso práctico
# ==========================
def modo_caso_practico():
    print("\n=== 💼 CASO PRÁCTICO ===")
    print(INTRO_CASE)
    temas = [
        "Filtración de datos personales",
        "Acceso no autorizado",
        "Falla en control de contraseñas",
        "Ransomware en servidores",
        "Pérdida de respaldo de información"
    ]
    used_q_texts = set()
    
    while True:
        tema = random.choice(temas)
        attempts = 0
        while True:
            # Generamos SOLO el escenario sin acciones ni pistas
            prompt = (
                f"Genera un escenario breve (3-4 líneas) sobre '{tema}' en el contexto de un SGSI. "
                "No incluyas soluciones, pasos o acciones de evaluación. Solo describe la situación."
            )
            escenario_raw = generar_respuesta(prompt, max_tokens=200, temperature=0.2)
            esc_clean = escenario_raw.strip()
            attempts += 1
            if esc_clean not in used_q_texts or attempts > 4:
                used_q_texts.add(esc_clean)
                break
        
        print(f"\n🔔 Escenario:\n{esc_clean}\n")
        respuesta = input("👤 Describe brevemente qué acciones tomarías (2-4 líneas o 'salir'): ").strip()
        if respuesta.lower() == "salir":
            break

        # Evaluamos la respuesta del usuario
        eval_prompt = (
            f"Escenario: {esc_clean}\n"
            f"Respuesta del usuario: {respuesta}\n\n"
            "Evalúa la respuesta con: ✅ Correcta, ⚠️ Parcial o ❌ Incorrecta. "
            "Da una retroalimentación breve y práctica (máx. 2 líneas), sin revelar otras acciones posibles."
        )
        evaluacion = generar_respuesta(eval_prompt, max_tokens=140, temperature=0.2)
        print(f"\n📊 Evaluación:\n{evaluacion}\n")
        
        seguir = input("¿Otro caso? (sí/no): ").strip().lower()
        if seguir not in ("sí", "si", "s", "y", "yes"):
            break

# --------------------------
# MAIN
# --------------------------
def main():
    warm_up_model()
    print(INTRO_MAIN)
    while True:
        print("=== MENÚ PRINCIPAL ===")
        print("1) Chat libre")
        print("2) Quiz básico (ISO 27001)")
        print("3) Quiz avanzado (27001/27002/27005)")
        print("4) Examen rápido")
        print("5) Caso práctico")
        print("6) Salir")
        opcion = input("👉 Elige una opción: ").strip()
        if opcion == "1":
            modo_chat_libre()
        elif opcion == "2":
            modo_quiz(basico=True)
        elif opcion == "3":
            modo_quiz(basico=False)
        elif opcion == "4":
            modo_examen_rapido()
        elif opcion == "5":
            modo_caso_practico()
        elif opcion == "6":
            print("👋 ¡Gracias por usar el Chatbot SGSI! ¡Éxitos en tu presentación!")
            break
        else:
            print("❌ Opción no válida. Intenta nuevamente.\n")

if __name__ == "__main__":
    main()

    