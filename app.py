import boto3
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

# Configuración de FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Cargar modelos de Hugging Face
sentiment_model = pipeline("sentiment-analysis", model="clapAI/modernBERT-base-multilingual-sentiment")
zero_shot_model = pipeline("zero-shot-classification", model="morit/spanish_xlm_xnli")
text_gen_model = pipeline("text-generation", model="mrm8488/spanish-gpt2")
fill_mask_model = pipeline("fill-mask", model="bertin-project/bertin-roberta-base-spanish")
ner_model = pipeline("ner", grouped_entities=True)
qa_model = pipeline("question-answering", model="JoelVIU/bert-base-spanish_squad_es-TFM_2-Question-Answering")
summarization_model = pipeline("summarization", model="josmunpen/mt5-small-spanish-summarization")
translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")

# Definir clases de entrada para los endpoints
class TextInput(BaseModel):
    text: str

class ZeroShotInput(BaseModel):
    text: str
    labels: list[str]

class QAInput(BaseModel):
    question: str
    context: str

# Endpoints para cada modelo
@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    result = sentiment_model(input.text)
    save_chat_to_s3("sentiment", input.text, result)
    return result

@app.post("/zero-shot")
def zero_shot_classify(input: ZeroShotInput):
    result = zero_shot_model(input.text, candidate_labels=input.labels)
    save_chat_to_s3("zero-shot", input.text, result)
    return result

@app.post("/text-gen")
def generate_text(input: TextInput):
    result = text_gen_model(input.text)
    save_chat_to_s3("text-gen", input.text, result)
    return result

@app.post("/fill-mask")
def fill_mask(input: TextInput):
    result = fill_mask_model(input.text)
    save_chat_to_s3("fill-mask", input.text, result)
    return result

@app.post("/ner")
def named_entity_recognition(input: TextInput):
    result = ner_model(input.text)
    save_chat_to_s3("ner", input.text, result)
    return result

@app.post("/qa")
def question_answer(input: QAInput):
    result = qa_model(question=input.question, context=input.context)
    save_chat_to_s3("qa", input.question, result)
    return result

@app.post("/summarization")
def summarize_text(input: TextInput):
    result = summarization_model(input.text)
    save_chat_to_s3("summarization", input.text, result)
    return result

@app.post("/translation")
def translate_text(input: TextInput):
    result = translation_model(input.text)
    save_chat_to_s3("translation", input.text, result)
    return result

# Configuración de S3
s3_client = boto3.client("s3")
S3_BUCKET_NAME = "logs-hugginface-mateo"  # Nombre del bucket para logs

def save_chat_to_s3(category: str, user_input: str, model_response: dict):
    filename = f"historial/{category}.json"

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        chat_history = json.loads(obj['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        chat_history = []

    # Asegurar que el formato de respuesta se mantiene correcto según el modelo
    formatted_response = model_response  # Mantener la estructura original
    
    # Agregar nueva entrada al historial
    chat_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "model_response": formatted_response
    })

    # Subir historial corregido a S3
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=filename,
        Body=json.dumps(chat_history, indent=2, ensure_ascii=False),
        ContentType="application/json"
    )


# Endpoint para obtener el historial
@app.get("/history/{category}")
def get_chat_history(category: str):
    filename = f"historial/{category}.json"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        chat_history = json.loads(obj['Body'].read().decode('utf-8'))
        return chat_history
    except s3_client.exceptions.NoSuchKey:
        return {"message": "No hay historial para esta categoría."}