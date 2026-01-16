import time
import os
import base64
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Multimodal RAG Backend")

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1")
client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)

history_db: Dict[str, List[Dict]] = {}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def get_history(session_id: str) -> List[Dict]:
    return history_db.get(session_id, [])


def update_history(session_id: str, message: str, max_history: int = 5):
    if session_id not in history_db:
        history_db[session_id] = []

    history_db[session_id].append(message)

    if len(history_db[session_id]) >= max_history:
        history_db[session_id] = history_db[session_id][-max_history:]


@app.post("/ask")
async def ask(session_id: str, prompt: str, file: UploadFile | None = File(default=None)):
    try:
        models = client.models.list()
        model_id = models.data[0].id
        content = [{"type": "text", "text": prompt}]

        if file:
            image_data = await file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

            content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            )

        user_message = {"role": "user", "content": content}
        user_history = get_history(session_id)
        full_context = user_history + [user_message]
        response = client.chat.completions.create(
            model=model_id, messages=full_context, max_tokens=1024, temperature=0.8
        )
        response_text = response.choices[0].message.content
        ai_message = {"role": "assistant", "content": response_text}

        update_history(session_id, user_message)
        update_history(session_id, ai_message)

        return {
            "session_id": session_id,
            "response": response_text,
            "history_length": len(get_history(session_id)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-diagram")
async def analyze_diagram(prompt: str, file: UploadFile = File(...)):
    """
    Logic: This endpoint receives an image and a text prompt,
    converts the image to base64, and forwards it to the VLM.
    """
    try:
        models = client.models.list()
        model_id = models.data[0].id

        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=1024, temperature=0.0
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
