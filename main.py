from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging
from typing import List, Dict, Any
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IP = "10.61.82.59"
OPENAI_BASE_URL = f"http://{IP}:11434/v1/chat/completions"
OPENAI_API_KEY = "teste"
CHAT_MODEL = "hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M"

app = FastAPI()

class ChatRequest(BaseModel):
    chat_content: str
    user_question: str

async def send_to_openai(chat_content: str, user_question: str) -> Dict[str, Any]:
    try:
        # Clean and encode the chat content
        chat_content = chat_content.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Prepare messages array
        try:
            chat_id = chat_content.split('\n')[0].replace('=== WhatsApp Chat - ', '').split(' ')[0]
        except Exception as e:
            logger.error(f"Error extracting chat ID: {str(e)}")
            chat_id = "unknown"

        user_content = (
            "This is a WhatsApp chat:\n\n"
            f"<chat>\n{chat_content}\n</chat>\n\n"
            f"Answer the following question in portuguese pt-BR STRICTLY based on the given chat excerpt, "
            f"quoting the chat id in this format: id_{chat_id}\n\n"
            f"<question>{user_question}</question>"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answer user questions strictly based on the given chat excerpt summaries."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        # Create request body
        request_body = {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 250
        }

        # Log request details
        logger.info(f"Sending request to API: {json.dumps(request_body, ensure_ascii=False)}")
        logger.info(f"Request URL: {OPENAI_BASE_URL}")

        # Send request using httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    OPENAI_BASE_URL,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    },
                    timeout=30.0  # Add timeout
                )

                # Log response status and body
                logger.info(f"API Response Status: {response.status_code}")
                logger.info(f"API Response Body: {response.text}")

                if response.status_code != 200:
                    error_msg = f"API request failed with status code: {response.status_code}"
                    logger.error(error_msg)
                    logger.error(f"Error response: {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=error_msg)

                response_data = response.json()
                choices = response_data.get("choices", [])
                
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    return {"response": content}
                
                return {"response": "No response from the model"}

            except httpx.TimeoutException:
                error_msg = "Request timed out"
                logger.error(error_msg)
                raise HTTPException(status_code=504, detail=error_msg)
            except httpx.RequestError as e:
                error_msg = f"Request error: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

    except Exception as e:
        error_msg = f"Error in send_to_openai: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint to process chat content and user questions
    """
    response = await send_to_openai(request.chat_content, request.user_question)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
