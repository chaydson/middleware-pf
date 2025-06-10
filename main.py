from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Dict, Any, Optional
from chat_graph import chat_graph, process_summaries
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    chat_content: str
    user_question: str

class SummaryRequest(BaseModel):
    summaries: str
    user_question: str

def is_html_content(content: str) -> bool:
    """Check if the content is HTML chat content by looking for WhatsApp chat HTML patterns"""
    # Look for common WhatsApp chat HTML patterns
    patterns = [
        r'<div class="linha"',
        r'<div class="(incoming|outgoing)"',
        r'<span class="time"',
    ]
    return any(re.search(pattern, content) for pattern in patterns)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint to process chat content and user questions using LangGraph.
    Handles both raw HTML chat content and pre-processed summaries.
    """
    try:
        # Check if the content is HTML or summaries
        if is_html_content(request.chat_content):
            # Process as HTML chat content
            initial_state = {
                "chat_content": request.chat_content,
                "user_question": request.user_question,
                "chunks": [],
                "current_chunk": 0,
                "responses": [],
                "final_response": ""
            }
            final_state = await chat_graph.ainvoke(initial_state)
            return {"response": final_state["final_response"]}
        else:
            # Process as summaries
            response = await process_summaries(request.chat_content, request.user_question)
            return {"response": response}

    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/summaries")
async def process_chat_summaries(request: SummaryRequest):
    """
    Dedicated endpoint for processing pre-processed chat summaries.
    This is an alternative to /api/chat that specifically handles summaries.
    """
    try:
        response = await process_summaries(request.summaries, request.user_question)
        return {"response": response}
    
    except Exception as e:
        error_msg = f"Error processing summaries: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
