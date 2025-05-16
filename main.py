from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Dict, Any
from chat_graph import chat_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    chat_content: str
    user_question: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint to process chat content and user questions using LangGraph
    """
    try:
        # Initialize the state
        initial_state = {
            "chat_content": request.chat_content,
            "user_question": request.user_question,
            "chunks": [],
            "current_chunk": 0,
            "responses": [],
            "final_response": ""
        }

        # Run the graph
        final_state = await chat_graph.ainvoke(initial_state)
        
        return {"response": final_state["final_response"]}

    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
