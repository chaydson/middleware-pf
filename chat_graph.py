from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from clean import parse_whatsapp_html, create_chunk_text, find_largest_time_gap
import httpx
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IP = "10.61.82.59"
OPENAI_BASE_URL = f"http://{IP}:11434/v1/chat/completions"
OPENAI_API_KEY = "teste"
CHAT_MODEL = "hf.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M"

class ChatState(TypedDict):
    """State for the chat processing graph."""
    chat_content: str
    user_question: str
    chunks: List[str]
    current_chunk: int
    responses: List[str]
    final_response: str

def create_chunks(state: ChatState) -> ChatState:
    """Split chat content into manageable chunks."""
    print("create_chunks")
    try:
        # Parse the HTML content
        messages = parse_whatsapp_html(state["chat_content"])
        if not messages:
            raise ValueError("No messages found in chat content")

        # Create chunks
        #MAX_SIZE = 25000
        MAX_SIZE = 1000
        chunks_of_messages = []
        current_messages = []

        for msg in messages:
            test_chunk = create_chunk_text(current_messages + [msg])
            
            if len(test_chunk) > MAX_SIZE:
                if len(current_messages) >= 2:
                    look_back = min(50, len(current_messages))
                    messages_to_check = current_messages[-look_back:]
                    split_idx = find_largest_time_gap(messages_to_check)
                    
                    if split_idx > 0:
                        actual_split_idx = len(current_messages) - look_back + split_idx
                        keep_messages = current_messages[:actual_split_idx]
                        messages_to_move = current_messages[actual_split_idx:]
                        
                        chunks_of_messages.append(keep_messages)
                        current_messages = messages_to_move + [msg]
                        continue
                
                chunks_of_messages.append(current_messages)
                current_messages = [msg]
            else:
                current_messages.append(msg)

        if current_messages:
            chunks_of_messages.append(current_messages)

        # Create formatted text chunks
        chunks = [create_chunk_text(chunk_messages) for chunk_messages in chunks_of_messages]
        
        return {
            **state,
            "chunks": chunks,
            "current_chunk": 0,
            "responses": []
        }
    except Exception as e:
        logger.error(f"Error in create_chunks: {str(e)}")
        raise

async def process_chunk(state: ChatState) -> ChatState:
    """Process a single chunk of chat content."""
    print("process_chunk")
    try:
        current_chunk = state["chunks"][state["current_chunk"]]
        
        # Extract chat ID from the chunk
        try:
            chat_id = current_chunk.split('\n')[0].replace('<firstMsgId>', '').replace('</firstMsgId>', '')
        except Exception as e:
            logger.error(f"Error extracting chat ID: {str(e)}")
            chat_id = "unknown"

        user_content = (
            "This is a WhatsApp chat:\n\n"
            f"<chat>\n{current_chunk}\n</chat>\n\n"
            f"Answer the following question in portuguese pt-BR STRICTLY based on the given chat excerpt, "
            f"quoting the chat id in this format: id_{chat_id}\n\n"
            f"<question>{state['user_question']}</question>"
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

        request_body = {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 250
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENAI_BASE_URL,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"API request failed with status code: {response.status_code}")

            response_data = response.json()
            choices = response_data.get("choices", [])
            
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "No response from the model")
            else:
                content = "No response from the model"

            responses = state["responses"] + [content]
            
            return {
                **state,
                "responses": responses,
                "current_chunk": state["current_chunk"] + 1
            }

    except Exception as e:
        logger.error(f"Error in process_chunk: {str(e)}")
        raise

def should_continue(state: ChatState) -> bool:
    """Determine if we should continue processing chunks."""
    return state["current_chunk"] < len(state["chunks"])

async def combine_responses(state: ChatState) -> ChatState:
    """Combine all chunk responses into a final response."""
    print("combine_responses")
    try:
        # Create a prompt to combine all responses
        combined_content = "\n\n".join([
            f"Response from chunk {i+1}:\n{response}"
            for i, response in enumerate(state["responses"])
        ])

        user_content = (
            "I have multiple responses from different parts of a WhatsApp chat. "
            "Please combine them into a single coherent response in portuguese pt-BR, "
            "maintaining all relevant information:\n\n"
            f"{combined_content}"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that combines multiple responses into a single coherent answer."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        request_body = {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 500
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENAI_BASE_URL,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"API request failed with status code: {response.status_code}")

            response_data = response.json()
            choices = response_data.get("choices", [])
            
            if choices:
                message = choices[0].get("message", {})
                final_response = message.get("content", "No response from the model")
            else:
                final_response = "No response from the model"

            return {
                **state,
                "final_response": final_response
            }

    except Exception as e:
        logger.error(f"Error in combine_responses: {str(e)}")
        raise

def create_chat_graph() -> Graph:
    """Create the chat processing graph."""
    # Create the graph
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("create_chunks", create_chunks)
    workflow.add_node("process_chunk", process_chunk)
    workflow.add_node("combine_responses", combine_responses)

    # Add edges
    workflow.add_edge("create_chunks", "process_chunk")
    workflow.add_conditional_edges(
        "process_chunk",
        should_continue,
        {
            True: "process_chunk",
            False: "combine_responses"
        }
    )

    # Set entry and exit points
    workflow.set_entry_point("create_chunks")
    workflow.set_finish_point("combine_responses")

    return workflow.compile()

# Create the graph instance
chat_graph = create_chat_graph() 