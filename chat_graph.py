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
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t[%(levelname)s]\t[%(name)s]\t\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

def log_content_preview(content: str, max_length: int = 100) -> str:
    """Create a preview of content for logging"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."

def create_chunks(state: ChatState) -> ChatState:
    """Split chat content into manageable chunks."""
    logger.info("Starting chunk creation process")
    try:
        # Parse the HTML content
        logger.info(f"Parsing HTML content (preview: {log_content_preview(state['chat_content'])})")
        messages = parse_whatsapp_html(state["chat_content"])
        if not messages:
            raise ValueError("No messages found in chat content")
        
        logger.info(f"Found {len(messages)} messages to process")

        # Create chunks
        MAX_SIZE = 20000
        chunks_of_messages = []
        current_messages = []
        chunk_sizes = []

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
                        chunk_sizes.append(len(create_chunk_text(keep_messages)))
                        logger.info(f"Created chunk with {len(keep_messages)} messages (size: {len(create_chunk_text(keep_messages))} chars)")
                        
                        current_messages = messages_to_move + [msg]
                        continue
                
                chunks_of_messages.append(current_messages)
                chunk_sizes.append(len(create_chunk_text(current_messages)))
                logger.info(f"Created chunk with {len(current_messages)} messages (size: {len(create_chunk_text(current_messages))} chars)")
                current_messages = [msg]
            else:
                current_messages.append(msg)

        if current_messages:
            chunks_of_messages.append(current_messages)
            chunk_sizes.append(len(create_chunk_text(current_messages)))
            logger.info(f"Created final chunk with {len(current_messages)} messages (size: {len(create_chunk_text(current_messages))} chars)")

        # Create formatted text chunks
        chunks = [create_chunk_text(chunk_messages) for chunk_messages in chunks_of_messages]
        
        logger.info(f"Created {len(chunks)} chunks in total")
        for i, (chunk, size) in enumerate(zip(chunks, chunk_sizes)):
            logger.info(f"Chunk {i+1}: {size} chars, {len(chunks_of_messages[i])} messages")
            logger.info(f"Chunk {i+1} preview: {log_content_preview(chunk)}")

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
    current_chunk_idx = state["current_chunk"]
    total_chunks = len(state["chunks"])
    logger.info(f"Processing chunk {current_chunk_idx + 1} of {total_chunks}")
    
    try:
        current_chunk = state["chunks"][current_chunk_idx]
        logger.info(f"Chunk {current_chunk_idx + 1} preview: {log_content_preview(current_chunk)}")
        
        # Extract chat ID from the chunk
        try:
            chat_id = current_chunk.split('\n')[0].replace('<firstMsgId>', '').replace('</firstMsgId>', '')
            logger.info(f"Processing chat ID: {chat_id}")
        except Exception as e:
            logger.warning(f"Could not extract chat ID: {str(e)}")
            chat_id = "unknown"

        user_content = (
            "This is a WhatsApp chat:\n\n"
            f"<chat>\n{current_chunk}\n</chat>\n\n"
            f"Answer the following question in portuguese pt-BR STRICTLY based on the given chat excerpt, "
            f"<question>{state['user_question']}</question>"
        )

        logger.info(f"Sending request to model for chunk {current_chunk_idx + 1}")
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
            "stream": False
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
                logger.info(f"Received response for chunk {current_chunk_idx + 1} (preview: {log_content_preview(content)})")
            else:
                content = "No response from the model"
                logger.warning(f"No response received for chunk {current_chunk_idx + 1}")

            responses = state["responses"] + [content]
            return {
                **state,
                "responses": responses,
                "current_chunk": state["current_chunk"] + 1
            }

    except Exception as e:
        logger.error(f"Error processing chunk {current_chunk_idx + 1}: {str(e)}")
        raise

def should_continue(state: ChatState) -> bool:
    """Determine if we should continue processing chunks."""
    return state["current_chunk"] < len(state["chunks"])

async def combine_responses(state: ChatState) -> ChatState:
    """Combine all chunk responses into a final response."""
    logger.info("Starting response combination process")
    try:
        # Create a prompt to combine all responses
        combined_content = "\n\n".join([
            f"Response from chunk {i+1}:\n{response}"
            for i, response in enumerate(state["responses"])
        ])
        
        logger.info(f"Combining {len(state['responses'])} responses")
        for i, response in enumerate(state["responses"]):
            logger.info(f"Response {i+1} preview: {log_content_preview(response)}")

        user_content = (
            "I have multiple responses from different parts of a WhatsApp chat. "
            "Please combine them into a single coherent response in portuguese pt-BR, "
            "maintaining all relevant information:\n\n"
            f"{combined_content}"
        )

        logger.info("Sending request to model for final combination")
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
            "stream": False
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
                logger.info(f"Final combined response (preview: {log_content_preview(final_response)})")
            else:
                final_response = "No response from the model"
                logger.warning("No final response received from model")

            return {
                **state,
                "final_response": final_response
            }

    except Exception as e:
        logger.error(f"Error combining responses: {str(e)}")
        raise

async def process_summaries(summaries: str, user_question: str) -> str:
    """Process pre-processed chat summaries and answer user questions."""
    logger.info("Starting summary processing")
    try:
        # Extract individual chat summaries
        logger.info(f"Processing summaries (preview: {log_content_preview(summaries)})")
        logger.info("--------------------------------")
        logger.info('Summaries:')
        logger.info(summaries)
        logger.info("--------------------------------")
        chat_blocks = re.split(r'=== Chat: WhatsApp Chat - \d+ ===', summaries)
        chat_blocks = [block.strip() for block in chat_blocks if block.strip()]
        logger.info("--------------------------------")
        logger.info('chat_blocks:')
        logger.info(chat_blocks)
        logger.info("--------------------------------")
        if not chat_blocks:
            raise ValueError("No valid chat summaries found in content")

        logger.info(f"Found {len(chat_blocks)} chat summaries to process")

        # Process each chat summary
        responses = []
        for i, block in enumerate(chat_blocks):
            logger.info(f"Processing summary {i+1} of {len(chat_blocks)}")
            # Clean up the summary text
            summary = re.sub(r'<think>\s*</think>', '', block).strip()
            logger.info(f"Summary {i+1} preview: {log_content_preview(summary)}")
            
            # Create prompt for the model
            user_content = (
                "This is a summary of a WhatsApp chat:\n\n"
                f"<summary>\n{summary}\n</summary>\n\n"
                f"Answer the following question in portuguese pt-BR STRICTLY based on the given summary:\n\n"
                f"<question>{user_question}</question>"
            )

            logger.info(f"Sending request to model for summary {i+1}")
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers user questions strictly based on the given chat summary."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            request_body = {
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": False
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
                    logger.info(f"Received response for summary {i+1} (preview: {log_content_preview(content)})")
                    responses.append(content)
                else:
                    logger.warning(f"No response received for summary {i+1}")
                    responses.append("No response from the model")

        # Combine all responses
        if len(responses) > 1:
            logger.info(f"Combining {len(responses)} summary responses")
            combined_content = "\n\n".join([
                f"Response from chat {i+1}:\n{response}"
                for i, response in enumerate(responses)
            ])

            for i, response in enumerate(responses):
                logger.info(f"Summary response {i+1} preview: {log_content_preview(response)}")

            user_content = (
                "I have multiple responses from different chat summaries. "
                "Please combine them into a single coherent response in portuguese pt-BR, "
                "maintaining all relevant information:\n\n"
                f"{combined_content}"
            )

            logger.info("Sending request to model for final combination")
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
                "stream": False
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
                    logger.info(f"Final combined response (preview: {log_content_preview(final_response)})")
                else:
                    final_response = "No response from the model"
                    logger.warning("No final response received from model")

            return final_response
        else:
            logger.info("Single summary response, no combination needed")
            return responses[0]

    except Exception as e:
        logger.error(f"Error processing summaries: {str(e)}")
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