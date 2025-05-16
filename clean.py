import re
from datetime import datetime

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------#
# utilitários auxiliares
# ---------------------------------------------------------------------------#

_TS_CLEAN = re.compile(r"\s+")


def _clean_timestamp(raw: str) -> str:
    """
    Normaliza carimbos como “2023-01-08 19:38:20-0300”
    para ISO 8601 (“2023-01-08T19:38:20-03:00”).
    Se não conseguir entender, devolve o texto original.
    """
    txt = _TS_CLEAN.sub(" ", raw).strip()
    try:
        # 2023-01-08 19:38:20-0300
        dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S%z")
        return dt.isoformat()
    except ValueError:
        pass
    try:
        # 2023-01-08 19:38:20-0300
        dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S %z")
        return dt.isoformat()
    except ValueError:
        return txt


def _extract_text_nodes(tag: Tag) -> str:
    """
    Concatena somente os nós-texto “soltos” dentro de *tag*.
    Ignora <br>, <span>, etc.
    """
    parts = [t.strip() for t in tag.contents
             if isinstance(t, NavigableString) and t.strip()]
    return " ".join(parts)


# ---------------------------------------------------------------------------#
# parser principal
# ---------------------------------------------------------------------------#

def parse_whatsapp_html(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: list[dict] = []

    for block in soup.select("div.linha"):
        msg_div = block.find("div", class_=["incoming", "outgoing"])
        if msg_div is None:                 # linha de sistema / vazia
            continue

        if msg_div.find("div", class_=["systemmessage"]):
            continue

        msg_id = block.get('id')

        direction = ("received"
                     if "incoming" in msg_div["class"]
                     else "sent")

        forwarded = False
        if msg_div.find("span", class_=["fwd"]):
            forwarded = True

        name = msg_div.find("span").get_text(" ", strip=True)
        #print(msg_div.prettify())
        timestamp_span = msg_div.find("span", class_="time")
        if not timestamp_span:
            print("No timestamp found")
            continue
        timestamp_raw = timestamp_span.get_text(" ", strip=True)

        timestamp = _clean_timestamp(timestamp_raw)

        # -------------------------------------------------------------------#
        # 1) transcrição de áudio (fica em <i> … </i>)
        # -------------------------------------------------------------------#
        content = ""
        kind = ""
        i_tag = msg_div.find("i")
        if i_tag and msg_div.find("div", class_=["audioImg"]):
            content = i_tag.get_text(" ", strip=True)
            kind = "audio transcription"


        # -------------------------------------------------------------------#
        # 2) anexo (áudio / vídeo / outro)
        # -------------------------------------------------------------------#
        if not content:
            #kind = "other"

            # áudio ➜ ícone <div class="audioImg">
            if msg_div.find("div", class_="audioImg"):
                kind = "audio"
                content = f" "
            if msg_div.find("div", class_="imageImg"):
                kind = "image"
                content = f" "
            if msg_div.find("div", class_="videoImg"):
                kind = "video"
                content = f" "
            # vídeo ou imagem ➜ thumbnail <img class="thumb" … title="video|image">
            else:
                thumb = msg_div.find("img", class_="thumb")
                if thumb and thumb.get("title"):
                    title = thumb["title"].lower()
                    if "video" in title:
                        kind = "video"
                        content = f" "
                    elif "image" in title:
                        kind = "image"
                        content = f" "

            #a_tag = msg_div.find("a", href=True)
            #if a_tag:
            #    content = f" "

        # -------------------------------------------------------------------#
        # 3) texto “puro”
        # -------------------------------------------------------------------#
        if not content:
            content = _extract_text_nodes(msg_div)
            kind = "text"

        # ainda vazio? provavelmente só thumbs ou attachments sem link → pula
        if not content:
            continue


        out.append(
            {
                "id":msg_id,
                "direction": direction,
                "name": name,
                "timestamp": timestamp,
                "content": content,
                "kind": kind,
                "forwarded": forwarded
            }
        )

    return out


def create_chunk_text(messages):
    """Create formatted text from a list of messages"""
    if not messages:
        return ""
        
    chunk_text = f"<firstMsgId>{messages[0]['id']}</firstMsgId>"
    for msg in messages:
        chunk_text += format_message(msg)
    return chunk_text

def find_largest_time_gap(messages):
    """Find the index with largest time gap between messages"""
    if len(messages) < 2:
        return 0
    
    # Calculate time gaps between messages
    gaps = []
    for i in range(len(messages) - 1):
        time1 = datetime.fromisoformat(messages[i]['timestamp'])
        time2 = datetime.fromisoformat(messages[i + 1]['timestamp'])
        gap_seconds = (time2 - time1).total_seconds()
        gaps.append((gap_seconds, i))
    
    # Find the largest gap
    largest_gap_seconds = 0
    largest_gap_index = 0
    
    for gap_seconds, index in gaps:
        if gap_seconds > largest_gap_seconds:
            largest_gap_seconds = gap_seconds
            largest_gap_index = index
    
    return largest_gap_index + 1 if gaps else 0

def format_message(msg):
    """Format a single message"""
    fwd_text = ' (forwarded)' if msg['forwarded'] else ''
    kind_text = '' if msg['kind'] == 'text' else f"[{msg['kind']}]"
    return f"\n<message>\n{msg['direction']} from {msg['name']} at {msg['timestamp']}{fwd_text}\nContent:{kind_text} {msg['content']}\n</message>"

# Main chunking code
# ---------------------------------------------------------------------------#
# exemplo de uso
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    with open("./tmp/tmpikfi6yu5.html", encoding="utf-8") as f:
        html = f.read()

    msgs = parse_whatsapp_html(html)
    if not msgs:
        print("Sem mensagens")
        exit()

    MAX_SIZE = 25000
    chunks_of_messages = []  # Will store lists of messages
    current_messages = []

    print(len(msgs), "messages found")
    
    for msg in msgs:
        # Check if adding message would exceed limit
        test_chunk = create_chunk_text(current_messages + [msg])
        
        if len(test_chunk) > MAX_SIZE:
            if len(current_messages) >= 2:
                # Look at last 10 messages for biggest gap
                look_back = min(50, len(current_messages))
                messages_to_check = current_messages[-look_back:]
                split_idx = find_largest_time_gap(messages_to_check)
                
                if split_idx > 0:
                    # Adjust split_idx to account for messages we didn't check
                    actual_split_idx = len(current_messages) - look_back + split_idx
                    
                    # Split messages into two groups

                    print(f"Content start: {current_messages[actual_split_idx]['content'][:20]}...")
                    print(f"Splitting at message ID: {current_messages[actual_split_idx]['id']}")

                    keep_messages = current_messages[:actual_split_idx]
                    messages_to_move = current_messages[actual_split_idx:]
                    
                    # Store first group as a chunk
                    chunks_of_messages.append(keep_messages)
                    print(f"Created chunk with {len(keep_messages)} messages")
                    
                    # Continue with second group
                    current_messages = messages_to_move + [msg]
                    continue
            
            # If no good split point, start new chunk
            chunks_of_messages.append(current_messages)
            print(f"Created chunk with {len(current_messages)} messages")
            current_messages = [msg]
        else:
            current_messages.append(msg)
    
    # Add final chunk
    if current_messages:
        chunks_of_messages.append(current_messages)
        print(f"Created final chunk with {len(current_messages)} messages")

    # Create formatted text chunks
    chunks = [create_chunk_text(chunk_messages) for chunk_messages in chunks_of_messages]

    # Print all chunks first
    print("\nAll Chunks Content:")
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"CHUNK {i+1}")
        print('='*50)
        print(chunk)
        print('='*50)

    # Print final stats
    print("\nFinal Statistics:")
    total_messages = 0 
    for i, chunk in enumerate(chunks):
        # Count messages in this chunk
        messages_in_chunk = chunk.count('<message>')
        total_messages += messages_in_chunk
        
        print(f"\n=== Chunk {i+1} ===")
        print(f"Size: {len(chunk)} characters ({len(chunk)/MAX_SIZE*100:.1f}%)")
        print(f"Messages in chunk: {messages_in_chunk}")

        # Print first message in chunk
        first_msg = chunks_of_messages[i][0]
        print(f"First message: {first_msg['name']} - {first_msg['timestamp']}")
        print(f"Content: {first_msg['content'][:100]}...")
        
        # Verify counts match
        if messages_in_chunk != len(chunks_of_messages[i]):
            print("WARNING: Message count mismatch!")

    print(f"\nTotal messages: {total_messages}")
    print(f"Original message count: {len(msgs)}")