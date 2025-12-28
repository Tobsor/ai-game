import csv
import os
from pathlib import Path

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from server_models import InitChatRequest

from classes.Character import Character
from logger import configure_logging, get_logger

APP_ROOT = Path(__file__).resolve().parent
CHARACTER_CSV = APP_ROOT / "data" / "character_data_cop.csv"
DEFAULT_SITUATION = (
    "{{user}} enters the village of Rack and stumbles upon {{char}}. "
    "{{char}} initiates the contact to {{user}}"
)

app = FastAPI(title="AI Game NPC Service")
configure_logging()
logger = get_logger(__name__)


def load_characters(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")
        return [row for row in reader]


CHARACTERS = load_characters(CHARACTER_CSV)

def find_character(name: str) -> dict[str, str] | None:
    lowered = name.strip().lower()
    for character in CHARACTERS:
        if (character.get("name") or "").strip().lower() == lowered:
            return character
    return None

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.websocket("/talk-to-npc")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Websocket connection received")

    payload = ""

    try:
        logger.info("Awaiting first request")
        payload = await websocket.receive_text()
        data = json.loads(payload)
        request = InitChatRequest(**data)
        logger.info("Conversation parameter initialized")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Bad payload: %s", payload)

        await websocket.send_json({"event": "error", "data": f"Invalid request: {exc}"})
        logger.info("Conversation concluded: invalid request")
        await websocket.close(code=1003)
        return

    character_data = find_character(request.name)
    if character_data is None:
        logger.error("No character with name: %s", request.name)

        await websocket.send_json({"event": "error", "data": "Character not found"})
        logger.info("Conversation concluded: error no character")
        await websocket.close(code=1008)
        return

    situation = request.situation or DEFAULT_SITUATION
    npc = Character(character_data, situation)

    try:
        logger.info("Conversation initialized with: %s", request.name)
        await websocket.send_json({"event": "start", "data": {"npc": npc.name}})
        
        await npc.initiate_conversation(socket=websocket)
        
        await websocket.send_json({"event": "end", "data": "done"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"event": "error", "data": str(exc)})
    finally:
        logger.info("Conversation concluded")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

def main() -> None:
    import uvicorn

    host = os.getenv("AIGAME_HOST", "127.0.0.1")
    port = int(os.getenv("AIGAME_PORT", "8000"))
    configure_logging()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
