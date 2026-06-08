import csv
import os
from pathlib import Path
import argparse

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import uvicorn
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
logger = get_logger(__name__)
app.state.persist_enabled = False


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
    conversation_token = None

    try:
        logger.info("Awaiting first request")
        payload = await websocket.receive_text()
        
        # Received NPC interaction request
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
    if bool(app.state.persist_enabled):
        conversation_token = logger.start_conversation_trace(
            root_dir="logs/conversations",
            character_name=npc.name,
            profile=npc.ai_settings.profile,
            providers={
                "decision": f"{npc.ai_settings.decision_llm.provider}:{npc.ai_settings.decision_llm.model}",
                "response": f"{npc.ai_settings.response_llm.provider}:{npc.ai_settings.response_llm.model}",
                "judge": f"{npc.ai_settings.judge_llm.provider}:{npc.ai_settings.judge_llm.model}",
                "embedding": f"{npc.ai_settings.embedding_model.provider}:{npc.ai_settings.embedding_model.model}",
            },
            persist_enabled=True,
        )

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
        if conversation_token is not None:
            logger.reset_conversation_id(conversation_token)
        logger.info("Conversation concluded")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Game NPC Service")
    parser.add_argument("--persist", action="store_true", help="Persist per-conversation AI trace files")
    parser.add_argument("--verbose", action="store_true", help="Force log level to VERBOSE")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    host = os.getenv("AIGAME_HOST", "127.0.0.1")
    port = int(os.getenv("AIGAME_PORT", "8000"))
    configure_logging("VERBOSE" if args.verbose else None)
    app.state.persist_enabled = args.persist
    logger.info("Server startup configured: persist=%s verbose=%s", args.persist, args.verbose)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
