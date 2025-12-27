from pydantic import BaseModel, Field

class InitChatRequest(BaseModel):
    name: str = Field(..., description="Character name")
    situation: str | None = None

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User input for the NPC")
    end: bool = Field(default=False, description="Ends the conversation if true")