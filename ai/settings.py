import os
from dataclasses import dataclass
from functools import lru_cache

from logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RoleProviderConfig:
    provider: str
    model: str
    hf_provider: str = ""
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: int = 60


@dataclass(frozen=True)
class LocalChromaConfig:
    path: str
    collection: str
    distance_space: str


@dataclass(frozen=True)
class AISettings:
    profile: str
    decision_llm: RoleProviderConfig
    response_llm: RoleProviderConfig
    judge_llm: RoleProviderConfig
    embedding_model: RoleProviderConfig
    chroma: LocalChromaConfig


DEFAULT_CHROMA = LocalChromaConfig(
    path="./faction_db",
    collection="factions",
    distance_space="cosine",
)

BUILT_IN_PROFILES: dict[str, AISettings] = {
    "local": AISettings(
        profile="local",
        decision_llm=RoleProviderConfig(
            provider="ollama",
            model="qwen3:4b-instruct-2507-q8_0",
        ),
        response_llm=RoleProviderConfig(
            provider="ollama",
            model="nollama/mythomax-l2-13b:Q4_K_M",
        ),
        judge_llm=RoleProviderConfig(
            provider="ollama",
            model="mistral:7b-instruct-v0.3-q8_0",
        ),
        embedding_model=RoleProviderConfig(
            provider="ollama",
            model="mxbai-embed-large",
        ),
        chroma=DEFAULT_CHROMA,
    ),
    "hugging_face__remote": AISettings(
        profile="hugging_face__remote",
        decision_llm=RoleProviderConfig(
            provider="huggingface",
            model="Qwen/Qwen3-4B-Instruct-2507:nscale",
            api_key_env="HF_API_KEY",
            base_url="https://router.huggingface.co/v1"
        ),
        response_llm=RoleProviderConfig(
            provider="huggingface",
            model="Gryphe/MythoMax-L2-13b",
            hf_provider="featherless-ai",
            api_key_env="HF_API_KEY",
        ),
        judge_llm=RoleProviderConfig(
            provider="huggingface",
            model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            api_key_env="HF_API_KEY",
        ),
        embedding_model=RoleProviderConfig(
            provider="huggingface",
            model="mixedbread-ai/mxbai-embed-large-v1",
            hf_provider="hf-inference",
            api_key_env="HF_API_KEY",
        ),
        chroma=DEFAULT_CHROMA,
    ),
    "remote_llm_local_chroma": AISettings(
        profile="remote_llm_local_chroma",
        decision_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        response_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        judge_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        embedding_model=RoleProviderConfig(
            provider="ollama",
            model="mxbai-embed-large",
        ),
        chroma=DEFAULT_CHROMA,
    ),
    "remote_llm_remote_embeddings_local_chroma": AISettings(
        profile="remote_llm_remote_embeddings_local_chroma",
        decision_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        response_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        judge_llm=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        embedding_model=RoleProviderConfig(
            provider="huggingface",
            model="",
            hf_provider="hf-inference",
            base_url="",
            api_key_env="HF_API_KEY",
        ),
        chroma=DEFAULT_CHROMA,
    ),
}


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def _override_role(prefix: str, config: RoleProviderConfig) -> RoleProviderConfig:
    provider = os.getenv(f"{prefix}_PROVIDER", config.provider)
    model = os.getenv(f"{prefix}_MODEL", config.model)
    hf_provider = os.getenv(f"{prefix}_HF_PROVIDER", config.hf_provider)
    base_url = os.getenv(f"{prefix}_BASE_URL", config.base_url)
    api_key_env = os.getenv(config.api_key_env)
    timeout_seconds = _get_env_int(f"{prefix}_TIMEOUT_SECONDS", config.timeout_seconds)

    return RoleProviderConfig(
        provider=provider,
        model=model,
        hf_provider=hf_provider,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
    )


def _override_chroma(config: LocalChromaConfig) -> LocalChromaConfig:
    return LocalChromaConfig(
        path=os.getenv("CHROMA_PATH", config.path),
        collection=os.getenv("CHROMA_COLLECTION", config.collection),
        distance_space=os.getenv("CHROMA_DISTANCE_SPACE", config.distance_space),
    )


def _apply_env_overrides(settings: AISettings) -> AISettings:
    return AISettings(
        profile=settings.profile,
        decision_llm=_override_role("DECISION", settings.decision_llm),
        response_llm=_override_role("RESPONSE", settings.response_llm),
        judge_llm=_override_role("JUDGE", settings.judge_llm),
        embedding_model=_override_role("EMBEDDING", settings.embedding_model),
        chroma=_override_chroma(settings.chroma),
    )


def _validate_role(name: str, config: RoleProviderConfig) -> None:
    if config.provider not in {"ollama", "huggingface", "openai_compatible"}:
        raise ValueError(f"Unsupported provider '{config.provider}' for role '{name}'")

    if config.model == "":
        raise ValueError(f"Missing model for role '{name}'")

    if config.provider == "openai_compatible" and config.base_url == "":
        raise ValueError(f"Missing base_url for hosted provider role '{name}'")


def _validate_settings(settings: AISettings) -> None:
    _validate_role("decision_llm", settings.decision_llm)
    _validate_role("response_llm", settings.response_llm)
    _validate_role("judge_llm", settings.judge_llm)
    _validate_role("embedding_model", settings.embedding_model)

    if settings.chroma.path == "":
        raise ValueError("Missing CHROMA_PATH")

    if settings.chroma.collection == "":
        raise ValueError("Missing CHROMA_COLLECTION")


def _log_settings(settings: AISettings) -> None:
    logger.info("Resolved AI profile: %s", settings.profile)
    logger.info(
        "Decision provider=%s model=%s",
        settings.decision_llm.provider,
        settings.decision_llm.model,
    )
    logger.info(
        "Response provider=%s model=%s",
        settings.response_llm.provider,
        settings.response_llm.model,
    )
    logger.info(
        "Judge provider=%s model=%s",
        settings.judge_llm.provider,
        settings.judge_llm.model,
    )
    logger.info(
        "Embedding provider=%s model=%s",
        settings.embedding_model.provider,
        settings.embedding_model.model,
    )
    logger.info(
        "Local Chroma path=%s collection=%s distance=%s",
        settings.chroma.path,
        settings.chroma.collection,
        settings.chroma.distance_space,
    )


@lru_cache(maxsize=1)
def get_ai_settings() -> AISettings:
    profile_name = os.getenv("AI_PROFILE", "local")
    if profile_name not in BUILT_IN_PROFILES:
        raise ValueError(f"Unknown AI profile '{profile_name}'")

    base_settings = BUILT_IN_PROFILES[profile_name]
    settings = _apply_env_overrides(base_settings)
    _validate_settings(settings)
    _log_settings(settings)
    return settings
