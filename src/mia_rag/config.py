import json
import os
from dataclasses import asdict, dataclass, field

APP = "mia-rag"


def config_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, APP)


def default_data_dir():
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP)


@dataclass
class Settings:
    corpus_dir: str = "text"
    data_dir: str = field(default_factory=default_data_dir)
    api_base: str = "https://opencode.ai/inference/go/openai/v1"
    model: str = "deepseek-v4.1-flash"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    api_key: str = ""
    top_k: int = 10
    chunk_size: int = 1400
    overlap: int = 250
    reasoning: bool = False
    full_text_max_chars: int = 2800000

    @property
    def chunks_path(self):
        return os.path.join(self.data_dir, "chunks.jsonl")

    @property
    def embeddings_path(self):
        return os.path.join(self.data_dir, "embeddings.npy")

    @property
    def fts_path(self):
        return os.path.join(self.data_dir, "fts.db")


def load_settings():
    path = os.path.join(config_dir(), "settings.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
    return Settings(**known)


def save_settings(settings):
    os.makedirs(config_dir(), exist_ok=True)
    path = os.path.join(config_dir(), "settings.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
