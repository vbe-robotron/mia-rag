import json
import os
import urllib.request
import uuid


def resolve_api_key(settings):
    if settings.api_key:
        return settings.api_key.strip()
    env = os.environ.get("OPENCODE_API_KEY")
    if env:
        return env.strip()
    auth = os.path.expanduser("~/.local/share/opencode/auth.json")
    if os.path.exists(auth):
        try:
            with open(auth, encoding="utf-8") as f:
                data = json.load(f)
            for name in ("opencode-go", "opencode"):
                entry = data.get(name)
                if isinstance(entry, dict) and entry.get("key"):
                    return entry["key"]
        except Exception:
            pass
    raise RuntimeError(
        "OpenCode Go API key not found. Set it in Settings, or export "
        "OPENCODE_API_KEY."
    )


class ApiClient:
    def __init__(self, settings):
        self.settings = settings
        self.session = str(uuid.uuid4())

    def chat(self, messages, max_tokens=6000, temperature=0.2):
        key = resolve_api_key(self.settings)
        url = self.settings.api_base.rstrip("/") + "/chat/completions"
        content = ""
        for attempt in range(3):
            payload = json.dumps(
                {
                    "model": self.settings.model,
                    "messages": messages,
                    "max_tokens": max_tokens * (2**attempt),
                    "temperature": temperature,
                }
            ).encode()
            headers = {
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "User-Agent": "mia-rag/0.1",
            }
            if "opencode.ai" in self.settings.api_base:
                headers["x-opencode-session"] = self.session
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read())
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if content.strip() and choice.get("finish_reason") != "length":
                return content
        return content
