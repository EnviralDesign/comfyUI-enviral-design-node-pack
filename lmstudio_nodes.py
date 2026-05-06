import base64
import hashlib
import io
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _normalize_base_url(base_url):
    value = str(base_url or "").strip()
    if not value:
        value = DEFAULT_BASE_URL

    if "://" not in value:
        host = value.split("/", 1)[0].split(":", 1)[0].lower()
        private_prefixes = (
            "localhost",
            "127.",
            "10.",
            "192.168.",
            "100.",
        )
        scheme = "http" if host.startswith(private_prefixes) else "https"
        value = f"{scheme}://{value}"

    return value.rstrip("/")


def _chat_completions_url(base_url):
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _image_to_data_url(image):
    frame = image[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(frame)
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _read_api_key(api_key, api_key_env_var):
    explicit_key = str(api_key or "").strip()
    if explicit_key:
        return explicit_key

    env_var = str(api_key_env_var or "").strip()
    if env_var:
        return os.environ.get(env_var, "").strip()

    return ""


def _extract_message_content(response_json):
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    return json.dumps(response_json, indent=2)


class EnviralLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "LM Studio OpenAI-compatible base URL. Accepts https://host, https://host/v1, or full /v1/chat/completions URL.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional LM Studio API key. Sent as Authorization: Bearer <key> when provided.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "Model id from LM Studio /v1/models.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                        "tooltip": "User prompt. Can be used with an image input or by itself.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful AI assistant.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key_env_var": (
                    "STRING",
                    {
                        "default": "LMSTUDIO_API_KEY",
                        "tooltip": "Optional environment variable to read the API key from when api_key is blank.",
                    },
                ),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 32768}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600}),
                "user_agent": (
                    "STRING",
                    {
                        "default": DEFAULT_USER_AGENT,
                        "tooltip": "HTTP User-Agent sent to LM Studio. Useful when Cloudflare blocks Python's default urllib signature.",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "EnviralDesign/llm"

    @classmethod
    def IS_CHANGED(
        cls,
        base_url,
        api_key,
        model,
        prompt,
        system_prompt,
        seed,
        image=None,
        api_key_env_var="LMSTUDIO_API_KEY",
        max_tokens=1000,
        temperature=0.7,
        timeout_seconds=300,
        user_agent=DEFAULT_USER_AGENT,
        debug=False,
    ):
        m = hashlib.sha256()
        for value in (
            base_url,
            api_key,
            model,
            prompt,
            system_prompt,
            seed,
            api_key_env_var,
            max_tokens,
            temperature,
            timeout_seconds,
            user_agent,
            debug,
        ):
            m.update(str(value).encode("utf-8"))
        if image is not None:
            if hasattr(image, "detach"):
                m.update(image.detach().cpu().numpy().tobytes())
            else:
                m.update(np.asarray(image).tobytes())
        return m.hexdigest()

    def generate(
        self,
        base_url,
        api_key,
        model,
        prompt,
        system_prompt,
        seed,
        image=None,
        api_key_env_var="LMSTUDIO_API_KEY",
        max_tokens=1000,
        temperature=0.7,
        timeout_seconds=300,
        user_agent=DEFAULT_USER_AGENT,
        debug=False,
    ):
        prompt = str(prompt or "")
        system_prompt = str(system_prompt or "")
        model = str(model or "").strip()

        if not model:
            return ("Error: model is required.",)
        if image is None and not prompt.strip():
            return ("Error: provide prompt text, an image input, or both.",)

        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

        user_content = prompt
        if image is not None:
            content_parts = []
            if prompt.strip():
                content_parts.append({"type": "text", "text": prompt})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
            )
            user_content = content_parts

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
            "seed": int(seed),
        }

        url = _chat_completions_url(base_url)
        token = _read_api_key(api_key, api_key_env_var)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": str(user_agent or DEFAULT_USER_AGENT),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        if debug:
            print(f"Enviral LM Studio: POST {url}")
            print(f"Enviral LM Studio: model={model}, image={image is not None}")

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
                response_body = response.read().decode("utf-8")
            response_json = json.loads(response_body)
            return (_extract_message_content(response_json),)
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            return (f"LM Studio HTTP {err.code}: {body}",)
        except urllib.error.URLError as err:
            return (f"LM Studio connection error: {err.reason}",)
        except Exception as err:
            return (f"LM Studio error: {err}",)


NODE_CLASS_MAPPINGS = {
    "EnviralLmstudioUnified": EnviralLmstudioUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnviralLmstudioUnified": "LM Studio Unified (URL + API Key)",
}
