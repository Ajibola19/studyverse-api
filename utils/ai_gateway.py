import os
import time
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
POLLINATIONS_TEXT_MODEL = os.getenv("POLLINATIONS_TEXT_MODEL", "openai/gpt-5.4-nano")
CLOUDFLARE_TEXT_MODEL = os.getenv(
    "CLOUDFLARE_TEXT_MODEL",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)

GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

REQUEST_TIMEOUT = int(os.getenv("AI_REQUEST_TIMEOUT", "45"))
TRANSIENT_RETRIES = int(os.getenv("AI_TRANSIENT_RETRIES", "1"))
RETRY_BACKOFF_SECONDS = float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "1.5"))
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _sleep_for_retry(attempt):
    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))


def _call_groq(prompt):
    if not GROQ_API_KEY:
        return None, None, "missing GROQ_API_KEY"

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.55,
        )
        result = llm.invoke(prompt)
        text = getattr(result, "content", None)
        if text and str(text).strip():
            return str(text).strip(), f"groq:{GROQ_MODEL}", None
        return None, f"groq:{GROQ_MODEL}", "empty response"
    except Exception as exc:
        return None, f"groq:{GROQ_MODEL}", str(exc)


def _call_gemini(prompt, model):
    if not GEMINI_API_KEY:
        return None, model, "missing GEMINI_API_KEY"

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": str(prompt)}]}],
        "generationConfig": {
            "temperature": 0.55,
            "maxOutputTokens": 5000,
        },
    }

    last_error = None
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {exc}"
            if attempt < TRANSIENT_RETRIES:
                _sleep_for_retry(attempt)
                continue
            return None, model, last_error

        if response.status_code in TRANSIENT_STATUS_CODES and attempt < TRANSIENT_RETRIES:
            last_error = f"HTTP {response.status_code}"
            _sleep_for_retry(attempt)
            continue

        if response.status_code != 200:
            return None, model, f"HTTP {response.status_code}: {response.text[:300]}"

        try:
            body = response.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            if text and str(text).strip():
                return str(text).strip(), model, None
            return None, model, "empty response"
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return None, model, f"bad response: {exc}"

    return None, model, last_error or "unknown Gemini failure"


def _call_pollinations(prompt):
    model = POLLINATIONS_TEXT_MODEL
    provider_name = f"pollinations:{model}"
    if not POLLINATIONS_API_KEY:
        return None, provider_name, "missing POLLINATIONS_API_KEY"

    url = "https://gen.pollinations.ai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the user's instructions exactly. "
                    "Return only the requested MCQ output."
                ),
            },
            {"role": "user", "content": str(prompt)},
        ],
        "temperature": 0.55,
    }

    last_error = None
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=max(REQUEST_TIMEOUT, 60),
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {exc}"
            if attempt < TRANSIENT_RETRIES:
                _sleep_for_retry(attempt)
                continue
            return None, provider_name, last_error

        if response.status_code in TRANSIENT_STATUS_CODES and attempt < TRANSIENT_RETRIES:
            last_error = f"HTTP {response.status_code}"
            _sleep_for_retry(attempt)
            continue

        if response.status_code != 200:
            return None, provider_name, f"HTTP {response.status_code}: {response.text[:300]}"

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            if text and str(text).strip():
                return str(text).strip(), provider_name, None
            return None, provider_name, "empty response"
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return None, provider_name, f"bad response: {exc}"

    return None, provider_name, last_error or "unknown Pollinations failure"


def _extract_cloudflare_text(body):
    result = body.get("result") if isinstance(body, dict) else None
    if isinstance(result, str) and result.strip():
        return result.strip()
    if isinstance(result, dict):
        for key in ("response", "text", "generated_text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _call_cloudflare(prompt):
    provider_name = f"cloudflare:{CLOUDFLARE_TEXT_MODEL}"
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        return None, provider_name, "missing Cloudflare credentials"

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_TEXT_MODEL}"
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the user's instructions exactly. "
                    "Return only the requested MCQ output."
                ),
            },
            {"role": "user", "content": str(prompt)},
        ],
        "temperature": 0.55,
        "max_tokens": 5000,
    }

    last_error = None
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=max(REQUEST_TIMEOUT, 60),
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {exc}"
            if attempt < TRANSIENT_RETRIES:
                _sleep_for_retry(attempt)
                continue
            return None, provider_name, last_error

        if response.status_code in TRANSIENT_STATUS_CODES and attempt < TRANSIENT_RETRIES:
            last_error = f"HTTP {response.status_code}"
            _sleep_for_retry(attempt)
            continue

        if response.status_code != 200:
            return None, provider_name, f"HTTP {response.status_code}: {response.text[:300]}"

        try:
            text = _extract_cloudflare_text(response.json())
            if text:
                return text, provider_name, None
            return None, provider_name, "empty response"
        except ValueError as exc:
            return None, provider_name, f"bad response: {exc}"

    return None, provider_name, last_error or "unknown Cloudflare failure"


def iter_text_candidates(prompt):
    """
    Yield usable responses in failover order.

    The caller can reject a syntactically valid response for quality reasons,
    then automatically continue to the next model/provider.
    """
    providers = [("groq", None)]
    providers.extend(("gemini", model) for model in GEMINI_MODELS)
    providers.extend([
        ("pollinations", None),
        ("cloudflare", None),
    ])

    for provider, model in providers:
        if provider == "groq":
            text, name, error = _call_groq(prompt)
        elif provider == "gemini":
            text, name, error = _call_gemini(prompt, model)
        elif provider == "pollinations":
            text, name, error = _call_pollinations(prompt)
        else:
            text, name, error = _call_cloudflare(prompt)

        if text:
            print(f"✅ AI candidate ready: {name}")
            yield {
                "text": text,
                "model": name,
                "error": None,
            }
        else:
            print(f"⚠️ AI unavailable: {name} — {error}")
