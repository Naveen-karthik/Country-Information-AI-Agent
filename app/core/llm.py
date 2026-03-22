import httpx
from typing import Optional
from app.core.config import settings


async def call_mistral(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Async Mistral API client.
    Returns the assistant message content or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                settings.MISTRAL_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as e:
            # Caller handles None — no silent swallowing
            raise RuntimeError(f"Mistral API error: {e}") from e