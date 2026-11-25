"""Fixed OpenRouter free-model variant of the OCR/text architecture."""
from app.providers.cloud import TextProvider
from app.providers.base import ProviderError


class OpenRouterProvider(TextProvider):
    provider = "ocr_openrouter"
    key_name = "OPENROUTER_API_KEY"
    model_name = "OPENROUTER_MODEL"
    default_model = "google/gemma-4-26b-a4b-it:free"

    async def request(self, client, messages, key):
        if not self.model.endswith(":free"):
            raise ProviderError("This experiment adapter requires a fixed :free model")
        response = await client.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": self.model, "messages": messages, "temperature": 0,
                  "max_tokens": 8192, "stream": False, "reasoning": {"enabled": False}})
        response.raise_for_status()
        data = response.json()
        details = data.get("usage", {})
        usage = {"input_tokens": details.get("prompt_tokens", 0) or 0,
                 "output_tokens": details.get("completion_tokens", 0) or 0,
                 "thinking_tokens": (details.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                 "usage_reported": all(details.get(k) is not None for k in ("prompt_tokens", "completion_tokens"))}
        details["served_model"] = data.get("model")
        details["served_provider"] = data.get("provider")
        return data["choices"][0]["message"].get("content", ""), usage, details

    async def extract(self, *args, **kwargs):
        result = await super().extract(*args, **kwargs)
        costs = [a.get("token_details", {}).get("cost") for a in result["metadata"]["attempts"]]
        result["metadata"]["reported_cost_usd"] = sum(costs) if all(c is not None for c in costs) else None
        result["metadata"]["price_scope"] = "free quota trial; not a production price guarantee"
        return result
