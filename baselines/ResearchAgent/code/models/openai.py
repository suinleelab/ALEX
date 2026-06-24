import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


class OpenAIClient:
    def __init__(
        self,
        model: str = 'gpt-4o',
        api_provider: str = 'openai',
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout_seconds: int = 30,
        medgemma_model: Optional[str] = None,
        medgemma_device: Optional[str] = None,
    ) -> None:
        provider = (api_provider or 'openai').strip().lower()

        if provider == 'medgemma':
            from src.agent_utils import MedGemmaClient
            mg_model = medgemma_model or 'google/medgemma-27b-text-it'
            mg_device = medgemma_device or 'cuda'
            self._client = MedGemmaClient(model_name=mg_model, device=mg_device)
            self._is_medgemma = True
        elif provider == 'openrouter':
            resolved_key = api_key or os.getenv('OPENROUTER_API_KEY')
            resolved_base_url = api_base_url or 'https://openrouter.ai/api/v1'
        else:
            resolved_key = api_key or os.getenv('OPENAI_API_KEY')
            resolved_base_url = api_base_url

        if not hasattr(self, '_is_medgemma'):
            self._is_medgemma = False

        if not self._is_medgemma:
            kwargs = {'api_key': resolved_key}
            if resolved_base_url:
                kwargs['base_url'] = resolved_base_url

            self._client = OpenAI(**kwargs).with_options(timeout=timeout_seconds)
        self.model = model
        self.api_provider = provider

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        attempt = 0

        while True:
            try:
                response = None

                if self._is_medgemma:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1000,
                    )
                else:
                    try:
                        response = self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_completion_tokens=1000,
                            temperature=1.25,
                        )
                    except Exception as first_error:
                        first_error_text = str(first_error)
                        if "max_completion_tokens" in first_error_text and "unsupported" in first_error_text.lower():
                            response = self._client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=1000,
                                temperature=1.25,
                            )
                        else:
                            raise

                content = response.choices[0].message.content if response and response.choices else ""
                return (content or "").strip()
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    return ""

                # Exponential backoff: 4s, 16s, 64s ...
                sleep_s = 4 ** attempt
                time.sleep(sleep_s)
                continue
