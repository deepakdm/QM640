"""Provider-agnostic LLM client. Swap providers via LLM_PROVIDER in .env.

Transient failures (rate limits, timeouts, 5xx) are retried by the provider SDK with exponential
backoff (`max_retries`). A call that still fails raises, and the experiment script records the
case as not yet run so a re-run fills it in; an API failure is never scored as a model error.
"""
from __future__ import annotations
import os


class LLM:
    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int = 800,
                 max_retries: int = 6, timeout_s: float = 120.0):
        self.provider = os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model or os.getenv("GENERATION_MODEL", "claude-sonnet-4-6")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(max_retries=max_retries, timeout=timeout_s)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
        self.last_model: str | None = None  # model string echoed by the provider for the last call
        self.temperature_applied: bool | None = None  # None = not yet known, False = provider refused it
        self._warned = False

    def _anthropic_create(self, system: str, user: str):
        """Send the request with temperature set. Some SDK or model versions do not accept the
        `temperature` argument (TypeError from the SDK, or a 400 from the API). Try the argument, then
        the raw request body, then no temperature, and record which one worked so the report can say so."""
        import inspect
        base = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                    messages=[{"role": "user", "content": user}])
        create = self._client.messages.create
        attempts = []
        if self.temperature_applied is not False:
            try:
                has_arg = "temperature" in inspect.signature(create).parameters
            except (TypeError, ValueError):
                has_arg = True
            if has_arg:
                attempts.append(dict(base, temperature=self.temperature))
            attempts.append(dict(base, extra_body={"temperature": self.temperature}))
        attempts.append(base)
        last = None
        for i, kw in enumerate(attempts):
            try:
                r = create(**kw)
                self.temperature_applied = ("temperature" in kw) or ("extra_body" in kw)
                if not self.temperature_applied and not self._warned:
                    self._warned = True
                    print(f"WARNING: the provider rejected temperature={self.temperature} for {self.model}; "
                          "calls ran with the provider default. State this deviation in the report.")
                return r
            except TypeError as e:
                last = e
            except Exception as e:  # anthropic.BadRequestError mentioning temperature
                if "temperature" in str(e).lower() and i < len(attempts) - 1:
                    last = e
                    continue
                raise
        raise last

    def complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            r = self._anthropic_create(system, user)
            self.last_model = getattr(r, "model", self.model)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        r = self._client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        self.last_model = getattr(r, "model", self.model)
        return r.choices[0].message.content or ""
