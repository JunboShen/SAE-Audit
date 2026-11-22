"""A minimal client for an OpenAI-compatible chat endpoint, with a call cache.

The audited LLM is a black box: SAE-Audit only sends prompts and reads text. Set the endpoint with the environment
variables SAE_AUDIT_API_BASE (for example https://api.deepseek.com/v1) and SAE_AUDIT_API_KEY, or pass them to `LLM`.
All calls use temperature 0. DeepSeek models are called with thinking disabled.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request


class LLM:
    def __init__(self, model, base_url=None, api_key=None, cache=None, extra=None, timeout=120, retries=4):
        self.model = model
        self.base_url = (base_url or os.environ.get("SAE_AUDIT_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("SAE_AUDIT_API_KEY", "")
        self.extra = dict(extra or {})                # further fields of the request body
        if model.startswith("deepseek"):
            self.extra.setdefault("thinking", {"type": "disabled"})
        self.timeout, self.retries = timeout, retries
        self.calls = 0
        self.cache_path, self._cache, self._lock = cache, {}, threading.Lock()
        if cache and os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("v") is not None:
                        self._cache.setdefault(r["k"], r["v"])

    def chat(self, system, user, max_tokens=600, key=None):
        """One call. With `key`, the answer is read from and written to the cache file. Returns None on failure."""
        if key is not None and key in self._cache:
            return self._cache[key]
        if not self.base_url:
            raise RuntimeError("no endpoint: set SAE_AUDIT_API_BASE and SAE_AUDIT_API_KEY")
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0.0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "sae-audit/0.1",
                                                  "Authorization": "Bearer " + self.api_key})
        out = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    out = (json.loads(response.read())["choices"][0]["message"].get("content") or "").strip() or None
            except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError, TimeoutError):
                out = None
            if out:
                break
            time.sleep(min(2 ** attempt, 8))
        with self._lock:
            self.calls += 1
            if out and key is not None:
                self._cache[key] = out
                if self.cache_path:
                    os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
        return out
