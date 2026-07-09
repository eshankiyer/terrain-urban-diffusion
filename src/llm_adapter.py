"""Pluggable language-model backend for the planner tools.

Every text task in this system (parsing a planner's request, drafting a
report, reading a section of municipal code) needs a language model, and
the choice of model should not be baked into any one module. This adapter
defines one small interface, LLMBackend, with a single generate method,
and ships three implementations:

  OllamaBackend   a local model served by Ollama, default tag "gemma3".
                  Nothing leaves the machine, which matters when the input
                  is a city's unpublished draft code or resident comments.
  OpenAIBackend   any OpenAI-compatible HTTP endpoint (OpenAI, vLLM,
                  llama.cpp server, Together, and so on) via an API key.
  EchoBackend     deterministic, offline, returns a canned reply. Used by
                  tests and as the fallback when no model is reachable, so
                  importing this module never requires a network.

The default returned by default_backend() is Ollama on gemma3, chosen
because it runs on a laptop, is permissively licensed, and keeps sensitive
planning inputs local. Callers that want a hosted model pass their own
backend; nothing in the planner modules names a provider.

A backend is just an object with generate(prompt, system, ...) -> str, so
a caller can also pass any function or object of their own. as_backend
wraps a bare callable.

Dependencies: standard library only.
"""

import json
import os
import urllib.error
import urllib.request


class LLMBackend:
    """Interface: implement generate and, optionally, available."""

    name = "abstract"

    def generate(self, prompt, system=None, temperature=0.0,
                 max_tokens=1024, as_json=False):
        raise NotImplementedError

    def available(self):
        """Return True if a real call would plausibly succeed."""
        return True


class EchoBackend(LLMBackend):
    """Offline, deterministic. Returns a fixed reply, or a supplied one.

    With as_json=True it returns a minimal valid object so callers that
    json.loads the reply do not crash in tests or offline runs.
    """

    name = "echo"

    def __init__(self, reply=None):
        self._reply = reply

    def generate(self, prompt, system=None, temperature=0.0,
                 max_tokens=1024, as_json=False):
        if self._reply is not None:
            return self._reply
        if as_json:
            return "{}"
        return "[offline: no language model configured]"


class OllamaBackend(LLMBackend):
    """Local Ollama server. Default model gemma3.

    The default host and model can be overridden by the environment
    variables PLANNER_OLLAMA_URL and PLANNER_LLM_MODEL, so deployment can
    point at a different local model without code changes.
    """

    name = "ollama"

    def __init__(self, model=None, url=None, timeout=60):
        self.model = model or os.environ.get("PLANNER_LLM_MODEL", "gemma3")
        self.url = (url or os.environ.get("PLANNER_OLLAMA_URL",
                                          "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def available(self):
        try:
            req = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def generate(self, prompt, system=None, temperature=0.0,
                 max_tokens=1024, as_json=False):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "stream": False,
                   "options": {"temperature": temperature,
                               "num_predict": max_tokens}}
        if as_json:
            payload["format"] = "json"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + "/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]


class OpenAIBackend(LLMBackend):
    """Any OpenAI-compatible chat-completions endpoint.

    base_url defaults to the public OpenAI API; point it at a local vLLM
    or llama.cpp server to keep inference on-premises with a hosted-style
    interface. The API key is read from the api_key argument or the
    PLANNER_OPENAI_KEY environment variable, never hardcoded.
    """

    name = "openai"

    def __init__(self, model="gpt-4o-mini", api_key=None,
                 base_url="https://api.openai.com/v1", timeout=60):
        self.model = model
        self.api_key = api_key or os.environ.get("PLANNER_OPENAI_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self):
        return bool(self.api_key)

    def generate(self, prompt, system=None, temperature=0.0,
                 max_tokens=1024, as_json=False):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if as_json:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


class _CallableBackend(LLMBackend):
    name = "callable"

    def __init__(self, fn):
        self._fn = fn

    def generate(self, prompt, system=None, temperature=0.0,
                 max_tokens=1024, as_json=False):
        return self._fn(prompt, system)


def as_backend(obj):
    """Coerce a backend, a bare callable, or None into an LLMBackend.

    None yields the default backend. A plain function fn(prompt, system)
    is wrapped. Anything already exposing generate is returned as is.
    """
    if obj is None:
        return default_backend()
    if isinstance(obj, LLMBackend):
        return obj
    if hasattr(obj, "generate"):
        return obj
    if callable(obj):
        return _CallableBackend(obj)
    raise TypeError("backend must be an LLMBackend, a callable, or None")


def default_backend(prefer_local=True):
    """Ollama on gemma3 if it is reachable, otherwise the echo backend.

    The fallback keeps every caller working with no model installed; the
    text tasks then return their offline placeholder rather than raising.
    """
    if prefer_local:
        ol = OllamaBackend()
        if ol.available():
            return ol
    return EchoBackend()


def generate_json(backend, prompt, system=None, temperature=0.0,
                  max_tokens=1024, default=None):
    """Call a backend expecting JSON and parse it defensively.

    Extracts the outermost brace-delimited object from the reply, so a
    model that wraps JSON in prose still parses. On any failure returns
    default (an empty dict if not given) rather than raising, since these
    calls sit behind human review and should degrade, not crash.
    """
    backend = as_backend(backend)
    try:
        reply = backend.generate(prompt, system=system,
                                 temperature=temperature,
                                 max_tokens=max_tokens, as_json=True)
        start = reply.index("{")
        end = reply.rindex("}") + 1
        return json.loads(reply[start:end])
    except Exception:
        return {} if default is None else default


if __name__ == "__main__":
    echo = EchoBackend()
    assert echo.generate("anything") == \
        "[offline: no language model configured]"
    assert echo.generate("x", as_json=True) == "{}"
    assert EchoBackend(reply="hi").generate("x") == "hi"

    # callable coercion
    b = as_backend(lambda prompt, system: "seen: " + prompt)
    assert b.generate("p") == "seen: p"
    assert as_backend(echo) is echo

    # a backend already exposing generate passes through untouched
    class Custom:
        def generate(self, prompt, system=None, **kw):
            return "custom"
    assert as_backend(Custom()).generate("x") == "custom"

    try:
        as_backend(42)
        raise AssertionError("coerced a non-backend")
    except TypeError:
        pass

    # generate_json parses JSON wrapped in prose, and falls back cleanly
    wrapped = EchoBackend(reply='here you go: {"a": 1, "b": [2,3]} thanks')
    assert generate_json(wrapped, "p") == {"a": 1, "b": [2, 3]}
    broken = EchoBackend(reply="no json here")
    assert generate_json(broken, "p", default={"ok": False}) == {"ok": False}

    # unreachable Ollama reports unavailable and default falls back to echo
    assert OllamaBackend(url="http://127.0.0.1:1").available() is False
    assert OpenAIBackend(api_key="").available() is False
    assert OpenAIBackend(api_key="sk-x").available() is True

    # the documented default is gemma3 on Ollama
    assert OllamaBackend().model == "gemma3"
    d = default_backend()
    assert d.name in ("ollama", "echo")
    print("llm_adapter self-tests passed")
