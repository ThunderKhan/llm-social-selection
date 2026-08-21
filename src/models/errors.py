class ModelProviderError(Exception):
    """A model provider could not complete a request."""


class OllamaConnectionError(ModelProviderError):
    """The local Ollama service could not be reached."""


class OllamaTimeoutError(ModelProviderError):
    """The local Ollama request exceeded its configured timeout."""


class OllamaResponseError(ModelProviderError):
    """Ollama returned an invalid or unusable response."""


class OllamaModelError(ModelProviderError):
    """The requested Ollama model is not locally available."""
