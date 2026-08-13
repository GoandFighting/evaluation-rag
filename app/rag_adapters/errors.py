class RAGAdapterError(RuntimeError):
    """Base class for target invocation failures."""


class AdapterConfigurationError(RAGAdapterError):
    """The adapter cannot load or validate its local configuration."""


class AdapterExecutionError(RAGAdapterError):
    """The target process or service could not complete the request."""


class AdapterResponseError(RAGAdapterError):
    """The target returned a response that cannot be normalized."""
