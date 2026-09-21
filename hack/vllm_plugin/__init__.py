"""vLLM integration of HACK: attention backend, KV connector and disaggregation proxy."""

BACKEND_CLASS = "hack.vllm_plugin.backend.HackAttentionBackend"
CONNECTOR_MODULE = "hack.vllm_plugin.connector"
CONNECTOR_CLASS = "HackKVConnector"


def register() -> None:
    """Entry point of the `vllm.general_plugins` group: makes the backend selectable as CUSTOM."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS)
