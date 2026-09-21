"""Settings of the vLLM integration.

Every option can be given as an environment variable or under the key "hack" of vLLM's
`--additional-config`; the latter takes precedence.

    option                   environment variable            default
    partition_size           HACK_PARTITION_SIZE             64
    kv_bits                  HACK_KV_BITS                    2
    stochastic               HACK_STOCHASTIC                 0
    summation_elimination    HACK_SUMMATION_ELIMINATION      1
    requant_elimination      HACK_REQUANT_ELIMINATION        1
    sink_tokens              HACK_SINK_TOKENS                4
    quantized_prefill        HACK_QUANTIZED_PREFILL          0
    side_slots               HACK_SIDE_SLOTS                 max_num_seqs + max(8, max_num_seqs / 4)
    attention                HACK_ATTENTION                  kernels     (kernels | reference)
    decode                   HACK_DECODE                     batched     (batched | loop)
    debug_checks             HACK_DEBUG_CHECKS               0
"""

import os
from dataclasses import dataclass, fields
from typing import Any

from hack.config import HackConfig

ADDITIONAL_CONFIG_KEY = "hack"
_BOOL_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PluginSettings:
    config: HackConfig
    side_slots: int | None = None
    attention: str = "kernels"
    decode: str = "batched"
    debug_checks: bool = False

    def __post_init__(self):
        if self.attention not in ("reference", "kernels"):
            raise ValueError("attention must be 'reference' or 'kernels'")
        if self.decode not in ("batched", "loop"):
            raise ValueError("decode must be 'batched' or 'loop'")

    def num_side_slots(self, max_num_seqs: int) -> int:
        if self.side_slots is not None:
            return self.side_slots
        return max_num_seqs + max(8, max_num_seqs // 4)


def _convert(value: Any, kind: type) -> Any:
    if kind is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in _BOOL_TRUE
    return kind(value)


def _read(options: dict[str, Any], name: str, kind: type, default: Any) -> Any:
    if name in options:
        return _convert(options[name], kind)
    raw = os.environ.get("HACK_" + name.upper())
    return default if raw is None or raw == "" else _convert(raw, kind)


def settings_from(options: dict[str, Any] | None = None) -> PluginSettings:
    options = dict(options or {})
    defaults = HackConfig()
    config = HackConfig(
        **{
            f.name: _read(options, f.name, type(getattr(defaults, f.name)), getattr(defaults, f.name))
            for f in fields(HackConfig)
        }
    )
    side_slots = _read(options, "side_slots", int, None)
    return PluginSettings(
        config=config,
        side_slots=side_slots,
        attention=_read(options, "attention", str, "kernels"),
        decode=_read(options, "decode", str, "batched"),
        debug_checks=_read(options, "debug_checks", bool, False),
    )


_cached: PluginSettings | None = None


def get_settings() -> PluginSettings:
    """Settings of this process, resolved once from the active vLLM configuration."""
    global _cached
    if _cached is not None:
        return _cached
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return settings_from(None)
    additional = getattr(vllm_config, "additional_config", None)
    options = additional.get(ADDITIONAL_CONFIG_KEY) if isinstance(additional, dict) else None
    _cached = settings_from(options)
    return _cached


def reset_settings() -> None:
    global _cached
    _cached = None
