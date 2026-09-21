from dataclasses import dataclass


@dataclass(frozen=True)
class HackConfig:
    """Quantization settings of HACK.

    partition_size: number of elements that share one (min, scale) pair (Pi in the paper).
    kv_bits: bit width of the stored K and V codes.
    qp_bits: bit width of the transient Q and attention-probability codes.
    stochastic: use stochastic rounding (otherwise round to nearest).
    summation_elimination: cache the per-partition code sums of K and V.
    requant_elimination: keep the last, partially filled block in 16-bit until it is full.
    sink_tokens: number of leading tokens whose K and V stay in 16-bit.
    quantized_prefill: compute the attention of the prefill stage on the codes as well
        (otherwise the prefill stage attends to the 16-bit K and V it has just computed).
    """

    partition_size: int = 64
    kv_bits: int = 2
    qp_bits: int = 8
    stochastic: bool = False
    summation_elimination: bool = True
    requant_elimination: bool = True
    sink_tokens: int = 4
    quantized_prefill: bool = False

    def __post_init__(self):
        if self.partition_size % 16 != 0:
            raise ValueError("partition_size must be a multiple of 16")
        if self.kv_bits not in (2, 4, 8):
            raise ValueError("kv_bits must be 2, 4 or 8")
        if self.qp_bits != 8:
            raise ValueError("qp_bits must be 8")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be >= 0")

    @property
    def kv_levels(self) -> int:
        return (1 << self.kv_bits) - 1

    @property
    def qp_levels(self) -> int:
        return (1 << self.qp_bits) - 1
