"""rANS entropy coder that advances many independent streams in lockstep (PyTorch, CPU or GPU).

Every stream keeps a 32-bit state and emits 16-bit words. The words of all streams are interleaved
in the order in which the decoder asks for them, so no per-stream lengths are stored.
"""

from dataclasses import dataclass

import torch

PRECISION = 14
_TOTAL = 1 << PRECISION
_LOWER = 1 << 16


def normalize_counts(counts: torch.Tensor) -> torch.Tensor:
    """Turn symbol counts [tables, alphabet] into frequencies >= 1 that sum to 2**PRECISION per table."""
    counts = counts.double()
    alphabet = counts.shape[1]
    total = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    freq = torch.floor(counts * (_TOTAL - alphabet) / total).long() + 1
    rows = torch.arange(freq.shape[0], device=freq.device)
    freq[rows, freq.argmax(dim=1)] += _TOTAL - freq.sum(dim=1)
    return freq.to(torch.int32)


@dataclass
class Tables:
    """Frequency tables with individual alphabet sizes, stored back to back.

    freq: int32 [sum of alphabet sizes]; start: int64 [tables], offset of every table in `freq`.
    """

    freq: torch.Tensor
    start: torch.Tensor

    @classmethod
    def from_frequencies(cls, groups: list[torch.Tensor]) -> "Tables":
        """Concatenate groups of tables, each [tables, alphabet]; tables are numbered in the given order."""
        device = groups[0].device
        sizes = torch.cat([torch.full((g.shape[0],), g.shape[1], dtype=torch.int64, device=device) for g in groups])
        freq = torch.cat([g.reshape(-1).to(torch.int32) for g in groups])
        return cls(freq, torch.cumsum(sizes, dim=0) - sizes)

    def table_of_entry(self) -> torch.Tensor:
        entries = torch.arange(len(self.freq), device=self.freq.device)
        return torch.searchsorted(self.start, entries, right=True) - 1

    def lower(self) -> torch.Tensor:
        """Cumulative frequency below every symbol within its table."""
        running = torch.cumsum(self.freq.long(), dim=0) - self.freq
        return running - running[self.start][self.table_of_entry()]


@dataclass
class Streams:
    """Shape of a batch of streams: step j of stream i uses table `first_table[i] + (j % period[i]) * stride[i]`.

    lengths: number of symbols of every stream; all fields are int64 [streams].
    """

    lengths: torch.Tensor
    first_table: torch.Tensor
    period: torch.Tensor
    stride: torch.Tensor

    def fields(self) -> tuple[torch.Tensor, ...]:
        return self.lengths, self.first_table, self.period, self.stride

    @classmethod
    def cat(cls, parts: list["Streams"]) -> "Streams":
        return cls(*(torch.cat(fields) for fields in zip(*(part.fields() for part in parts))))

    def to(self, device: torch.device | str) -> "Streams":
        return Streams(*(field.to(device) for field in self.fields()))

    def tables(self, step: int) -> torch.Tensor:
        return self.first_table + (step % self.period) * self.stride


def _wrap(values: torch.Tensor, bits: int, dtype: torch.dtype) -> torch.Tensor:
    """Store unsigned `bits`-bit integers in the signed type of the same width."""
    half = 1 << (bits - 1)
    return (((values + half) & ((1 << bits) - 1)) - half).to(dtype)


def encode(symbols: torch.Tensor, streams: Streams, tables: Tables) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the leading `streams.lengths` symbols of every row of `symbols` [streams, steps].

    Returns the emitted words (int16, in decoding order) and the final states (int32) of the streams.
    """
    freq, lower = tables.freq.long(), tables.lower()
    state = torch.full((symbols.shape[0],), _LOWER, dtype=torch.int64, device=symbols.device)
    emitted = []
    for step in range(symbols.shape[1] - 1, -1, -1):
        entry = tables.start[streams.tables(step)] + symbols[:, step].long()
        f, c = freq[entry], lower[entry]
        active = streams.lengths > step
        overflow = active & (state >= (f << (32 - PRECISION)))
        emitted.append(_wrap(state[overflow], 16, torch.int16))
        state = torch.where(overflow, state >> 16, state)
        coded = (torch.div(state, f, rounding_mode="floor") << PRECISION) + torch.remainder(state, f) + c
        state = torch.where(active, coded, state)
    words = torch.cat(emitted[::-1]) if emitted else torch.empty(0, dtype=torch.int16, device=symbols.device)
    return words, _wrap(state, 32, torch.int32)


def decode(words: torch.Tensor, states: torch.Tensor, streams: Streams, tables: Tables) -> torch.Tensor:
    """Inverse of `encode`: uint8 symbols [streams, longest stream]; entries past the end of a stream are 0."""
    freq, lower = tables.freq.long(), tables.lower()
    keys = tables.table_of_entry() * _TOTAL + lower + freq
    state = states.long() & 0xFFFFFFFF
    words = words.long() & 0xFFFF
    steps = int(streams.lengths.max()) if streams.lengths.numel() else 0
    symbols = torch.zeros(states.shape[0], steps, dtype=torch.uint8, device=words.device)
    position = 0
    for step in range(steps):
        table = streams.tables(step)
        active = streams.lengths > step
        slot = state & (_TOTAL - 1)
        entry = torch.searchsorted(keys, table * _TOTAL + slot, right=True)
        state = torch.where(active, freq[entry] * (state >> PRECISION) + slot - lower[entry], state)
        refill = active & (state < _LOWER)
        count = int(refill.sum())
        state[refill] = (state[refill] << 16) | words[position : position + count]
        position += count
        symbols[:, step] = torch.where(active, entry - tables.start[table], 0).to(torch.uint8)
    return symbols
