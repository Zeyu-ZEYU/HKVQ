"""Per-step plan: which rows decode, which prefill, and which side-pool slots they use.

The plan is built once per scheduler step from host-side sequence lengths and is shared by
all layers. Block ids stay on the GPU; everything that depends on them is computed there.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from hack.config import HackConfig
from hack.vllm_plugin.slots import SlotPool


@dataclass
class PrefillRow:
    start: int
    stop: int
    context: int
    total: int
    blocks: torch.Tensor
    open_slot: torch.Tensor
    old_open_slot: torch.Tensor
    sink_slot: torch.Tensor


@dataclass
class DecodeBatch:
    size: int
    max_total: int
    max_closed: int
    token_index: torch.Tensor
    totals: torch.Tensor
    totals_cpu: np.ndarray
    block_table: torch.Tensor
    open_slot: torch.Tensor
    sink_slot: torch.Tensor
    quant_rows: torch.Tensor
    open_offset: torch.Tensor
    sink_rows: torch.Tensor
    sink_pos: torch.Tensor
    seal_rows: torch.Tensor
    seal_blocks: torch.Tensor
    requant_groups: list[tuple[int, torch.Tensor]] = field(default_factory=list)


@dataclass
class StepPlan:
    num_tokens: int
    decode: DecodeBatch | None
    prefills: list[PrefillRow]


def _upload(values: np.ndarray, device: torch.device) -> torch.Tensor:
    host = torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


class StepPlanner:
    def __init__(self, config: HackConfig, open_slots: SlotPool, sink_slots: SlotPool | None, device: torch.device):
        self.config = config
        self.open_slots = open_slots
        self.sink_slots = sink_slots
        self.device = device

    def plan(
        self,
        query_start_loc: np.ndarray,
        seq_lens: np.ndarray,
        block_table: torch.Tensor,
        num_tokens: int,
    ) -> StepPlan:
        block, sinks = self.config.partition_size, self.config.sink_tokens
        starts = query_start_loc[:-1].astype(np.int64)
        lengths = np.diff(query_start_loc).astype(np.int64)
        totals = seq_lens.astype(np.int64)[: len(lengths)]
        rows = np.nonzero(lengths > 0)[0]
        starts, lengths, totals = starts[rows], lengths[rows], totals[rows]
        contexts = totals - lengths

        quant_before = np.maximum(contexts - sinks, 0)
        quant_after = np.maximum(totals - sinks, 0)
        had_open = quant_before % block != 0
        has_open = quant_after % block != 0
        old_part, new_part = quant_before // block, quant_after // block
        reuse = had_open & has_open & (old_part == new_part)
        allocate = has_open & ~reuse

        table = block_table[_upload(rows, self.device)] if len(rows) != block_table.shape[0] else block_table
        self.open_slots.begin_step()
        scratch = self.open_slots.scratch_slot
        old_slot = torch.full((len(rows),), scratch, dtype=torch.int64, device=self.device)
        new_slot = torch.full((len(rows),), scratch, dtype=torch.int64, device=self.device)
        touched = np.nonzero(had_open)[0]
        if len(touched):
            index = _upload(touched, self.device)
            old_slot[index] = self.open_slots.touch(table[index, _upload(old_part[touched], self.device)].long())
        fresh = np.nonzero(allocate)[0]
        if len(fresh):
            index = _upload(fresh, self.device)
            new_slot[index] = self.open_slots.allocate(table[index, _upload(new_part[fresh], self.device)].long())
        kept = np.nonzero(reuse)[0]
        if len(kept):
            index = _upload(kept, self.device)
            new_slot[index] = old_slot[index]

        sink_slot = self._sink_slots(table, contexts)

        is_decode = lengths == 1
        decode = self._decode_batch(
            np.nonzero(is_decode)[0], starts, totals, table, old_slot, new_slot, sink_slot
        )
        prefills = [
            PrefillRow(
                start=int(starts[i]),
                stop=int(starts[i] + lengths[i]),
                context=int(contexts[i]),
                total=int(totals[i]),
                blocks=table[i].long(),
                open_slot=new_slot[i : i + 1],
                old_open_slot=old_slot[i : i + 1],
                sink_slot=sink_slot[i : i + 1],
            )
            for i in np.nonzero(~is_decode)[0]
        ]
        return StepPlan(num_tokens=num_tokens, decode=decode, prefills=prefills)

    def _sink_slots(self, table: torch.Tensor, contexts: np.ndarray) -> torch.Tensor:
        if self.sink_slots is None:
            return torch.zeros(len(contexts), dtype=torch.int64, device=self.device)
        self.sink_slots.begin_step()
        result = torch.full((len(contexts),), self.sink_slots.scratch_slot, dtype=torch.int64, device=self.device)
        first_blocks = table[:, 0].long()
        running = np.nonzero(contexts > 0)[0]
        if len(running):
            index = _upload(running, self.device)
            result[index] = self.sink_slots.touch(first_blocks[index])
        started = np.nonzero(contexts == 0)[0]
        if len(started):
            index = _upload(started, self.device)
            result[index] = self.sink_slots.allocate(first_blocks[index])
        return result

    def _decode_batch(
        self,
        rows: np.ndarray,
        starts: np.ndarray,
        totals: np.ndarray,
        table: torch.Tensor,
        old_slot: torch.Tensor,
        new_slot: torch.Tensor,
        sink_slot: torch.Tensor,
    ) -> DecodeBatch | None:
        if len(rows) == 0:
            return None
        block, sinks = self.config.partition_size, self.config.sink_tokens
        totals = totals[rows]
        positions = totals - 1
        max_total = int(totals.max())
        width = (max_total + block - 1) // block

        index = _upload(rows, self.device)
        sub_table = table[index, :width].long()
        quantized = positions >= sinks
        open_offset = (positions - sinks) % block
        write_old = quantized & (open_offset > 0)
        write_slot = torch.where(_upload(write_old, self.device).bool(), old_slot[index], new_slot[index])

        quant_rows = np.nonzero(quantized)[0]
        sink_rows = np.nonzero(~quantized)[0]
        seal_rows = np.nonzero(quantized & (open_offset == block - 1))[0]
        seal_rows_gpu = _upload(seal_rows, self.device)
        seal_parts = _upload((positions[seal_rows] - sinks) // block, self.device)
        seal_blocks = sub_table[seal_rows_gpu, seal_parts] if len(seal_rows) else seal_rows_gpu

        groups: list[tuple[int, torch.Tensor]] = []
        if not self.config.requant_elimination:
            open_rows = quantized & (open_offset != block - 1)
            for offset in np.unique(open_offset[open_rows]):
                members = np.nonzero(open_rows & (open_offset == offset))[0]
                groups.append((int(offset) + 1, _upload(members, self.device)))

        return DecodeBatch(
            size=len(rows),
            max_total=max_total,
            max_closed=int((np.maximum(totals - sinks, 0) // block).max()),
            token_index=_upload(starts[rows], self.device),
            totals=_upload(totals, self.device),
            totals_cpu=totals,
            block_table=sub_table,
            open_slot=write_slot,
            sink_slot=sink_slot[index],
            quant_rows=_upload(quant_rows, self.device),
            open_offset=_upload(open_offset[quant_rows], self.device),
            sink_rows=_upload(sink_rows, self.device),
            sink_pos=_upload(positions[sink_rows], self.device),
            seal_rows=seal_rows_gpu,
            seal_blocks=seal_blocks,
            requant_groups=groups,
        )
