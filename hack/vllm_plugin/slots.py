"""Slot allocation for the 16-bit side pools (open blocks, sink tokens).

A slot is keyed by a vLLM block id. All bookkeeping lives on the GPU and uses tensor
operations of fixed shape, so assigning slots never synchronizes with the host.
A slot that is used by a scheduled sequence is touched in every step; new slots are
taken from the least recently used ones, which recycles the slots of sequences that
ended. Slots of sequences that are not scheduled yet can be pinned.
"""

import torch

_NEVER = torch.iinfo(torch.int64).max


class SlotPool:
    def __init__(self, num_slots: int, device: torch.device, debug_checks: bool = False):
        self.num_slots = num_slots
        self.device = device
        self.debug_checks = debug_checks
        self.clock = 0
        self.num_blocks = 0
        self.block_slot: torch.Tensor | None = None
        self.slot_block = torch.full((num_slots + 1,), -1, dtype=torch.int64, device=device)
        self.last_use = torch.zeros(num_slots + 1, dtype=torch.int64, device=device)
        self.pinned = torch.zeros(num_slots + 1, dtype=torch.bool, device=device)
        self._pinned_by: dict[str, list[torch.Tensor]] = {}
        self.num_pinned = 0

    @property
    def scratch_slot(self) -> int:
        """Index of the extra slot that absorbs reads and writes of sequences without a slot."""
        return self.num_slots

    def bind(self, num_blocks: int) -> None:
        if self.block_slot is None or self.num_blocks != num_blocks:
            self.num_blocks = num_blocks
            self.block_slot = torch.full((num_blocks + 1,), -1, dtype=torch.int64, device=self.device)
            self.slot_block.fill_(-1)
            self.last_use.zero_()

    def begin_step(self) -> None:
        self.clock += 1

    def lookup(self, blocks: torch.Tensor) -> torch.Tensor:
        """Slots of `blocks`; the scratch slot for blocks without one."""
        slots = self.block_slot[blocks]
        if self.debug_checks and blocks.numel():
            assert bool((slots >= 0).all()), "a scheduled sequence lost its side-pool slot"
        return torch.where(slots >= 0, slots, self.scratch_slot)

    def touch(self, blocks: torch.Tensor) -> torch.Tensor:
        slots = self.lookup(blocks)
        self.last_use[slots] = self.clock
        return slots

    def allocate(self, blocks: torch.Tensor) -> torch.Tensor:
        """Assign a slot to each of the distinct `blocks` and return the slots."""
        count = blocks.numel()
        if count == 0:
            return blocks.new_empty(0)
        if count > self.num_slots:
            raise RuntimeError(f"{count} side-pool slots requested, the pool has {self.num_slots}")
        previous = self.block_slot[blocks]
        previous = torch.where(previous >= 0, previous, self.scratch_slot)
        self.slot_block[previous] = -1
        self.last_use[previous] = 0

        priority = torch.where(self.pinned[: self.num_slots], _NEVER, self.last_use[: self.num_slots])
        picked = torch.topk(priority, count, largest=False).indices
        if self.debug_checks:
            assert bool((priority[picked] < self.clock).all()), "side pool exhausted: raise HACK_SIDE_SLOTS"
        evicted = self.slot_block[picked]
        self.block_slot[torch.where(evicted >= 0, evicted, self.num_blocks)] = -1
        self.slot_block[picked] = blocks
        self.block_slot[blocks] = picked
        self.last_use[picked] = self.clock
        return picked

    def pin(self, owner: str, slots: torch.Tensor) -> None:
        self.pinned[slots] = True
        self._pinned_by.setdefault(owner, []).append(slots)
        self.num_pinned += slots.numel()

    def unpin(self, owner: str) -> None:
        for slots in self._pinned_by.pop(owner, []):
            self.pinned[slots] = False
            self.last_use[slots] = self.clock
            self.num_pinned -= slots.numel()
