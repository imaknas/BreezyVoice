"""Pre-allocated attention key/value storage for autoregressive decoding."""
import torch


class LayerKV:
    """Keys and values of one attention layer, written in place as decoding advances.

    Growing a cache by concatenation copies every earlier position on every step, so the cost of a
    step rises with the length of the sequence. Here the buffers are allocated once for the longest
    sequence the caller can produce, and a step only writes its own position.
    """

    def __init__(self, heads: int, d_k: int, capacity: int, device: torch.device, dtype: torch.dtype):
        self._k = torch.empty((1, heads, capacity, d_k), device=device, dtype=dtype)
        self._v = torch.empty((1, heads, capacity, d_k), device=device, dtype=dtype)
        self.length = 0

    @property
    def capacity(self) -> int:
        return self._k.size(2)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: (1, heads, time, d_k)."""
        end = self.length + k.size(2)
        if end > self.capacity:
            raise ValueError(f"KV cache overflow: {end} positions, capacity {self.capacity}")
        self._k[:, :, self.length:end] = k
        self._v[:, :, self.length:end] = v
        self.length = end

    @property
    def keys(self) -> torch.Tensor:
        return self._k[:, :, :self.length]

    @property
    def values(self) -> torch.Tensor:
        return self._v[:, :, :self.length]
