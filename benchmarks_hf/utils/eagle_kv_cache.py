"""
EAGLE-style KV Cache with pre-allocation and efficient copy operations.

完全对齐EAGLE官方实现 (EAGLE-main/eagle/model/kv_cache.py)
核心特性：
1. Pre-allocated固定大小cache - 避免动态内存分配
2. index_select + copy_ - 高效KV重组
3. CPU length tracking - 快速访问无GPU同步开销
"""

import torch
from typing import List, Tuple, Optional


class KVCache:
    """
    单个KV cache (key或value)。

    使用pre-allocated tensor和current_length tracking实现高效操作。
    """

    def __init__(self, data: torch.Tensor, current_length: torch.Tensor):
        """
        Args:
            data: Pre-allocated [batch, heads, max_len, head_dim]
            current_length: Scalar tensor on CPU tracking actual length
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """返回实际shape (用current_length替代max_len)"""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        核心操作：复制指定indices的KV states到新位置。

        用于speculative decoding的关键优化：
        验证后，将accepted tokens的KV states复制到连续位置，
        避免re-forward这些tokens。

        Args:
            indices: 要复制的位置索引 [num_accepted]
            prev_length: 目标起始位置
            dim: 操作维度 (默认2=sequence dim)
        """
        # 用index_select提取指定位置的KV
        tgt = self.data.index_select(dim, indices)

        # 获取目标slice
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])

        # In-place copy (高效！)
        dst.copy_(tgt, non_blocking=True)

        # 更新长度
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        添加新的KV states到cache末尾。

        Args:
            tensor: 新的KV states
            dim: 拼接维度

        Returns:
            完整cache的view (up to current_length)
        """
        curr_len = self.current_length.item()

        # 获取目标slice
        dst = self.data.narrow(dim, curr_len, tensor.shape[dim])

        # Copy新数据
        dst.copy_(tensor)

        # 更新长度
        self.current_length.add_(tensor.shape[dim])

        # 返回view
        return self.data.narrow(dim, 0, self.current_length.item())


def initialize_eagle_kv_cache(
    model,
    max_length: int = 4096,
    batch_size: int = 1
) -> Tuple[List[Tuple[KVCache, KVCache]], List[torch.Tensor], torch.Tensor]:
    """
    初始化EAGLE-style pre-allocated KV cache。

    完全对齐EAGLE官方实现 (kv_cache.py:69-157)

    Args:
        model: Transformer model
        max_length: 最大序列长度
        batch_size: Batch size

    Returns:
        past_key_values: List of (key_cache, value_cache) for each layer
        past_key_values_data_list: Raw data tensors (for multi-GPU)
        current_length_data: Length tracking tensor on CPU
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # 获取每层的device (处理model parallelism)
    devices = []
    for i in range(num_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except AttributeError:
            try:
                device = model.layers[i].self_attn.q_proj.weight.device
            except AttributeError:
                device = next(model.parameters()).device
        devices.append(device)

    # 按device分组layers
    past_key_values_data_list = []
    start_device = devices[0]
    start_num = 0

    for idx, device in enumerate(devices):
        if device != start_device:
            # 为前一组分配tensor
            data = torch.zeros(
                start_num * 2,  # *2 for keys and values
                batch_size,
                num_heads,
                max_length,
                head_dim,
                device=start_device,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(data)

            # 开始新组
            start_device = device
            start_num = 0
        start_num += 1

    # 处理最后一组
    data = torch.zeros(
        start_num * 2,
        batch_size,
        num_heads,
        max_length,
        head_dim,
        device=start_device,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(data)

    # 在CPU上初始化length tracking (关键优化！避免GPU同步)
    current_length_data = torch.zeros(
        num_layers * 2,  # *2 for keys and values
        dtype=torch.long,
        device="cpu"
    )

    # 创建KVCache objects
    past_key_values = []
    bias = 0
    start_data_idx = 0

    for i in range(num_layers):
        data_idx = 0
        # 找到当前layer对应的data tensor
        for idx, device in enumerate(set(devices)):
            if devices[i] == device:
                data_idx = idx
                break

        if data_idx != start_data_idx:
            bias = 0
            start_data_idx = data_idx

        # Key cache
        key_cache = KVCache(
            past_key_values_data_list[data_idx][bias * 2],
            current_length_data[i * 2]
        )

        # Value cache
        value_cache = KVCache(
            past_key_values_data_list[data_idx][bias * 2 + 1],
            current_length_data[i * 2 + 1]
        )

        past_key_values.append((key_cache, value_cache))
        bias += 1

    return past_key_values, past_key_values_data_list, current_length_data


def update_kv_with_accepted_tokens(
    past_key_values_data_list: List[torch.Tensor],
    current_length_data: torch.Tensor,
    select_indices: torch.Tensor,
    prev_input_len: int
):
    """
    用accepted tokens的KV states更新cache。

    这是1-forward-per-iteration的核心操作：
    通过copy而不是re-forward来更新cache。

    完全对齐EAGLE官方 (utils.py:444-452)

    Args:
        past_key_values_data_list: Raw KV data tensors
        current_length_data: Length tracking
        select_indices: Accepted tokens的绝对位置
        prev_input_len: 更新前的长度
    """
    for past_key_values_data in past_key_values_data_list:
        # 提取accepted positions的KV states
        # [layers*2, batch, heads, max_len, head_dim]
        # -> [layers*2, batch, heads, num_accepted, head_dim]
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]

        # 目标位置：从prev_input_len开始
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]

        # In-place copy
        dst.copy_(tgt, non_blocking=True)

    # 更新length
    new_length = prev_input_len + tgt.shape[-2]
    current_length_data.fill_(new_length)


def eagle_kv_to_hf_format(
    past_key_values: List[Tuple[KVCache, KVCache]],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    将EAGLE KVCache转换为HuggingFace tuple format。

    优化：返回narrow view而不是contiguous copy，减少内存操作。

    Args:
        past_key_values: List of (key_cache, value_cache)

    Returns:
        Tuple of (key, value) for each layer (as views, not copies!)
    """
    hf_cache = []
    for key_cache, value_cache in past_key_values:
        # 提取actual data (up to current_length) - 返回view而不是copy！
        curr_len = key_cache.current_length.item()
        key = key_cache.data.narrow(2, 0, curr_len)  # view, no copy
        value = value_cache.data.narrow(2, 0, curr_len)  # view, no copy
        hf_cache.append((key, value))

    return tuple(hf_cache)


def reset_eagle_kv_cache(
    past_key_values: List[Tuple[KVCache, KVCache]],
    current_length_data: torch.Tensor
):
    """
    重置cache lengths为0 (用于新的generation)。

    Args:
        past_key_values: List of (key_cache, value_cache)
        current_length_data: Length tracking tensor
    """
    current_length_data.fill_(0)
