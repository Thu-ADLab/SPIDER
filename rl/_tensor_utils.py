import torch


def pad_at_end(x, dim=0, padding_channels=1, padding_value=0.0):
    """
    Pad a tensor at the end of a dimension.
    qzl: 没有用torch.nn.functional.pad，参数有点复杂

    :param x: The tensor to pad.
    :param dim: The dimension to pad.
    :param padding_channels: The number of channels to pad.
    :param padding_value: The value to pad with.
    """
    if dim >= len(x.size()):
        raise ValueError("Dimension specified for padding is out of range for the input tensor.")

    # 构造 pad 的 shape
    pad_shape = list(x.shape)
    pad_shape[dim] = padding_channels

    # 构造 padding tensor，并填充为指定的值
    padding_tensor = torch.full(pad_shape, padding_value, dtype=x.dtype, device=x.device)

    # 使用 torch.cat 拼接原始张量和 padding 张量
    padded_tensor = torch.cat([x, padding_tensor], dim=dim)

    return padded_tensor


def normalize_to_range(x, min_val, max_val):
    """
    Min-Max normalize a tensor to [0, 1] given a specific range.

    :param x: The input tensor to normalize.
    :param min_val: The minimum value of the range.
    :param max_val: The maximum value of the range.
    """

    clamped_x = torch.clamp(x, min=min_val, max=max_val) # Clip to the specified range
    normalized_x = (clamped_x - min_val) / (max_val - min_val)
    return normalized_x

def convert_to_relative():
    pass

