import torch

def pack_indices(indices: torch.Tensor, b: int) -> torch.Tensor:
    if b not in [1, 2, 4]:
        raise ValueError("Bit-width must be 1, 2, or 4 for standard byte packing.")

    elements_per_byte = 8 // b
    if indices.shape[-1] % elements_per_byte != 0:
        raise ValueError(f"Last dimension must be divisible by {elements_per_byte} for {b}-bit packing.")

    new_shape = list(indices.shape)
    new_shape[-1] = new_shape[-1] // elements_per_byte
    new_shape.append(elements_per_byte)
    
    grouped = indices.view(*new_shape).to(torch.uint8)
    packed = torch.zeros(new_shape[:-1], dtype=torch.uint8, device=indices.device)
    
    for i in range(elements_per_byte):
        shift = 8 - b * (i + 1)
        packed |= (grouped[..., i] << shift)
        
    return packed

def unpack_indices(packed: torch.Tensor, b: int, original_dim: int) -> torch.Tensor:
    if b not in [1, 2, 4]:
        raise ValueError("Bit-width must be 1, 2, or 4.")
        
    elements_per_byte = 8 // b
    mask = (1 << b) - 1
    
    unpacked_shape = list(packed.shape)
    unpacked_shape.append(elements_per_byte)
    
    unpacked = torch.zeros(unpacked_shape, dtype=torch.int64, device=packed.device)
    
    for i in range(elements_per_byte):
        shift = 8 - b * (i + 1)
        unpacked[..., i] = (packed >> shift) & mask
        
    final_shape = list(packed.shape[:-1]) + [original_dim]
    return unpacked.view(*final_shape)
