import numpy as np

def allocate_aligned_array(size, dtype=np.float64, alignment=64):
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = size * bytes_per_element
    
    raw_memory = np.empty(total_bytes + alignment, dtype=np.uint8)
    
    address = raw_memory.__array_interface__['data'][0]
    offset = (alignment - (address % alignment)) % alignment
    
    aligned_memory = raw_memory[offset:offset + total_bytes]
    return aligned_memory.view(dtype)
