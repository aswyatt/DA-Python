#%%
import numpy as np
def random_bytes(num_bytes: int) -> bytes:
    return np.random.randint(0, 256, size=num_bytes, dtype=np.uint8).tobytes()

def bytes_to_bits(byte_stream: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in byte_stream)

def get_n_msb(value: int, num_bits: int) -> int:
    """
    Extract the top num_bits MSBs from an integer.
    Returns an integer with n bits.
    """
    total_bits = value.bit_length()
    shift = total_bits - num_bits
    return (value >> shift) & ((1 << num_bits) - 1)

def chunk_bytes(byte_stream: bytes, num_bytes: int=1) -> np.ndarray:
    """
    Interpret data as an array of N-bit unsigned integers.
    Pads with zeros if data length is not a multiple of N/8.
    """

    # pad data to multiple of byte_width
    pad_len = (-len(byte_stream)) % num_bytes
    if pad_len:
        byte_stream = byte_stream + b"\x00" * pad_len

    dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[num_bytes]
    return np.frombuffer(byte_stream, dtype=dtype)

# --- Collapse functions ---

def collapse_xor(chunks: np.ndarray) -> int:
    return int(np.bitwise_xor.reduce(chunks))

def chunk_sum(chunks:np.ndarray):
    return int(chunks.sum(dtype=np.uint64))

def collapse_modular(chunks: np.ndarray, num_bytes: int) -> int:
    N = 8*num_bytes
    mask = (1 << N) - 1
    return chunk_sum(chunks) & mask

def collapse_circular(chunks: np.ndarray, num_bytes: int) -> int:
    """
    End-around carry addition: fold carry-out back into LSB.
    """
    N = num_bytes*8
    mask = (1 << N) - 1
    total= chunk_sum(chunks)
    # fold carries until none remain
    while total >> N:
        total = (total & mask) + (total >> N)
    return total

def collapse_leading(chunks:np.ndarray, num_bytes:int):
    return get_n_msb(chunk_sum(chunks), 8*num_bytes)




#%% --- Example usage ---
if __name__ == "__main__":
    data = b"Hello world"
    num_chunk_bytes = 1
    num_hash_bytes = 1

    chunks = chunk_bytes(data, num_chunk_bytes)
    print("Chunks:", chunks)

    print("XOR collapse:", collapse_xor(chunks))
    print("Modular collapse:", collapse_modular(chunks, num_hash_bytes))
    print("Circular collapse:", collapse_circular(chunks, num_hash_bytes))
    print("MSB collapse:", collapse_leading(chunks, num_hash_bytes))
