import numpy as np
import struct


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


def read_ibin(filename):
    n, d = np.fromfile(filename, count=2, dtype="int32")
    a = np.fromfile(filename, dtype="int32")
    print(f"\t{filename} readed. shape: {(n, d)}")
    if len(a) != 2 + n * d:
        print(f"Warning: file size mismatch {len(a)} vs {2+n*d}")
    return a[2:2+n*d].reshape(n, d)


def read_fbin(filename):
    return read_ibin(filename).view("float32")

def write_fbin(filename, m):
    """
    Ultra-fast writing of fbin format.
    Fbin format: [N(int32) | D(int32) | v1_1(int32) v1_2(int32) ... v1_D(int32) | v2_1 ... | ...]
    All data stored as int32 (float32 viewed as int32)

    Args:
        filename: Output file path
        m: Numpy array of shape (N, D) with float32 data
    """
    m = m.astype("float32")
    n, d = m.shape

    with open(filename, "wb") as f:
        # Write header: N and D as int32
        f.write(struct.pack("ii", n, d))
        # Write all vector data as int32 (reinterpret float32 as int32)
        f.write(m.tobytes())

    print(f"Written {filename} ({(8 + n * d * 4) / 1e9:.2f} GB)")

def read_somefiles(filename):
    if filename.endswith(".fvecs"):
        return read_fvecs(filename)
    elif filename.endswith(".ivecs"):
        return read_ivecs(filename)
    elif filename.endswith(".fbin"):
        return read_fbin(filename)
    elif filename.endswith(".bin"):
        return read_ibin(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
