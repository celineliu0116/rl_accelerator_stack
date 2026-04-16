import struct
bin_data = open("firmware/include/model.bin", "rb").read()
magic, num_layers, ins, outs = struct.unpack("<IIII", bin_data[:16])
print(f"Magic: {magic:x}, layers: {num_layers}, ins: {ins}, outs: {outs}")
m, np, kp, tm, tn, burst, act, ws, w_off, b_off, pref, flags, c_nnz, c_col, c_ptr, sp = struct.unpack("<16I", bin_data[16:80])
print(f"w_off: {w_off}, b_off: {b_off}, c_nnz: {c_nnz}, c_ptr: {c_ptr}")
# parse ptrs
import numpy as np
row_ptrs = np.frombuffer(bin_data[c_ptr:c_ptr + 4*(np+1)], dtype=np.uint32)
print(f"Row ptrs: {row_ptrs}")
