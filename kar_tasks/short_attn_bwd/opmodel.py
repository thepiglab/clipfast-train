def min_bytes(BH, N):
    e = BH * N * 64 * 2
    return {"read": 5 * e + BH * N * 4, "write": 3 * e}


def flops(BH, N):
    # recompute S (1), dV (1), dP (1), dQ (1), dK (1) GEMMs of N x N x 64
    return {"tc_bf16": 5 * 2 * BH * N * N * 64}
