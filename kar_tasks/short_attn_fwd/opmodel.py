def min_bytes(BH, N):
    e = BH * N * 64 * 2
    return {"read": 3 * e, "write": e + BH * N * 4}


def flops(BH, N):
    return {"tc_bf16": 2 * 2 * BH * N * N * 64}
