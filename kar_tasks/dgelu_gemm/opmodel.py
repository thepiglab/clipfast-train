def min_bytes(M, K, N):
    return {"read": 2 * (M * K + N * K + M * N), "write": 2 * M * N}


def flops(M, K, N):
    return {"tc_bf16": 2 * M * K * N}
