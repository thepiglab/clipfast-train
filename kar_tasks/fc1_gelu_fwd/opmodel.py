def min_bytes(M, K, N):
    return {"read": 2 * (M * K + N * K + N), "write": 2 * 2 * M * N}


def flops(M, K, N):
    return {"tc_bf16": 2 * M * K * N}
