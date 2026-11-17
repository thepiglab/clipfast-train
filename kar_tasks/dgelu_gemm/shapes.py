# ViT fc1 at B=64/128/256 images (197 tokens) and packed text (~14k tokens)
SCORED = [{"name": f"m{M}", "weight": 1, "args": {"M": M, "K": 768, "N": 3072}} for M in (12608, 25216, 50432)]
NEARBY = [{"name": "m13600_text", "args": {"M": 13600, "K": 768, "N": 3072}},
          {"name": "m1000", "args": {"M": 1000, "K": 768, "N": 3072}},
          {"name": "m197", "args": {"M": 197, "K": 768, "N": 3072}}]
