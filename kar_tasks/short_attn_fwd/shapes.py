# B images x 12 heads; N=197 (ViT-B/16 @224). Nearby: other token counts (CLS-only is separate), text-like 64.
SCORED = [{"name": f"b{B}", "weight": 1, "args": {"BH": B * 12, "N": 197}} for B in (64, 256, 1024)]
NEARBY = [{"name": "b8", "args": {"BH": 96, "N": 197}}, {"name": "n64", "args": {"BH": 3072, "N": 64}},
          {"name": "n256", "args": {"BH": 1536, "N": 256}}, {"name": "n130", "args": {"BH": 768, "N": 130}}]
