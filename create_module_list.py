import torch
import re

module = torch.jit.load("/Users/lucaayscough/Downloads/zee-sync-lowlat-v1_e924c81e40_streaming.ts")

log_text = str(module)

pattern = re.compile(r"original_name=([A-Za-z0-9_]+)")
found_names = re.findall(pattern, log_text)

unique_names = list(set(found_names))

print(unique_names)
