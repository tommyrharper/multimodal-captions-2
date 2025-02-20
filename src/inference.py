import torch
from src.dataloader import Flick
from src.models import Frank
from torch.backends import mps
from torch import cuda
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

torch.manual_seed(42)
device_type = "mps" if mps.is_available() else "cuda" if cuda.is_available() else "cpu"
print("device_type", device_type)
dev = torch.device(device_type)

frk = Frank().to(dev)
frk.eval()
ds = Flick(split="test")
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=ds.collate)
txt, img = next(iter(dl))
img = img.to(dev)

# Autoregressive generation
tokens = torch.tensor([[ds.tk.bos_token_id]], device=dev)
print('ds.tk.bos_token_id', ds.tk.bos_token_id)
print('tokens', tokens)
print('tokens.shape', tokens.shape)
print('---------- loop about to start:')
with torch.no_grad():
    for i in range(77):  # Max length safety limit
        print('----------- loop round:', i)
        print('img.shape', img.shape)
        print('tokens.shape', tokens.shape)
        out = frk(img, tokens)
        print('out.shape', out.shape)
        next_token_dist = out[:, -1, :]
        print('next_token_dist.shape', next_token_dist.shape)

        # Preemptively penalize tokens
        # 1. Prevent tokens that appear 3 or more times
        token_counts = torch.bincount(tokens[0], minlength=ds.tk.vocab_size)
        frequent_tokens = torch.where(token_counts >= 2)[0]
        next_token_dist[0, frequent_tokens] = float("-inf")

        # prevent immediate repetition
        if i > 0:  # Skip first iteration since there's no previous token
            last_token = tokens[0, -1]
            next_token_dist[0, last_token] = float("-inf")

        # 2. Prevent BOS token
        next_token_dist[0, ds.tk.bos_token_id] = float("-inf")

        # Select the next token after all modifications
        next_token = torch.argmax(next_token_dist, dim=1)
        print('next_token.shape', next_token.shape)
        print('next_token', next_token)
        print('eos token', ds.tk.eos_token_id)

        if next_token.item() == ds.tk.eos_token_id:
            print('----> breaking -> eos found')
            break

        # Append the selected token
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        print('tokens.shape', tokens.shape)

caption = ds.tk.decode(tokens[0])
print("Generated caption:", caption)
print("Ground truth:", ds.tk.decode(txt[0]))
