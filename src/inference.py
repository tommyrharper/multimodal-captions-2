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
it = iter(dl)
txt, img = next(it)
img = img.to(dev)
txt = txt.to(dev)
with torch.no_grad():
    out = frk(img, txt)
print(out)
