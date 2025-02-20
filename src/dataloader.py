import torch
from datasets import load_dataset
from transformers import CLIPProcessor


class Flick(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.ds = load_dataset("nlphuji/flickr30k", split="test", cache_dir="./flickr")
        self.pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        itm = self.ds[idx]
        img = itm["image"]
        txt = itm["caption"][0]
        enc = self.pr(text=txt, images=img, return_tensors="pt")
        img = enc["pixel_values"].squeeze()
        tkn = enc["input_ids"].squeeze()
        return tkn, img


if __name__ == "__main__":
    ds = Flick()
    tkn, img = ds[0]
    print("tkn", tkn.shape)  # torch.Size([19])
    print("img", img.shape)  # torch.Size([3, 224, 224])
