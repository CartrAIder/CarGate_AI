"""Product recognition embedder: frozen DINOv2-S/14 backbone + a trainable
ArcFace projection head, trained on GPU-augmented cutouts.

The head reshapes DINOv2 features so same-SKU crops cluster tightly and different
SKUs push apart, giving clean cosine matching against a per-cart receipt gallery.
All augmentation (rotate/scale, paste on random backgrounds, blur/low-res/noise)
runs as batched GPU ops so training is GPU-bound, and fresh views are drawn each
step. Only the projection head + ArcFace prototypes train; the backbone is frozen.

    conda run -n cartgate python train_recognition.py    # -> dino_arc.onnx (256-d)
"""
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cartgate.train_embed import load_cutouts, IMSZ

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoBackbone(nn.Module):
    def __init__(self, model="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.net = timm.create_model(model, pretrained=True, num_classes=0, img_size=IMSZ)
        self.dim = self.net.num_features

    def forward(self, x):
        return self.net(x)


class ProjHead(nn.Module):
    def __init__(self, din=384, dhid=512, dout=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dhid), nn.BatchNorm1d(dhid), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dhid, dout), nn.BatchNorm1d(dout),
        )

    def forward(self, x):
        return self.net(x)


class ArcMargin(nn.Module):
    def __init__(self, dim, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_cls, dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, feat, label):
        cos = F.normalize(feat) @ F.normalize(self.W).t()
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)
        oh = F.one_hot(label, cos.size(1)).float()
        return self.s * (oh * target + (1 - oh) * cos)


class Embed(nn.Module):
    """Deployable embedder: L2-normalized proj(backbone(x))."""
    def __init__(self, back, head):
        super().__init__()
        self.back, self.head = back, head

    def forward(self, x):
        v = self.head(self.back(x))
        return v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)


def prep_cutouts(cut, S=160):
    """Pack all cutouts into one [K, 4, S, S] RGBA tensor on the GPU."""
    skus = sorted(cut.keys())
    tens, labs = [], []
    for i, sku in enumerate(skus):
        for co in cut[sku]:
            a = co[:, :, 3]
            ys, xs = np.where(a > 0)
            if len(ys) < 5:
                continue
            co = co[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            co = cv2.resize(co, (S, S), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(co[:, :, :3], cv2.COLOR_BGR2RGB)
            rgba = np.concatenate([rgb, co[:, :, 3:4]], axis=2).astype(np.float32) / 255.0
            tens.append(torch.from_numpy(rgba.transpose(2, 0, 1)))
            labs.append(i)
    return torch.stack(tens).to(DEV), torch.tensor(labs, device=DEV), skus


def make_bg_bank(nb, size):
    """A bank of random backgrounds (gradient blends + noise + coarse clutter)."""
    grad = torch.linspace(0, 1, size, device=DEV).view(1, 1, size, 1)
    c1 = torch.rand(nb, 3, 1, 1, device=DEV)
    c2 = torch.rand(nb, 3, 1, 1, device=DEV)
    bg = c1 * (1 - grad) + c2 * grad
    bg = bg + 0.06 * torch.randn(nb, 3, size, size, device=DEV)
    clutter = 0.15 * torch.randn(nb, 3, size // 8, size // 8, device=DEV)
    clutter = F.interpolate(clutter, size=(size, size), mode="bilinear", align_corners=False)
    m = (torch.rand(nb, 1, 1, 1, device=DEV) < 0.5).float()
    return (bg + m * clutter).clamp(0, 1)


def _gauss_kernel(sigma, ch=3):
    r = 3
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    k2 = (k[:, None] * k[None, :]).view(1, 1, 2 * r + 1, 2 * r + 1)
    return k2.repeat(ch, 1, 1, 1).to(DEV)


def aug_batch(objs, labels, bg, B, size, gk):
    """Draw a batch of augmented training images entirely on the GPU."""
    idx = torch.randint(0, objs.size(0), (B,), device=DEV)
    obj, y = objs[idx], labels[idx]
    S = obj.size(-1)
    canvas = torch.zeros(B, 4, size, size, device=DEV)
    off = (size - S) // 2
    canvas[:, :, off:off + S, off:off + S] = obj

    # random rotation + scale (object fills 50-95%) + translation
    ang = torch.rand(B, device=DEV) * (2 * np.pi)
    fill = 0.5 + torch.rand(B, device=DEV) * 0.45
    invk = S / (fill * size)
    cos, sin = torch.cos(ang) * invk, torch.sin(ang) * invk
    shift = 1 - fill
    tx = (torch.rand(B, device=DEV) * 2 - 1) * shift
    ty = (torch.rand(B, device=DEV) * 2 - 1) * shift
    theta = torch.zeros(B, 2, 3, device=DEV)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, (B, 4, size, size), align_corners=False)
    w = F.grid_sample(canvas, grid, align_corners=False, padding_mode="zeros")
    rgb, alpha = w[:, :3], w[:, 3:4].clamp(0, 1)

    img = alpha * rgb + (1 - alpha) * bg[torch.randint(0, bg.size(0), (B,), device=DEV)]
    img = (img * (0.6 + torch.rand(B, 1, 1, 1, device=DEV) * 0.7)
               * (0.85 + torch.rand(B, 3, 1, 1, device=DEV) * 0.3))

    mb = (torch.rand(B, 1, 1, 1, device=DEV) < 0.35).float()
    blurred = F.conv2d(F.pad(img, (3, 3, 3, 3), mode="reflect"), gk, groups=3)
    img = mb * blurred + (1 - mb) * img

    if torch.rand(1).item() < 0.6:                       # simulate a low-res camera
        f = int(size * (0.30 + 0.40 * torch.rand(1).item()))
        low = F.interpolate(img, size=(f, f), mode="bilinear", align_corners=False)
        low = F.interpolate(low, size=(size, size), mode="bilinear", align_corners=False)
        ml = (torch.rand(B, 1, 1, 1, device=DEV) < 0.5).float()
        img = ml * low + (1 - ml) * img

    img = img + torch.randn_like(img) * (torch.rand(B, 1, 1, 1, device=DEV) * 0.06)
    img = img.clamp(0, 1)
    return (img - MEAN.to(DEV)) / STD.to(DEV), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--dout", type=int, default=256)
    ap.add_argument("--out", default="dino_arc.onnx")
    args = ap.parse_args()

    objs, labels, skus = prep_cutouts(load_cutouts())
    print(f"device={DEV} SKUs={len(skus)} cutouts={objs.size(0)} "
          f"batch={args.batch} iters/ep={args.iters}", flush=True)
    bg = make_bg_bank(256, IMSZ)
    gk = _gauss_kernel(1.2)

    back = DinoBackbone().to(DEV).eval()
    for p in back.parameters():
        p.requires_grad = False
    head = ProjHead(back.dim, 512, args.dout).to(DEV)
    arc = ArcMargin(args.dout, len(skus)).to(DEV)
    opt = torch.optim.AdamW(list(head.parameters()) + list(arc.parameters()), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    WARMUP, M_FINAL = 8, 0.30

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        arc.m = M_FINAL * min(1.0, ep / WARMUP)          # margin warm-up stabilizes small data
        head.train(); arc.train()
        tot = corr = 0
        ls = 0.0
        for _ in range(args.iters):
            with torch.no_grad():
                img, y = aug_batch(objs, labels, bg, args.batch, IMSZ, gk)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                    feat = back(img)
                feat = feat.float()
            opt.zero_grad()
            logits = arc(head(feat), y)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            ls += loss.item() * len(y); tot += len(y); corr += (logits.argmax(1) == y).sum().item()
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:2d} loss={ls/tot:.3f} train_acc={corr/tot:.3f} "
                  f"m={arc.m:.3f} [{time.time()-t0:.0f}s]", flush=True)

    emb = Embed(back, head).to("cpu").eval()
    torch.onnx.export(emb, torch.randn(1, 3, IMSZ, IMSZ), args.out,
                      input_names=["input"], output_names=["emb"],
                      opset_version=17, dynamic_axes={"input": {0: "b"}, "emb": {0: "b"}})
    print(f"exported {args.out}  ({args.dout}-d)  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
