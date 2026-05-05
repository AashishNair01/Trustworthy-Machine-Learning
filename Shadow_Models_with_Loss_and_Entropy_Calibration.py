
"""
Shadow Model Membership Inference Attack
=========================================
Pipeline:
  1. Split pub.pt into 4 disjoint shadow datasets
  2. Train 4 shadow ResNet-18s (same arch as target)
  3. Use loss + entropy on TARGET model, calibrated with pub.pt labels
  4. Run target model on priv.pt → membership scores → submission.csv
"""

import sys
import random
import requests

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

# ── Config ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
PUB_PATH   = BASE / "pub.pt"
PRIV_PATH  = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"
API_KEY  = ""   # ← replace this
TASK_ID  = "01-mia"

NUM_SHADOW_MODELS = 32
SHADOW_EPOCHS     = 50
BATCH_SIZE        = 64
LR                = 0.1
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Dataset Classes ──────────────────────────────────────────────────────────
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []; self.imgs = []; self.labels = []
        self.transform = transform
    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform: img = self.transform(img)
        return self.ids[index], img, self.labels[index]
    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform); self.membership = []
    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=[0.7406, 0.5331, 0.7059],
                         std =[0.1491, 0.1864, 0.1301]),
])

# ── Custom collate for priv.pt (membership=None crashes default_collate) ────
def priv_collate(batch):
    ids    = torch.tensor([b[0] for b in batch])
    imgs   = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch])
    return ids, imgs, labels

# ── Model factory (identical arch to target) ─────────────────────────────────
def make_resnet18(num_classes=9):
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, num_classes)
    return m

def load_target_model():
    m = make_resnet18()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return m.eval().to(DEVICE)

# ── Train one shadow model ───────────────────────────────────────────────────
def train_shadow_model(subset):
    loader    = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model     = make_resnet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SHADOW_EPOCHS)
    model.train()
    for epoch in range(SHADOW_EPOCHS):
        total = 0.0
        for _, imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{SHADOW_EPOCHS}]  loss={total/len(loader):.4f}")
    return model.eval()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    print("\n[1/6] Loading datasets...")
    pub_ds  = torch.load(PUB_PATH,  weights_only=False)
    priv_ds = torch.load(PRIV_PATH, weights_only=False)
    pub_ds.transform = priv_ds.transform = transform
    print(f"  pub.pt={len(pub_ds)}  priv.pt={len(priv_ds)}")

    # 2. Split pub.pt into shadow chunks
    print(f"\n[2/6] Splitting into {NUM_SHADOW_MODELS} shadow chunks...")
    idx = list(range(len(pub_ds))); random.shuffle(idx)
    chunk = len(pub_ds) // NUM_SHADOW_MODELS
    chunks = [idx[i*chunk:(i+1)*chunk] for i in range(NUM_SHADOW_MODELS)]

    # 3. Train shadow models
    print(f"\n[3/6] Training {NUM_SHADOW_MODELS} shadow models ({SHADOW_EPOCHS} epochs each)...")
    for i, c in enumerate(chunks):
        print(f"\n  ── Shadow {i+1}/{NUM_SHADOW_MODELS} ──")
        mid = len(c) // 2
        print(f"  train={mid}  held-out={len(c)-mid}")
        sm = train_shadow_model(Subset(pub_ds, c[:mid]))
        del sm
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # 4. Load target model
    print("\n[4/6] Loading target model...")
    target      = load_target_model()
    loss_fn     = nn.CrossEntropyLoss(reduction='none')
    softmax_fn  = nn.Softmax(dim=1)

    # 5. Calibrate on pub.pt using ground-truth membership
    # Members → lower loss, lower entropy → negate both → higher score
    print("\n[5/6] Calibrating on pub.pt...")
    losses, entropies, truths = [], [], []
    with torch.no_grad():
        for _, imgs, labels, mem in DataLoader(pub_ds, 256, num_workers=0):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = target(imgs)
            l = loss_fn(logits, labels).cpu().numpy()
            p = softmax_fn(logits).cpu().numpy()
            e = -np.sum(p * np.log(p + 1e-9), axis=1)
            losses.extend(l); entropies.extend(e); truths.extend(mem.numpy())

    L, E, T  = np.array(losses), np.array(entropies), np.array(truths)
    feat_pub = np.stack([-L, -E], axis=1)
    scaler   = StandardScaler()
    clf      = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(scaler.fit_transform(feat_pub), T)

    scores_pub  = clf.predict_proba(scaler.transform(feat_pub))[:, 1]
    fpr, tpr, _ = roc_curve(T, scores_pub)
    print(f"  ✓ TPR@5%FPR on pub.pt = {np.interp(0.05, fpr, tpr):.4f}  (random=0.05)")

    # 6. Score priv.pt → submission.csv
    # FIX: use priv_collate to handle membership=None without crashing
    print("\n[6/6] Scoring priv.pt...")
    p_ids, p_losses, p_ent = [], [], []
    with torch.no_grad():
        for ids, imgs, labels in DataLoader(priv_ds, 256, num_workers=0,
                                            collate_fn=priv_collate):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = target(imgs)
            l = loss_fn(logits, labels).cpu().numpy()
            p = softmax_fn(logits).cpu().numpy()
            e = -np.sum(p * np.log(p + 1e-9), axis=1)
            p_ids.extend(ids.tolist()); p_losses.extend(l); p_ent.extend(e)

    feat_priv  = np.stack([-np.array(p_losses), -np.array(p_ent)], axis=1)
    priv_scores = clf.predict_proba(scaler.transform(feat_priv))[:, 1]

    df = pd.DataFrame({"id": p_ids, "score": priv_scores})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(df)} rows → {OUTPUT_CSV}")
    print(f"  Score range: [{priv_scores.min():.4f}, {priv_scores.max():.4f}]")

    print("\nSubmitting...")
    # submit(OUTPUT_CSV)


# ── Submission ────────────────────────────────────────────────────────────────
# def submit(path):
#     path = Path(path)
#     if not path.exists():
#         print(f"Not found: {path}", file=sys.stderr); sys.exit(1)
#     try:
#         with open(path, "rb") as f:
#             resp = requests.post(
#                 f"{BASE_URL}/submit/{TASK_ID}",
#                 headers={"X-API-Key": API_KEY},
#                 files={"file": (path.name, f, "application/csv")},
#                 timeout=(10, 600),
#             )
#         try:   body = resp.json()
#         except: body = {"raw": resp.text}
#         if resp.status_code == 413:
#             print("File too large.", file=sys.stderr); sys.exit(1)
#         resp.raise_for_status()
#         print("Submitted:", body)
#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")
#         r = getattr(e, "response", None)
#         if r:
#             try:   print(r.json())
#             except: print(r.text)
#         sys.exit(1)


if __name__ == "__main__":
    main()

