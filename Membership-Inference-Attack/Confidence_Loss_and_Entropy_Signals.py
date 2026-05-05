taking entropy, loss and confidence score
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import requests
import random
import argparse
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve

# ─── config ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PUB_PATH  = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   # DO NOT CHANGE
API_KEY  = ""      # ← replace with your key
TASK_ID  = "01-mia"                 # DO NOT CHANGE


# ─── dataset classes (keep as-is from template) ───────────────────────────────
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_  = self.ids[index]
        img  = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# ─── load datasets ────────────────────────────────────────────────────────────
print("Loading datasets...")
pub_ds  = torch.load(PUB_PATH,  weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

# ─── normalization (same as training) ─────────────────────────────────────────
MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform  = transform
priv_ds.transform = transform

# ─── load target model ────────────────────────────────────────────────────────
print("Loading model...")
model = resnet18(weights=None)
model.conv1  = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc     = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Using device: {device}")


# ─── Signal 1: Max Confidence ─────────────────────────────────────────────────
def get_max_confidence_scores(dataset, mdl):
    """Higher max softmax prob → more likely a member."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False,collate_fn=safe_collate)
    all_ids, all_scores = [], []
    with torch.no_grad():
        for id_, imgs, labels, *_ in loader:
            imgs  = imgs.to(device)
            probs = F.softmax(mdl(imgs), dim=1)
            scores = probs.max(dim=1).values.cpu().numpy()
            all_ids.extend(id_.tolist())
            all_scores.extend(scores.tolist())
    return np.array(all_ids), np.array(all_scores)


# ─── Signal 2: Negative Loss ──────────────────────────────────────────────────
def safe_collate(batch):
    """Replace None membership values with -1 before collating."""
    fixed = []
    for item in batch:
        item = list(item)
        item = [-1 if x is None else x for x in item]
        fixed.append(item)
    from torch.utils.data.dataloader import default_collate
    return default_collate(fixed)

def get_negative_loss_scores(dataset, mdl):
    """Lower CE loss → more likely a member → negate so higher = member."""
    loader    = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=safe_collate)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_ids, all_scores = [], []
    with torch.no_grad():
        for id_, imgs, labels, *_ in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = mdl(imgs)
            loss   = criterion(logits, labels).cpu().numpy()
            scores = -loss   # negate: lower loss = more member-like
            all_ids.extend(id_.tolist())
            all_scores.extend(scores.tolist())
    return np.array(all_ids), np.array(all_scores)


# ─── Signal 3: Entropy (bonus signal) ────────────────────────────────────────
def get_entropy_scores(dataset, mdl):
    """Lower entropy → more confident → more member-like → negate entropy."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=safe_collate)
    all_ids, all_scores = [], []
    with torch.no_grad():
        for id_, imgs, labels, *_ in loader:
            imgs  = imgs.to(device)
            probs = F.softmax(mdl(imgs), dim=1)
            # entropy = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            scores  = -entropy.cpu().numpy()  # negate: lower entropy = member
            all_ids.extend(id_.tolist())
            all_scores.extend(scores.tolist())
    return np.array(all_ids), np.array(all_scores)


# ─── Normalize scores to [0, 1] ───────────────────────────────────────────────
def normalize(scores):
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-10)


# ─── Evaluate on pub.pt (known labels) ───────────────────────────────────────
def evaluate_tpr_at_fpr(scores, true_labels, fpr_threshold=0.05):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    idx = np.searchsorted(fpr, fpr_threshold)
    return tpr[idx]


# ─── Run attacks & pick best signal ──────────────────────────────────────────
print("\n--- Evaluating signals on pub.pt ---")

pub_labels = np.array(pub_ds.membership)  # ground truth: 1=member, 0=non-member

# Signal 1
pub_ids, pub_conf = get_max_confidence_scores(pub_ds, model)
tpr1 = evaluate_tpr_at_fpr(normalize(pub_conf), pub_labels)
print(f"Max Confidence    → TPR@5%FPR: {tpr1:.4f}")

# Signal 2
pub_ids, pub_negloss = get_negative_loss_scores(pub_ds, model)
tpr2 = evaluate_tpr_at_fpr(normalize(pub_negloss), pub_labels)
print(f"Negative Loss     → TPR@5%FPR: {tpr2:.4f}")

# Signal 3
pub_ids, pub_entropy = get_entropy_scores(pub_ds, model)
tpr3 = evaluate_tpr_at_fpr(normalize(pub_entropy), pub_labels)
print(f"Negative Entropy  → TPR@5%FPR: {tpr3:.4f}")

# Combined score (average of all normalized signals)
pub_combined = (normalize(pub_conf) + normalize(pub_negloss) + normalize(pub_entropy)) / 3
tpr_combined = evaluate_tpr_at_fpr(pub_combined, pub_labels)
print(f"Combined (avg)    → TPR@5%FPR: {tpr_combined:.4f}")

# Pick best
best_tpr = max(tpr1, tpr2, tpr3, tpr_combined)
print(f"\nBest TPR@5%FPR on pub.pt: {best_tpr:.4f}")


# ─── Generate scores for priv.pt using best method ───────────────────────────
print("\n--- Generating scores for priv.pt ---")

priv_ids_conf,    priv_conf    = get_max_confidence_scores(priv_ds, model)
priv_ids_negloss, priv_negloss = get_negative_loss_scores(priv_ds, model)
priv_ids_entropy, priv_entropy = get_entropy_scores(priv_ds, model)

# Use combined score (usually best)
priv_scores = (
    normalize(priv_conf) +
    normalize(priv_negloss) +
    normalize(priv_entropy)
) / 3


# ─── Save submission.csv ──────────────────────────────────────────────────────
print("Saving submission.csv ...")
df = pd.DataFrame({
    "id":    [str(i) for i in priv_ids_conf],
    "score": priv_scores
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}  ({len(df)} rows)")


# ─── Submit to leaderboard ────────────────────────────────────────────────────
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV
if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()
    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
