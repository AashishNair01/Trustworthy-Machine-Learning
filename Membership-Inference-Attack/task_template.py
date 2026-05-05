import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse
from scipy.stats import norm
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data._utils.collate import default_collate


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "4e0cae0fcb684d189a32f5ce1e010c38"
TASK_ID = "01-mia"  #DONOT CHANGE
eps=1e-8



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
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
        if hasattr(self, "membership") and index < len(self.membership):
            return id_, img, label, self.membership[index]
        return id_, img, label


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)



# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


def private_collate_fn(batch):
    """
    batch = list of samples
    each sample = (id, img, label, membership)
    membership is None → remove it
    """

    cleaned_batch = []

    for sample in batch:
        id_, img, label, _ = sample   # discard membership
        cleaned_batch.append((id_, img, label))

    return default_collate(cleaned_batch)


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

#Calculate confidence scores and cross-entropy loss on the private dataset
def compute_confidence_and_loss(model, dataloader,device='cpu'):
    ids_all = []
    conf_all = []
    loss_all= []
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for batch_ids, x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, _ = probs.max(dim=1)
            loss = criterion(logits, y)
            ids_all.extend(batch_ids)
            conf_all.extend(confidence.cpu().numpy())
            loss_all.extend((-loss).cpu().numpy())
    return (
        np.array(ids_all), 
        np.array(conf_all), 
        np.array(loss_all)
    )

#Load likelihood ratio statistics
lira = torch.load(BASE / "lira_max_logits_stats.pt",weights_only=False)
mu_in = lira["mu_in"]
sigma_in = lira["sigma_in"]
mu_out = lira["mu_out"]
sigma_out = lira["sigma_out"]

print(f'mu_in: {mu_in}')
print(f'mu_out: {mu_out}')
# likelihood ratio function
def lira_score(phi):
    log_in = norm.logpdf(phi, mu_in, sigma_in)
    log_out = norm.logpdf(phi, mu_out, sigma_out)
    return log_in - log_out



#calculate the Lira scores for the private dataset
device='cpu'
ids = []
raw_scores = []
priv_loader = DataLoader(priv_ds, batch_size=128, shuffle=False,collate_fn=private_collate_fn)
with torch.no_grad():
    for batch_ids, x, y in priv_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        true_logits = logits[torch.arange(len(y), device=device), y]
        other_logits = logits.clone()

        # Mask true class so only competing logits remain
        other_logits[torch.arange(len(y), device=device), y] = float("-inf")
        max_other_logits = other_logits.max(dim=1).values

        phi = (true_logits - max_other_logits).cpu().numpy()
        for i, sample_id in enumerate(batch_ids):
            ids.append(int(sample_id))
            raw_scores.append(lira_score(phi[i])) 

#normalize scores to [0,1]
def calibrated(phi, T=2.0):
    return 1 / (1 + np.exp(-phi / T))
raw_scores = np.array(raw_scores)

norm_scores = calibrated(raw_scores, T=2.0)

scores = []
for i in range(len(ids)):
    scores.append(float(norm_scores[i]))

#ensemble with confidence scores
ids_conf, conf_scores, loss_scores = compute_confidence_and_loss(model, priv_loader, device=device)
conf_scores = np.array(conf_scores)
loss_scores = np.array(loss_scores) 
#normalize confidence scores to [0,1]
conf_scores = (conf_scores - conf_scores.min()) / (conf_scores.max() - conf_scores.min() + eps)
#normalize loss scores to [0,1]
loss_scores = (loss_scores - loss_scores.min()) / (loss_scores.max() - loss_scores.min() + eps)
#combine scores with weights
alpha = 0.5  # weight for Lira score
beta = 0.2   # weight for confidence score
gamma = 0.3  # weight for loss score
final_scores = alpha * norm_scores + beta * conf_scores + gamma * loss_scores

print("Final scores calculated:", final_scores)

#save submission in csv format
df = pd.DataFrame({
    "id": ids,
    "score": final_scores
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)


# #submit
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