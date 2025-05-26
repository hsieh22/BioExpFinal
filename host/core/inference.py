import numpy as np
import os, time
import torch, torch.nn as nn
from torch.utils.data import Dataset
import json
import torch.nn.functional as F

LABELS = ["relax", "focus", "stress", "memory"]   # map to 0,1,2,3 order
TRIAL_LEN = 5000                                 # samples (10 s @ 500 Hz)

class InferenceEngine:
    def __init__(self, model_path):
        """Load the trained model weights to CPU."""
        self.model = EEGNetWithFeatureFusionTransformer(feat_dim=14)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()          # inference mode
        print(f"🤖 Model loaded from {model_path}")
        # print("🤖 Model loaded")

    def predict_from_json(self, eeg_json): # eeg_json is a vector with 5000 numbers for the model to predict
        """Return state probabilities for a 5 000-sample vector."""
        vec = (np.asarray(json.loads(eeg_json), dtype=np.float32)
               if isinstance(eeg_json, str) else
               np.asarray(eeg_json,        dtype=np.float32))

        wf, feat = self._prepare_input(vec)

        with torch.no_grad():
            logits = self.model(wf, feat)          # (1,4)
            # probs  = F.softmax(logits, dim=1).numpy()[0] # V1
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0] # GPT
            # probs = torch.softmax(logits, dim=1).cpu().numpy()[0]   # from maze model

        return {lbl: float(p) for lbl, p in zip(LABELS, probs)}
        # # 模擬輸出機率
        # return {
        #     'relax': float(np.random.rand()),
        #     'focus': float(np.random.rand()),
        #     'memory': float(np.random.rand()),
        #     'stress': float(np.random.rand())
        # }
    def _prepare_input(self, eeg_vec: np.ndarray):
        """
        Returns waveform and feature tensors on CPU, always exactly TRIALLEN
        samples long.  If eeg_vec is longer, it is trimmed; if shorter, raise
        an error (caller can decide how to pad).
        """
        n = eeg_vec.shape[0]

        if n < TRIAL_LEN:
            raise ValueError(f"EEG vector only {n} samples (<{TRIAL_LEN}); "
                            "not enough for one window.")

        if n > TRIAL_LEN:
            if False:
                eeg_vec = eeg_vec[-TRIAL_LEN:]      # keep last window
            else:
                eeg_vec = eeg_vec[:TRIAL_LEN]       # keep first window

        # n == TRIALLEN or trimmed to that length
        wf = torch.from_numpy(eeg_vec[np.newaxis, np.newaxis, :])  # (1,1,5000)

        feat_np = compute_features(eeg_vec).reshape(1, -1).astype(np.float32)
        feat    = torch.from_numpy(feat_np)                        # (1,14)
        return wf, feat
    


def compute_features(trial, fs: int = 500) -> np.ndarray:
    """
    Return order:
        0  mean
        1  std
        2  delta_abs     (0.5–4 Hz)
        3  theta_abs     (4–8 Hz)
        4  alpha_abs     (8–13 Hz)
        5  beta_abs      (13–30 Hz)
        6  spec_entropy
        7  alpha_beta_ratio
        8  delta_rel     ★ absolute / total
        9  theta_rel     ★
       10  alpha_rel     ★
       11  beta_rel      ★
       12  hjorth_mob    ★
       13  hjorth_comp   ★
    Shape: (14,)
    """
    mean_val, std_val = trial.mean(), trial.std()

    # ---- PSD ---------------------------------------------------------
    freqs  = np.fft.rfftfreq(len(trial), d=1 / fs)
    pow_s  = np.abs(np.fft.rfft(trial)) ** 2          # power spectrum
    total_power = pow_s.sum() + 1e-12

    def bp(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return pow_s[idx].sum()

    delta = bp(0.5, 4)
    theta = bp(4,   8)
    alpha = bp(8,  13)
    beta  = bp(13, 30)

    # ---- spectral entropy -------------------------------------------
    p = pow_s / total_power
    spec_ent = -(p * np.log2(p + 1e-12)).sum()

    # ---- ratios + relative powers -----------------------------------
    alpha_beta = alpha / (beta + 1e-6)
    delta_rel  = delta / total_power
    theta_rel  = theta / total_power
    alpha_rel  = alpha / total_power
    beta_rel   = beta  / total_power

    # ---- Hjorth parameters ------------------------------------------
    diff1 = np.diff(trial)
    diff2 = np.diff(diff1)

    var0 = np.var(trial)
    var1 = np.var(diff1) + 1e-12
    var2 = np.var(diff2) + 1e-12

    hjorth_mob  = np.sqrt(var1 / var0)               # mobility
    hjorth_comp = np.sqrt(var2 / var1) / hjorth_mob  # complexity

    return np.array(
        [mean_val, std_val,
         delta, theta, alpha, beta,
         spec_ent, alpha_beta,
         delta_rel, theta_rel, alpha_rel, beta_rel,
         hjorth_mob, hjorth_comp],
        dtype=np.float32
    )

# ----------------------------------------------------------------------
# 1.  Dataset class
# ----------------------------------------------------------------------
class EEGDatasetFusion(Dataset):
    """
    For training:
        EEGDatasetFusion(root_dir, ["S01", …], inference=False, train_subset=True)
        -> each item = (waveform, feature, label)

    For inference / prediction:
        EEGDatasetFusion(root_dir, ["S19"], inference=True)
        -> each item = (waveform, feature)
    """
    def __init__(self,
                 root_dir: str,
                 subject_list: list[str],
                 trial_len: int      = 5000,
                 sampling_rate: int  = 500,
                 train_subset: bool  = False,
                 inference: bool     = False):

        self.waveforms, self.features, self.labels = [], [], []

        # ── label handling ──────────────────────────────────────────
        if not inference:
            label_map   = {'1':0, '2':1, '3':2, '4':2, '5':3, '6':3}
            keep_digits = {'1','2','3','6'} if train_subset else label_map.keys()
        else:
            label_map   = None          # no labels at inference
            keep_digits = None          # accept every file

        # ── iterate subjects / files ───────────────────────────────
        for subj in subject_list:
            subj_dir = os.path.join(root_dir, subj)
            if not os.path.isdir(subj_dir):
                continue

            for fname in sorted(os.listdir(subj_dir)):
                if not fname.endswith(".txt"):
                    continue
                first_char = fname[0]

                # training: skip unwanted files
                if keep_digits is not None and first_char not in keep_digits:
                    continue

                path = os.path.join(subj_dir, fname)
                sig  = read_eeg_txt(path)                  # skip UTF-8 header
                n    = len(sig) // trial_len

                for i in range(n):
                    trial = sig[i*trial_len:(i+1)*trial_len]
                    if len(trial) != trial_len:            # ignore leftover
                        continue
                    self.waveforms.append(trial[np.newaxis, :].astype(np.float32))
                    self.features.append(compute_features(trial, fs=sampling_rate))
                    if not inference:
                        self.labels.append(label_map[first_char])

        # ── z-score handcrafted features ───────────────────────────
        self.features = np.asarray(self.features, dtype=np.float32)
        if len(self.features) > 0:
            mu, sigma = self.features.mean(0), self.features.std(0) + 1e-6
            self.features = (self.features - mu) / sigma

    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        wf   = torch.tensor(self.waveforms[idx])      # (1, 5000)
        feat = torch.tensor(self.features[idx])       # (14,)
        if self.labels:                               # training mode
            lab = torch.tensor(self.labels[idx])
            return wf, feat, lab
        return wf, feat                               # inference mode

    
# ----------------------------------------------------------------------
# 2.  Model class
# ----------------------------------------------------------------------
class EEGNetWithFeatureFusionTransformer(nn.Module):
    def __init__(self, feat_dim = 14, num_classes = 4):
        super().__init__()

        # ------------ CNN branch for waveform ------------
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=48, stride=6, padding=28),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)           # (B,64,32)
        )

        # ------------ Transformer encoder ---------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=256,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

        # ------------ MLP branch for hand-crafted features ------------
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32)
        )

        # ------------ Fusion classifier ------------
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    # ---------------------------------------------------
    def forward(self, waveform, features):
        x_wave = self.cnn(waveform)                 # (B,64,32)
        x_wave = self.transformer(x_wave.permute(0,2,1)).mean(dim=1)  # (B,64)

        x_feat = self.mlp(features)                 # (B,32)

        fused   = torch.cat([x_wave, x_feat], dim=1)  # (B,96)
        return self.classifier(fused)
    
# ----------------------------------------------------------------------
# helper: read EEG data from text file
# ----------------------------------------------------------------------
def read_eeg_txt(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        next(f)                              # skip header line
        return np.loadtxt(f, dtype=np.float32)