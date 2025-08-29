#!/usr/bin/env python3
# stress_voice_demo.py
import sys, os
import numpy as np
import librosa

def extract_features(y, sr):
    hop_length = 512
    frame_length = 1024
    energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
        f0 = f0[~np.isnan(f0)]
        mean_f0 = float(np.mean(f0)) if len(f0)>0 else 0.0
        std_f0 = float(np.std(f0)) if len(f0)>0 else 0.0
    except Exception:
        mean_f0 = 0.0; std_f0 = 0.0
    mean_energy = float(np.mean(energy)) if len(energy)>0 else 0.0
    std_energy = float(np.std(energy)) if len(energy)>0 else 0.0
    voiced_ratio = float(np.sum(energy > np.percentile(energy, 60)))/(len(energy)+1e-9)
    return {"mean_f0":mean_f0, "std_f0":std_f0, "mean_energy":mean_energy, "std_energy":std_energy, "voiced_ratio":voiced_ratio}

def predict_stress(features):
    score = 0.0
    if features["mean_f0"] > 180: score += 1.0
    if features["std_f0"] > 30: score += 0.8
    if features["mean_energy"] > 1e-3: score += 1.0
    if features["std_energy"] > 1e-3: score += 0.8
    if features["voiced_ratio"] > 0.6: score += 0.5
    if score >= 3.0: lvl = "HIGH"
    elif score >= 1.5: lvl = "MEDIUM"
    else: lvl = "LOW"
    conf = min(0.99, max(0.2, score/4.0))
    return lvl, round(conf,2), score

def main():
    if len(sys.argv) < 2:
        print("Usage: python stress_voice_demo.py path/to/sample.wav")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print("File not found:", path); return
    print("Loading audio:", path)
    y, sr = librosa.load(path, sr=None)
    feats = extract_features(y, sr)
    lvl, conf, raw = predict_stress(feats)
    print("\n--- Voice Stress Report ---")
    print(f"Duration: {len(y)/sr:.2f}s")
    print("Features:", {k:round(v,4) for k,v in feats.items()})
    print(f"Predicted Stress Level: {lvl} (confidence: {conf})")
    print("---------------------------\n")

if __name__ == "__main__":
    main()
