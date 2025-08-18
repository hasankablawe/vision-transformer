# Vision-Transformer-ViT

🚀 A simple replication of the **Vision Transformer (ViT)** paper:  
<img width="693" height="364" alt="Screenshot 2025-08-18 223710" src="https://github.com/user-attachments/assets/ef7883c4-d3b8-462b-848e-0b31a74a2990" />


---

## 📌 Project Overview
This repository contains a PyTorch implementation of the Vision Transformer (ViT) architecture.  
The goal of this project is **educational**: to build ViT from scratch and understand how transformers can be applied to computer vision tasks.

---

## ⚡ Features
- Patch Embedding (splitting images into fixed-size patches).
- Learnable Positional Embeddings.
- Transformer Encoder (Multi-Head Self Attention + MLP).
- Classification Head.
- Configurable parameters (patch size, embedding dim, depth, heads, etc.).
- Training pipeline on small datasets (CIFAR-10 / MNIST).

---

## 🛠 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/vision-transformer.git
cd vision-transformer
pip install -r requirements.txt
