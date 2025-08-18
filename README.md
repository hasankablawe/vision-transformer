# Vision-Transformer-ViT

🚀 A simple replication of the **Vision Transformer (ViT)** paper:  
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

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
