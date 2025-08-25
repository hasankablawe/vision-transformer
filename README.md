# Vision Transformer (ViT) — Paper Replication  
*"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* (Dosovitskiy et al., 2020)

---

## 📖 Overview  

This repository is an **educational implementation** of the Vision Transformer (ViT) model from the influential paper:  
[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).  

The paper introduced a paradigm shift in computer vision by applying the **Transformer architecture**, originally designed for NLP, to image recognition tasks. Instead of using convolutional layers (CNNs), images are divided into fixed-size patches (e.g., `16x16`), flattened, and processed as a sequence of tokens — just like words in text.

---

## 🧩 How Vision Transformer Works  

1. **Image to Patches**  
   - An image (e.g., `224x224x3`) is split into smaller patches (e.g., `16x16`).  
   - Each patch is flattened into a vector.  
   - The collection of patch vectors is treated as a sequence (like words in NLP).  

2. **Linear Projection + Position Embedding**  
   - Each patch is linearly projected into a fixed dimension (`D`).  
   - Positional encodings are added so the model retains spatial information.  

3. **Transformer Encoder**  
   - A stack of **multi-head self-attention + feed-forward layers** processes the sequence.  
   - Self-attention allows patches to "communicate" globally, unlike CNNs which are local.  

4. **Classification Token (`[CLS]`)**  
   - A special learnable token is prepended to the sequence.  
   - After the Transformer encoder, this token contains global image representation.  
   - A final linear layer maps it to class probabilities.  

---

## 📊 Key Contributions from the Paper  

- **Scalability**: ViT performs extremely well when trained on very large datasets (e.g., ImageNet-21k, JFT-300M).  
- **Simplicity**: Removes convolutions, instead using only a pure Transformer encoder.  
- **Performance**: Outperforms ResNets when trained on sufficient data.  
- **Transfer Learning**: Pretrained ViTs transfer well to mid-sized datasets (CIFAR, Flowers, etc.).  

---

## 🏗️ Repository Structure  

```bash
vision-transformer/
│── VIT_model.py        # Vision Transformer model implementation
│── train.py            # Training script
│── engine.py           # Train and test helpers
│── data.py             # Dataset download and dataloader setup
│── utils/              # Helper functions (saving models, plotting, etc.)
│── requirements.txt    # Dependencies
│── README.md           # Project documentation (this file)
```

---

## 🚀 Getting Started  

### Install dependencies
```bash
pip install -r requirements.txt
```

### Clone the repo
```bash
git clone https://github.com/hasankablawe/vision-transformer.git
cd vision-transformer
```

### Train the model
```bash
python train.py
```
