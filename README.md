# 🔬 Evaluating GELU vs ReLU in NN Architectures for Image Classification

This project benchmarks **GELU** vs **ReLU** activation functions using a **custom ResNet architecture built from scratch** in TensorFlow/Keras. The model is trained and evaluated on CIFAR-10 and CIFAR-100 datasets to analyze how different activations affect training convergence and classification accuracy.

---

## 🚀 Project Highlights

- ✨ **Custom ResNet Architecture**: Built GElU activation function from scratch using Keras functional API with full control over each residual block.
- 🧠 **Activation Function Focus**: Benchmarked **GELU** against **ReLU** in a fair and consistent training setup.
- ⚙️ **Distributed Training**: Employed TensorFlow's `tf.distribute.Strategy` for scalable training.
- 📊 **Performance Comparison**:
  - **CIFAR-10**: GELU peaked at **90.5%**, surpassing ReLU baseline (~88.4%).
  - **CIFAR-100**: GELU achieved **66.2%**, slightly outperforming ReLU (~64.7%).

---

## 🛠️ Technical Details

- **Framework**: TensorFlow 2.x / Keras  
- **Language**: Python  
- **Architecture**: ResNet (custom implementation, not prebuilt)  
- **Datasets**:  
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32×32 RGB images (10 classes)  
  - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32×32 RGB images (100 classes)  
- **Training Setup**:
  - Optimizer: `Adam`
  - Loss Function: `SparseCategoricalCrossentropy`
  - Metric: `SparseCategoricalAccuracy`
  - Epochs: 50
  - Batch Size: 128 (adjustable)
  - Strategy: `tf.distribute.MirroredStrategy` (for multi-GPU training)


---

## 📈 Results Summary

| Dataset   | Activation | Peak Accuracy |
|-----------|------------|----------------|
| CIFAR-10  | GELU       | **90.5%**       |
| CIFAR-10  | ReLU       | ~88.4%         |
| CIFAR-100 | GELU       | **66.2%**       |
| CIFAR-100 | ReLU       | ~64.7%         |

> ✅ **Observation**: GELU consistently outperformed ReLU in both CIFAR-10 and CIFAR-100, especially during early and late training phases.

---

## 🧠 Key Skills Demonstrated

- CNN architecture design with residual blocks (ResNet)
- GELU vs ReLU activation function analysis
- TensorFlow/Keras model implementation and training pipeline
- Performance benchmarking and result interpretation
- Use of distributed strategy for training efficiency

---

## 📚 References

- [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [GELU Activation Function – Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)

---





