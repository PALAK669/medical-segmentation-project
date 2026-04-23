# Medical Segmentation with SDF-Guided Learning

A PyTorch-based framework for 3D medical image segmentation that integrates **Signed Distance Function (SDF)** as an auxiliary supervision signal to improve boundary precision and structural understanding in volumetric data.

---

## The Problem: Boundary Ambiguity in Segmentation Models

Conventional segmentation models treat medical images as pixel-wise classification tasks, relying solely on binary masks for supervision. While effective in general settings, this approach suffers from a fundamental limitation:

**Lack of Geometric Awareness**

Binary masks provide no information about how far a voxel lies from the object boundary. As a result:

* Boundaries appear blurry or poorly defined
* Fine anatomical structures are often lost
* Models struggle in regions with low contrast or noise

This issue becomes especially critical in **3D medical imaging**, where precise boundary delineation is essential for diagnosis and analysis.

---

## The Solution: Signed Distance Function (SDF) Supervision

To address this limitation, this project introduces **Signed Distance Function (SDF)** as an additional supervision signal.

Instead of learning only from binary labels, the model is guided by continuous distance information:

* Positive values → Outside the object
* Negative values → Inside the object
* Zero → Exact boundary

This transforms the learning process from simple classification to **geometry-aware representation learning**.

The Signed Distance Function is defined as:

[
SDF(x) = d_{\text{outside}}(x) - d_{\text{inside}}(x)
]

where distance transforms are computed for both foreground and background regions.

---

## Dataset Pipeline

A custom dataset class is designed to handle medical image data efficiently.

### Key Components

* Loads image and mask data from given file paths
* Computes Signed Distance Function (SDF) from binary masks
* Generates normalized spatial coordinates for sampling
* Returns structured tensors for training

Each sample contains:

* **Image** → Input tensor
* **Mask** → Ground truth segmentation
* **SDF** → Distance-based supervision signal
* **Coordinates** → Random spatial samples for learning

This design enables compatibility with **coordinate-based learning approaches** and advanced neural representations.

---

## Training Framework

A modular training pipeline is implemented using a custom trainer class.

### Training Process

* Load batches from dataset
* Perform forward pass through the model
* Compute segmentation loss
* Backpropagate gradients
* Update model parameters
* Track average training loss per epoch

The framework is designed to be **flexible, lightweight, and extensible**.

---

## Key Contributions

* Introduces **Signed Distance Function (SDF)** for enhanced supervision
* Improves **boundary precision** in segmentation tasks
* Implements a custom dataset pipeline
* Supports **geometry-aware learning**
* Provides a clean and modular training framework

---

## Applications

This framework can be applied to:

* Medical image segmentation
* Boundary refinement tasks
* Image-based structural analysis
* Research in geometry-aware deep learning

---

## Limitations

* Requires additional computation for SDF generation
* Performance depends on quality of mask annotations
* Currently supports binary segmentation

---

## Future Work

* Multi-class segmentation support
* Integration with physics-informed models (PINNs)
* Advanced loss functions (Dice + SDF hybrid)
* Quantitative evaluation (Dice Score, IoU)
* Data augmentation for improved generalization

---

## Conclusion

This project demonstrates that incorporating **geometric priors** through Signed Distance Functions significantly enhances segmentation performance. By moving beyond pixel-wise classification, the model learns a richer representation of spatial structure, leading to more accurate and reliable predictions.
