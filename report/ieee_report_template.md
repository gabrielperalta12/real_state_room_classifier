# Indoor Real Estate Room Classifier using CLIP Embeddings

## Abstract

This paper presents an academic prototype for automatic indoor room classification in real estate images using multimodal models and transfer learning. The proposed system uses CLIP embeddings for zero-shot classification and as visual representations for lightweight supervised classifiers. The target classes include living room, kitchen, bedroom, bathroom, dining room, laundry room, balcony or terrace, and gym or fitness room. Experimental results should be reported only after evaluating the system on an authorized public or local dataset.

## Keywords

Multimodal learning, CLIP, zero-shot classification, transfer learning, real estate images, indoor scene classification, image embeddings.

## I. Introduction

Real estate platforms commonly contain large collections of indoor images. Automatically identifying the type of room represented in each image can support search, catalog organization, recommendation systems, and listing quality control. However, real estate images often present high visual variability, mixed spaces, different lighting conditions, and non-standard camera perspectives.

This project addresses automatic classification of indoor real estate environments using CLIP, a multimodal vision-language model. The study focuses on an MVP that combines zero-shot classification with supervised learning over frozen image embeddings.

## II. Related Work

Indoor scene classification has traditionally relied on handcrafted visual features and later on convolutional neural networks. More recent approaches use transfer learning from large-scale pretrained models. Vision-language models such as CLIP enable zero-shot classification by comparing images with natural language prompts, reducing the need for large labeled datasets.

Discuss in this section prior work on indoor scene recognition, transfer learning, CLIP, and zero-shot image classification. Include only verifiable references used during the course work.

## III. Materials and Methods

### Dataset

The system is designed for local folders organized by class. Images may come from public datasets, own photographs, or authorized local collections. No unauthorized scraping from private platforms should be performed.

Classes:

| Internal label | Semantic label |
|---|---|
| `sala` | living room |
| `cocina` | kitchen |
| `dormitorio` | bedroom |
| `bano` | bathroom |
| `comedor` | dining room |
| `lavanderia` | laundry room |
| `balcon_terraza` | balcony or terrace |
| `gimnasio` | gym or fitness room |

### Zero-Shot Classification

Each image is compared against textual prompts such as "a photo of a kitchen in a real estate listing". CLIP produces image-text similarity scores, which are transformed into probabilities using softmax.

### Embedding Extraction

The image encoder of CLIP extracts fixed-length visual embeddings. These embeddings are L2-normalized and saved as NumPy arrays for downstream training.

### Supervised Classifiers

Two lightweight classifiers are trained over the embeddings:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline with balanced class weights |
| SVM | RBF-kernel Support Vector Machine with probability estimates |

## IV. Proposed Architecture

The proposed pipeline consists of four stages:

1. Image ingestion from class-organized local folders.
2. CLIP-based zero-shot inference or embedding extraction.
3. Training of lightweight classifiers over frozen embeddings.
4. Evaluation and reporting through CSV metrics and confusion matrix figures.

The implementation uses Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy, matplotlib, and seaborn.

## V. Experiments and Results

Do not invent results. Complete the following tables after running the experiments.

### Dataset Summary

| Class | Number of images |
|---|---:|
| sala | TBD |
| cocina | TBD |
| dormitorio | TBD |
| baño | TBD |
| comedor | TBD |
| lavandería | TBD |
| balcón/terraza | TBD |
| gimnasio | TBD |
| Total | TBD |

### Quantitative Results

| Model | Accuracy | Precision weighted | Recall weighted | F1 weighted |
|---|---:|---:|---:|---:|
| CLIP zero-shot | TBD | TBD | TBD | TBD |
| CLIP embeddings + Logistic Regression | TBD | TBD | TBD | TBD |
| CLIP embeddings + SVM | TBD | TBD | TBD | TBD |

### Confusion Matrix

Insert the generated confusion matrix from `outputs/figures/confusion_matrix.png` after evaluation.

## VI. Discussion

Discuss which classes are easier or harder to classify, possible confusion between integrated spaces, dataset imbalance, and differences between zero-shot and supervised embedding-based classification. Address practical limitations for real estate images, including lighting, perspective, image quality, and rooms containing multiple functional areas.

## VII. Conclusions and Future Work

This project demonstrates a compact transfer learning pipeline for indoor real estate room classification. CLIP enables zero-shot inference without labeled training data, while embeddings allow training lightweight classifiers when labeled examples are available.

Future work may include evaluating larger CLIP variants, using cross-validation, adding multi-label classification for integrated spaces, calibrating confidence scores, and building a user-facing demo application.

## References

[1] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML, 2021.

[2] Hugging Face, "Transformers Documentation," https://huggingface.co/docs/transformers.

[3] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, 2011.

Add course-required references and dataset citations here.
