
# Sub-Golgi Protein Classification using ESM-2

This repository implements a complete pipeline for classifying protein sequences into **cis-Golgi** and **trans-Golgi** subtypes using **ESM-2 Transformer models**.  
The goal is to **reproduce and surpass** the performance reported in the *isGPT paper* while experimenting with multiple embedding sizes and data balancing methods.

We evaluate three ESM-2 models:

- **esm2_t6_8M_UR50D** (8M parameters)
- **esm2_t12_35M_UR50D** (35M parameters)
- **esm2_t30_150M_UR50D** (150M parameters)

Each model is trained and tested **with and without SMOTE**, allowing comparison of how synthetic minority oversampling affects the downstream classification performance.

---

## Project Structure

```
├── train.py              # Main training pipeline
├── data/
│   ├── trainingset.csv
│   └── testingset.csv
├── output/
│   ├── model weights
│   ├── embeddings (ESM-2 CLS vectors)
│   ├── confusion matrices
│   └── metrics JSON files
|   ....
└── requirements.txt
```

---

## Goals of the Project

1. **Extract high-quality sequence embeddings** using ESM-2 transformer models.  
2. Train a **binary classifier** to distinguish cis-Golgi vs. trans-Golgi proteins.  
3. Evaluate models using:  
   - Independent test performance  
   - 10-fold cross-validation  
   - Leave-One-Out (Jackknife)  
4. Compare:
   - Different ESM-2 embedding sizes  
   - SMOTE vs. non-SMOTE training  
5. Aim to **beat the accuracy and MCC** reported in the isGPT benchmark paper.

---

## Requirements

Your `requirements.txt` should include:

```
torch>=2.5.1
transformers>=4.55.4

scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
imbalanced-learn>=0.11.0

matplotlib>=3.7.0
seaborn>=0.12.0

sentencepiece>=0.1.99
protobuf==3.20.3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How Training Works

### 1. Load data  
The code reads `trainingset.csv` and `testingset.csv`, converting class labels into binary (cis = 1, trans = 0).

### 2. Optional: SMOTE balancing  
If `--use-smote` is enabled, synthetic minority samples are generated.  
The resulting balanced dataset is used for both **training** and **embedding extraction**.

### 3. Embedding generation  
The script extracts **CLS token embeddings** from the chosen ESM-2 model.  
They are saved to:

```
output/<model_name>_embeddings.npy
output/<model_name>_embeddings_smote.npy
```

### 4. Model training  
A lightweight feed-forward classifier is trained using the embeddings.

### 5. Evaluation  
The script automatically computes:

- Independent test metrics  
- Confusion matrix  
- Cross-validation performance  
- Jackknife performance  

All results are saved as JSON for easy comparison.

---

## How to Run the Models

### 8M model (no SMOTE)

```bash
python train.py --model facebook/esm2_t6_8M_UR50D --device cpu
```

### 8M model (with SMOTE)

```bash
python train.py --model facebook/esm2_t6_8M_UR50D --device cpu --use-smote
```

### 35M model

```bash
python train.py --model facebook/esm2_t12_35M_UR50D --device cpu
```

### 150M model (GPU recommended)

```bash
python train.py --model facebook/esm2_t30_150M_UR50D --use-smote
```

---

## Output Files

Training produces:

### 1. Metrics JSON
```
output/esm2_t30_150M_UR50D_results.json
output/esm2_t30_150M_UR50D_results_smote.json
```

Contains accuracy, sensitivity, specificity, MCC, confusion matrix counts, CV scores, and jackknife scores.

### 2. Embedding files  
```
output/esm2_t30_150M_UR50D_embeddings.npy
output/esm2_t30_150M_UR50D_embeddings_smote.npy
```

### 3. Confusion matrix images  
```
output/esm2_t30_150M_UR50D_confusion_matrix.png
```

### 4. Saved model weights  
```
output/esm2_t30_150M_UR50D_state.pth
output/esm2_t30_150M_UR50D_model_full.pth
```

---

## Comparing Models

You can compare:

| Model | SMOTE? | Expected Behavior |
|-------|--------|------------------|
| **8M** | No | Fastest, lowest accuracy |
| **35M** | No | Good baseline |
| **150M** | No | Very strong embeddings |
| **150M + SMOTE** | Yes | Avoids bias if dataset is imbalanced |

Our experiments show:

- **Non-SMOTE embeddings** vary normally across model sizes  
- **SMOTE embeddings** should produce better CV and Jackknife values, due to modified sequence distribution  

---

## Citation

If you use this pipeline in research, cite: 
https://www.sciencedirect.com/science/article/abs/pii/S0933365717305171?fr=RR-2&ref=pdf_download&rr=9abb75db4b3ee0e0


```
ESM-2: Evolutionary Scale Modeling using Transformations
Facebook AI Research https://huggingface.co/facebook/esm2_t6_8M_UR50D

```

---

