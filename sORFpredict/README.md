# sORFpredict

`sORFpredict` is a deep learning-based tool for predicting bacterial small open reading frames (sORFs) using sequence data. It uses CNN and Transformer architectures, and is implemented in Python with PyTorch.

## Requirements
- Python ≥ 3.7  
- PyTorch ≥ 1.8  
- numpy  
- pandas  
- h5py

## Installation

Clone the repository and install the required packages:

```
git clone https://github.com/rnainformatics/RiboMicrobe.git
cd RiboMicrobe
```

Alternatively, you can use the provided conda environment:

```
conda env create -f environment.yml
conda activate sORFpredict
```

## Usage

### 1. Train the model

Prepare your dataset (`dataset_anno.txt`) and run:

```
python Training.py \
  --data_path /path/to/dataset_anno.txt \
  --save_path /path/to/trans_model.pth \
  --batch_size 256 \
  --lr 0.01 \
  --epochs 100
```

### 2. Predict using a trained model

Provide a FASTA file with candidate sequences and run:

```
python model_anno.py \
  --fasta_file /path/to/test_data.fasta \
  --model_path /path/to/trans_model.pth
```
