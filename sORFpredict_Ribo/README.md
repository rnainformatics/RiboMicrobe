# sORFpredict_Ribo

`sORFpredict_Ribo` is an advanced deep learning model for predicting bacterial sORFs using both sequence data and Ribo-seq signals. It combines CNN, residual blocks, Transformer encoders, and upsampling layers to improve prediction accuracy.

## Requirements

- Python ≥ 3.7  
- PyTorch ≥ 1.8  
- numpy  
- pandas  
- h5py  
- samtools (for BAM file processing)

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/rnainformatics/RiboMicrobe.git
cd RiboMicrobe
pip install -r requirements.txt
```

Or use the provided conda environment:

```
conda env create -f environment.yml
conda activate sORFpredict
```

## Usage

### 1. Prepare training data

Organize each species folder as follows:

```
species_name/
├── genome.fasta
├── annotation.gff
└── ribo_bams/
    ├── sample1.bam
    └── sample2.bam
```

Run:

```
python Dataparse.py \
  --species_dirs eco sau bsu sen ... \
  --output pretrain_data.h5 \
  --threads 4
```

### 2. Pre-train the model

```
python pretrain.py \
  --h5_path pretrain_data.h5 \
  --max_seq_len 1024 \
  --batch_size 64 \
  --lr 1e-4 \
  --epochs 100 \
  --seq_weight 0.3 \
  --ribo_weight 0.7 \
  --weight_decay 0.01 \
  --val_species_ratio 0.4
```

### 3. Fine-tune the model (e.g., on E. coli)

```
python ecoli_finetune.py \
  --genome /path/to/genome.fa \
  --annotation /path/to/annotation.gtf \
  --ribo_bam /path/to/sample.bam \
  --pretrained best_model.pt \
  --output_h5 ecoli_data.h5 \
  --output_dir ./results \
  --max_seq_len 1024 \
  --batch_size 32 \
  --lr 1e-4
```

### 4. Run prediction

```
python predict.py \
  --model best_model.pth \
  --fasta /path/to/genome.fa \
  --bam /path/to/sample.bam \
  --output predictions.csv \
  --max_len 1024
```

Output includes:
- A full prediction result file
- A filtered sORF list (<300 nt)
