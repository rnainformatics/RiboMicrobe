# RiboMicrobe

## Introduction
sORFpredict and sORFpredict_Ribo are deep learning models designed for predicting bacterial sORFs. Two models are written in python 3 and uses the PyTorch library for deep learning purposes. Fine-tuned models are currently available for Escherichia coli, Bacillus subtilis, Staphylococcus aureus, and Salmonella enterica, which can be directly used for training. For other species, users can download the scripts and train custom models. It is recommended to use GPU-base infrastructure for training and using new models with sORFpredict and sORFpredict_Ribo.
## Installation
To use sORFpredict and sORFpredict_Ribo, simply clone the repository and install the required Python libraries:
```
git clone https://github.com/rnainformatics/RiboMicrobe.git
```
These tools are developed for Python ≥ 3.7 and rely on the following packages:
```
numpy

pandas

h5py

PyTorch

samtools
```
Install PyTorch from https://pytorch.org, ensuring compatibility with your system's hardware. This package has been developed and tested with PyTorch ≥ 1.8.
## sORFpredict
sORFpredict applies convolutional neural networks (CNNs) and a Transformer architecture for deep learning.
<div align="center">
  <img src="https://github.com/user-attachments/assets/79d87516-80fc-4204-9c95-83cf2ad0e04d" width="80%">
</div>

## Data Preparation
The training dataset for sORFpredict consists of initiation codon fragments from 36 bacterial species. Sequence fragments within a 50nt upstream and 20nt downstream range of the start codons (ATG, GTG, TTG) are extracted as the positive set, while sequences with the CTG start codon of the same length are used as the negative set. The data is shuffled and then divided into training, validation, and test sets.
Training Model
```
Python Training.py --data_path /path/to/dataset_anno.txt --save_path /path/to/trans_model.pth --batch_size 256 --lr 0.01 --epochs 100
--data_path   help='Path to dataset txt file'
--save_path   help='Path to save the model'
 --batch_size   default=256
 --lr   default=0.01
--epochs   default=100
```
Prediction Model
```
python model_anno.py --fasta_file /path/to/test_data.fasta --model_path /path/to/saved_model.pth
```

## sORFpredict_Ribo
sORFpredict_Ribo is a deep learning model that includes components such as CNN, Residual Blocks, Transformer encoder, and upsampling layers, combining sequence features and Ribo-seq signal features for prediction. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/7ebea4ed-aaf4-493a-bdfd-e74e13f4203e" width="80%">
</div>

## Data Preparation:
The training dataset of sORFpredict_Ribo was derived from 19 species. For each species, sequence and coverage features of CDS regions were extracted from the reference genome, annotation file, and Ribo-seq BAM files, and then stored in a structured HDF5 file for downstream model pretraining.
```
python ./Dataparse.py --species_dirs bsu clj cvi eco eli hsa hvo msm mtu pae sau sav scl sen sgr sli sve syn zmo --output ./pretrain_data.h5 --threads 4
--species_dirs   help="Path to species directories"
--output    help="Output HDF5 filename"
--threads   help="Number of threads to use"
```
The parsed data for each species should be contained within a single folder：
```
e.coli/
├── genome.fasta         # Reference genome file
├── annotation.gff       # Annotation file
└── ribo_bams/           # BAM file
    ├── sample1.bam
    ├── sample2.bam
    ...
    └── samplen.bam
```

## Pre-training Model：
```
python pretrain.py --h5_path path/to/pretrain_data.h5 --max_seq_len 1024 --batch_size 64 --lr 1e-4 --epochs 100 --seq_weight 0.3 --ribo_weight 0.7 --weight_decay 0.01 --val_species_ratio 0.4
--h5_path
--max_seq_len   default=1024
--batch_size    default=64
--lr   default=1e-4
--epochs   default=100
--seq_weight   default=0.3
--ribo_weight   default=0.7
 --weight_decay   default=0.01
--val_species_ratio    default=0.4
```
## Fine-tuned Model：
If a candidate ORF satisfies both the sequence and Ribo-seq signal criteria, it is classified as positive, forming a binary classification task. We primarily fine-tuned the model using data from Escherichia coli, Staphylococcus aureus, Bacillus subtilis, and Salmonella enterica.
```
python ecoli_finetune.py –genome /path/to/genome.dna.fa --annotation /path/to/annotation.gtf --ribo_bam /path/to/sample.bam --pretrained best_model.pt --output_h5 /path/to/species.h5 --output_dir /path/to/output --max_seq_len 1024 --batch_size 32 --lr 1e-4
--genome 
 --annotation 
--ribo_bam 
--pretrained 
 --output_h5   default="processed_data.h5"
--output_dir
 --max_seq_len   default=1024
--batch_size   default=32
 --lr   default=1e-4
--epochs    default=100
--patience   help='Number of epochs to wait before early stopping'
```

## Prediction Model：
```
python predict.py --model best_model.pth --fasta /path/to/genome.dna.fa –bam /path/to/sample.bam --output ./predictions.csv --max_len 1024
--model   help='Path to trained model
--fasta   help='Genome FASTA file
--bam   help='Ribo-seq BAM file
--output   default='predictions.tsv', help='Output file
--max_len    default=1024, help='Max ORF length
```
sORFpredict_Ribo generates two result files: one contains all prediction results, including both ORFs and non-ORFs; the other contains predicted sORFs with lengths less than 300 nucleotides.
