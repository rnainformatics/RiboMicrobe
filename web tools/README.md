  # Diffco & DiffTE

## Introduction

**Diffco** (Differential Codon Occupancy) and **DiffTE** (Differential Translation Efficiency) are two analysis modules designed for bacterial Ribo-seq and RNA-seq data.  
They allow researchers to explore translational regulation mechanisms under different experimental conditions.  

We provide both **an online platform** and **local scripts** for running these analyses.  

> **Recommended:** Use our online platform: [RiboMicrobe Online Tool](https://rnainformatics.org.cn/RiboMicrobe/index.php)

---

## Online Tool Usage

1. Visit the [RiboMicrobe website](https://rnainformatics.org.cn/RiboMicrobe/index.php)  
2. Select the desired tool from the **Tools** menu  
3. Upload your data or use the example dataset  
4. Set analysis parameters  
5. Submit the analysis job  
6. View and download results  

For more details, see the **Help** page on our website or contact us.

---

## Local Installation

We provide scripts for running Diffco and DiffTE locally, suitable for custom analysis workflows.

### Environment Requirements

We recommend **R ≥ 4.1.0** with **Bioconductor** pre-installed.  
Required R packages:

GenomicAlignments, GenomicFeatures, data.table, ggplot2, jsonlite,
gridExtra, seqinr, cowplot, zoo, signal, parallel, plyr, Rsamtools,
getopt, ggpubr, reshape2, dplyr, ggthemes, RColorBrewer, ComplexHeatmap,
Cairo, circlize, ggplotify

## DiffTE: Differential Translation Efficiency Analysis

### Method Overview
- Normalize Ribo-seq and RNA-seq raw counts separately using **TMM** from `edgeR`
- Calculate **RPKM**
- Define **Translation Efficiency (TE)** as:  
  `TE = Ribo-seq RPKM / RNA-seq RPKM`
- Perform differential analysis on log₂(TE) using `limma` for linear modeling

### Significance Criteria
```

|log₂FC| ≥ 1.5 and p-value < 0.05 (uncorrected)

```

### Usage
```

Rscript diffTE.r -j <jobid> -s \<species\_dir> -n <samplenames> -fc <foldchange> -p <pvalue>

```

### Parameters
- `-j` : Job ID  
- `-s` : Path to species directory containing required files  
- `-n` : Comma-separated sample names  
- `-fc`: Fold change threshold (default: 1.5)  
- `-p` : p-value threshold (default: 0.05)  

### Notes
- RPKM calculation logic can be found in `RPKM.r`.

---

##  Diffco: Differential Codon Occupancy Analysis

### Method Overview
- Perform in-frame filtering of Ribo-seq reads within CDS regions  
- Remove first and last 15 codons to reduce edge effects  
- Calculate footprint coverage for A/P/E sites and downstream (+1 ~ +3) codons  
- Normalize by average downstream coverage  
- Convert codon triplets to amino acid occupancy levels  
- Perform statistical analysis using `limma` with empirical Bayes


### Step 1: Calculate Codon Usage
```

Rscript usage_codon.R -s \<species\_dir>

```

### Step 2: Analyze Differential Codon Occupancy
```

Rscript diffcodon\_occupancy.r -j <jobid> -s \<species\_dir> -n <samplenames> -i <bia> -o \<offset\_position> -r <foldchange> -p <pvalue>

```

### Parameters
- `-j` : Job ID  
- `-s` : Path to species directory containing required files  
- `-n` : Comma-separated sample names  
- `-i` : BIA file  
- `-o` : Offset position  
- `-r` : Fold change threshold (default: 1.5)  
- `-p` : p-value threshold (default: 0.05)  

---

## Contact
- **Online platform:** https://rnainformatics.org.cn/RiboMicrobe/index.php  
- **Help documentation:** Available on the website  
- **Email:** Please contact us via the official website
```
