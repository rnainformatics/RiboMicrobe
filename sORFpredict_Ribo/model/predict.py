import torch
import numpy as np
import pysam
from Bio import SeqIO
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from ecoli_finetune import FineTuneModel
from itertools import product

class ORFPredictor:
    def __init__(self, model_path, max_seq_len=1024, device='cuda'):
        self.device = device
        self.max_seq_len = max_seq_len
        self.model = self._load_model(model_path).to(device).eval()
        
        # 遗传密码参数设置
        self.start_codons = {'ATG', 'GTG', 'TTG', 'CTG'}
        self.stop_codons = {'TAA', 'TAG', 'TGA'}
        self.min_orf_length = 18
        self.max_orf_length = max_seq_len * 3  # 根据模型输入长度调整

    def _load_model(self, model_path):
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location='cpu')
        model = FineTuneModel(pretrained_path=None, freeze_backbone=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _find_orfs(self, seq_record):
        """在单条序列中查找所有ORF"""
        orfs = []
        seq = str(seq_record.seq).upper()
        seq_len = len(seq)
        reverse_seq = seq.translate(str.maketrans('ATCG', 'TAGC'))[::-1]

        for strand, nucleotide_seq in [('+', seq), ('-', reverse_seq)]:
            for frame in range(3):
                trans_seq = nucleotide_seq[frame:]
                codons = [trans_seq[i:i+3] for i in range(0, len(trans_seq)-2, 3)]

                in_orf = False
                start_pos = 0
                for i, codon in enumerate(codons):
                    pos = frame + i*3
                    if codon in self.start_codons and not in_orf:
                        in_orf = True
                        start_pos = pos
                    elif codon in self.stop_codons and in_orf:
                        end_pos = pos + 3
                        orf_length = end_pos - start_pos
                        if orf_length >= self.min_orf_length:
                            if strand == '+':
                                genome_start = start_pos
                                genome_end = end_pos
                            else:
                                genome_start = seq_len - end_pos
                                genome_end = seq_len - start_pos
                                
                            orf_seq = nucleotide_seq[start_pos:end_pos]
                            orfs.append({
                                'chrom': seq_record.id,
                                'start': genome_start,
                                'end': genome_end,
                                'strand': strand,
                                'frame': frame+1,
                                'sequence': orf_seq,
                                'length': orf_length
                            })
                        in_orf = False
        return orfs

    def _encode_sequence(self, seq):
        """将DNA序列编码为one-hot格式"""
        seq = seq.upper()
        mapping = {'A':0, 'T':1, 'C':2, 'G':3}
        one_hot = np.zeros((len(seq), 4), dtype=np.float32)
        
        for i, base in enumerate(seq):
            if base in mapping:
                one_hot[i, mapping[base]] = 1.0
            else:  # 处理未知碱基
                one_hot[i] = 0.25  # 均匀分布
        return one_hot

    def _get_ribo_coverage(self, bam_file, chrom, start, end, strand):
        """获取链特异性覆盖数据"""
        coverage = np.zeros(end - start, dtype=np.float32)
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            if chrom not in bam.references:
               print(f"Warning: Chromosome {chrom} not found in BAM file. Skipping...")
               return coverage 
            
            for read in bam.fetch(chrom, start, end):
                # 链特异性过滤
                if (strand == '+' and read.is_reverse) or (strand == '-' and not read.is_reverse):
                    continue
                
                positions = read.get_reference_positions()
                for pos in positions:
                    if start <= pos < end:
                        coverage[pos - start] += 1.0

        # 标准化处理
        coverage += 1e-6
        return np.log1p(coverage) / np.log1p(coverage.max())

    def _process_orf(self, orf_info, bam_path):
        """处理单个ORF"""
        # 序列编码
        seq_data = self._encode_sequence(orf_info['sequence'])
        
        # 获取ribo覆盖
        ribo_data = self._get_ribo_coverage(
            bam_path, 
            orf_info['chrom'], 
            orf_info['start'], 
            orf_info['end'],
            orf_info['strand']
        )

        # 填充/截断处理
        seq_len = seq_data.shape[0]
        if seq_len > self.max_seq_len:
            seq_data = seq_data[:self.max_seq_len]
            ribo_data = ribo_data[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - seq_len
            seq_data = np.pad(seq_data, ((0,pad_len),(0,0)), mode='constant')
            ribo_data = np.pad(ribo_data, (0,pad_len), mode='constant')

        return (
            torch.from_numpy(seq_data.T).float(),  # [4, L]
            torch.from_numpy(ribo_data).float(),   # [L]
            orf_info  # 保留ORF元数据
        )

    def predict_genome(self, fasta_path, bam_path, output_file, batch_size=32):
        """全基因组预测流程"""
        # 第一步：提取所有ORF
        all_orfs = []
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            valid_chroms = set(bam.references)

        for seq_record in SeqIO.parse(fasta_path, "fasta"):
            print(f"Processing {seq_record.id}...")
            if seq_record.id not in valid_chroms:
                print(f"Warning: Chromosome {seq_record.id} not in BAM, skipping...")
                continue
            orfs = self._find_orfs(seq_record)
            all_orfs.extend(orfs)
            print(f"Found {len(orfs)} ORFs in {seq_record.id}")

        # 第二步：构建数据集
        class GenomeDataset(Dataset):
            def __init__(self, parent, orfs):
                self.parent = parent
                self.orfs = orfs
                self.bam_path = bam_path
                
            def __len__(self):
                return len(self.orfs)
            
            def __getitem__(self, idx):
                return self.parent._process_orf(self.orfs[idx], self.bam_path)

        dataset = GenomeDataset(self, all_orfs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 第三步：批量预测
        predictions = []
        with torch.no_grad():
            for batch in loader:
                seq_inputs, ribo_inputs, orf_infos = batch
                seq_inputs = seq_inputs.to(self.device)
                ribo_inputs = ribo_inputs.to(self.device)
                
                outputs = self.model(seq_inputs, ribo_inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # 关联预测结果与ORF信息
                for i, prob in enumerate(probs):
                    info = {k: v[i].item() if isinstance(v, torch.Tensor) else v[i] 
                           for k, v in orf_infos.items()}
                    info['probability'] = float(prob)
                    info['prediction'] = 1 if prob >= 0.5 else 0
                    predictions.append(info)

        # 第四步：保存结果
        with open(output_file, 'w') as f:
            f.write("Chrom\tStart\tEnd\tStrand\tFrame\tLength\tProbability\tPrediction\n")
            for pred in predictions:
                f.write(f"{pred['chrom']}\t{pred['start']}\t{pred['end']}\t"
                       f"{pred['strand']}\t{pred['frame']}\t{pred['length']}\t"
                       f"{pred['probability']:.4f}\t{pred['prediction']}\n")
        print(f"Predicted {len(predictions)} ORFs, saved to {output_file}")

        base_path = os.path.splitext(output_file)[0]
        sorf_output = f"{base_path}_sORF.csv"

        sorf_predictions = [
            p for p in predictions 
            if 18 < p['length'] < 300 and p['prediction'] == 1
        ]

        with open(sorf_output, 'w') as f:
            f.write("Chrom\tStart\tEnd\tStrand\tFrame\tLength\tProbability\tPrediction\n")
            for pred in sorf_predictions:
                f.write(f"{pred['chrom']}\t{pred['start']}\t{pred['end']}\t"
                       f"{pred['strand']}\t{pred['frame']}\t{pred['length']}\t"
                       f"{pred['probability']:.4f}\t{pred['prediction']}\n")
        print(f"Filtered {len(sorf_predictions)} sORFs, saved to {sorf_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genome-wide sORF Prediction')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--fasta', required=True, help='Genome FASTA file')
    parser.add_argument('--bam', required=True, help='Ribo-seq BAM file')
    parser.add_argument('--output', default='predictions.tsv', help='Output file')
    parser.add_argument('--max_len', type=int, default=1024, help='Max ORF length')
    args = parser.parse_args()

    predictor = ORFPredictor(
        model_path=args.model,
        max_seq_len=args.max_len,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    predictor.predict_genome(args.fasta, args.bam, args.output)

#python predict.py --model ./best_model.pth --fasta ./samples/eco/genome.dna.fa --bam ./samples/eco/ribo_bams/SRX505573.bam --output ./predictions.csv --max_len 1024
