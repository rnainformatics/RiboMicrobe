import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from torch.nn.utils.rnn import pad_sequence
import os
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 模型模块 ----------
class TransformerLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm,
                                                attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, record_attn=False):
        super().__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weight_list = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)
        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden=64, layers=4, record_attn=False):
        super().__init__()
        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout=0.1)
        encoder_layers = TransformerLayer(hidden, nhead=2, dropout=0.1, dim_feedforward=128, batch_first=True)
        self.module = TransformerEncoder(encoder_layers, layers, record_attn=record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        return self.module(x)

class TransModel(nn.Module):
    def __init__(self, num_genomic_features=5, mid_hidden=64, record_attn=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_genomic_features, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = AttnModule(hidden=mid_hidden, record_attn=record_attn)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.Linear1 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(1, 2).float()
        x = self.conv1(x)
        x = x.transpose(1, 2).float()
        x = self.attn(x)
        x = x.transpose(1, 2).float()
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = nn.AdaptiveAvgPool1d(1)(x).squeeze(-1)
        x = self.Linear1(x)
        x = F.relu(x)
        return x

# ---------- 数据处理模块 ----------
def encode(sequence, strand='+'):
    base_dict = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1], 'N':[0,0,0,0]}
    strand_feature = [1] if strand == '+' else [0]
    return [base_dict.get(base.upper(), [0,0,0,0]) + strand_feature for base in sequence]

def load_fasta(path):
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(path, "fasta")}

def extract_start_codons(sequence, upstream=50, downstream=20):
    start_codons = ['ATG', 'GTG', 'TTG']
    fragments = []
    for i in range(len(sequence) - 3):
        codon = sequence[i:i+3]
        if codon in start_codons:
            start = max(0, i - upstream)
            frag = sequence[start: i + 3 + downstream]
            frag = frag.ljust(upstream + 3 + downstream, 'N')
            fragments.append((frag, start))
    return fragments

def predict(model, loader):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            X, _ = batch
            X = X.float().to(device)
            out = model(X)
            p = F.softmax(out, dim=1)
            preds.extend(torch.argmax(p, dim=1).tolist())
            probs.extend(p[:,1].tolist())
    return preds, probs

def prepare_data(fasta_file, model, batch_size=256, out_path="predictions.txt"):
    seqs = load_fasta(fasta_file)
    all_data = []
    meta = []
    for sid, seq in seqs.items():
        for strand_seq, strand in [(seq, '+'), (str(Seq(seq).reverse_complement()), '-')]:
            for frag, start in extract_start_codons(strand_seq):
                if 'N' in frag:
                    continue
                encoded = encode(frag, strand)
                all_data.append(torch.tensor(encoded))
                meta.append({
                    "Sequence_ID": sid,
                    "Fragment": frag,
                    "Start_Pos": start,
                    "End_Pos": start + len(frag) - 1,
                    "Strand": strand
                })
    if not all_data:
        raise ValueError("没有提取到任何有效片段")
    padded = pad_sequence(all_data, batch_first=True)
    loader = DataLoader(TensorDataset(padded, torch.zeros(len(padded))), batch_size=batch_size)
    preds, probs = predict(model, loader)
    with open(out_path, 'w') as f:
        f.write("Sequence_ID\tFragment\tStart_Pos\tEnd_Pos\tStrand\tTIS-seq\tProbability\n")
        for m, p, prob in zip(meta, preds, probs):
            if p == 1:
                f.write(f"{m['Sequence_ID']}\t{m['Fragment']}\t{m['Start_Pos']}\t{m['End_Pos']}\t{m['Strand']}\t1\t{prob:.4f}\n")
    return out_path

def generate_start_codons(predictions_file, out_file):
    results = []
    with open(predictions_file) as f:
        next(f)
        for line in f:
            sid, frag, spos, epos, strand, _, prob = line.strip().split("\t")
            results.append((sid, frag, int(spos), strand, float(prob)))
    with open(out_file, 'w') as f:
        f.write("Sequence_ID\tStart_Codon\tPosition\tStrand\tProbability\tUpstream\n")
        for sid, frag, spos, strand, prob in results:
            if len(frag) >= 73:
                start_codon = frag[50:53]
                f.write(f"{sid}\t{start_codon}\t{spos+50}\t{strand}\t{prob:.4f}\t{frag[:50]}\t{frag[53:73]}\n")
    return out_file

def find_sORFs(fasta_file, codon_file, out_file):
    seqs = load_fasta(fasta_file)
    stop_codons = ['TAA', 'TGA', 'TAG']
    orf_dict = {}

    with open(codon_file) as f:
        next(f)
        for line in f:
            sid, scodon, pos, strand, prob, *_ = line.strip().split("\t")
            pos = int(pos)
            prob = float(prob)
            original_seq = seqs.get(sid)
            if not original_seq:
                continue
            seq_len = len(original_seq)

            if strand == '+':
                seq = original_seq
                start = pos
            else:
                seq = str(Seq(original_seq).reverse_complement())
                start = pos  # pos 是反向互补后的坐标

            downstream = seq[start + 3:]

            for i in range(0, len(downstream) - 2, 3):
                codon = downstream[i:i + 3]
                if codon in stop_codons:
                    end = start + 3 + i + 3  
                    if strand == '+':
                        orf_seq = seq[start:end]
                        converted_start = start
                        converted_end = end
                    else:
                        converted_start = seq_len - end
                        converted_end = seq_len - start 
                        orf_seq = str(Seq(seq[start:end]).reverse_complement())
                    if len(orf_seq) >= 18:
                        key = (sid, strand, converted_end)
                        orf_tuple = (
                            sid, converted_start, converted_end,
                            scodon, codon, len(orf_seq),
                            strand, (converted_start % 3),
                            orf_seq, prob
                        )
                        if key not in orf_dict or len(orf_seq) > orf_dict[key][5]:
                            orf_dict[key] = orf_tuple
                    break

    sorf_out = re.sub(r'_ORFs\.txt$', '_sORFs.txt', out_file)
    with open(out_file, 'w') as f_all, open(sorf_out, 'w') as f_sorf:
        header = "sORF_ID\tSequence_ID\tStart\tEnd\tStart_Codon\tStop_Codon\tORF_Length\tStrand\tReading_Frame\tSequence\tProbability\n"
        f_all.write(header)
        f_sorf.write(header)
        for idx, s in enumerate(orf_dict.values(), 1):
            line = f"sORF_{idx}\t" + "\t".join(map(str, s)) + "\n"
            f_all.write(line)
            if s[5] < 300:
                f_sorf.write(line)
    return out_file

# ---------- 主函数 ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--resultdir', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    fasta_basename = os.path.basename(args.fasta_file)
    prefix = re.sub(r'_.*', '', fasta_basename)

    pred_file = os.path.join(args.resultdir, f"{prefix}_predictions.txt")
    start_file = os.path.join(args.resultdir, f"{prefix}_start_codons.txt")
    sorf_file = os.path.join(args.resultdir, f"{prefix}_ORFs.txt")

    model = TransModel(num_genomic_features=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    prepare_data(args.fasta_file, model, out_path=pred_file)
    generate_start_codons(pred_file, start_file)
    find_sORFs(args.fasta_file, start_file, sorf_file)

if __name__ == "__main__":
    main()

