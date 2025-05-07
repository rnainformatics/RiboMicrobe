import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
import argparse
from torch.nn.utils.rnn import pad_sequence
import os
import json
import datetime

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm,
                                                attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)
        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights


class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, record_attn=False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
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
    def __init__(self, hidden=64, layers=4, record_attn=False, input_dim=64):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout=0.1)
        encoder_layers = TransformerLayer(hidden,
                                          nhead=2,
                                          dropout=0.1,
                                          dim_feedforward=128,
                                          batch_first=True)
        self.module = TransformerEncoder(encoder_layers, layers, record_attn=record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output
# 假设这里是你已经定义的模型类
class TransModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=64, record_attn=False):
        super(TransModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = AttnModule(hidden=mid_hidden, record_attn=record_attn)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.Linear1 = nn.Linear(in_features=64, out_features=2)
        self.record_attn = record_attn
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous().float()
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = x.transpose(1, 2).contiguous().float()
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = nn.AdaptiveAvgPool1d(1)(x)
        x = x.squeeze(-1)
        x = self.Linear1(x)
        x = F.relu(x)
        if self.record_attn:
            return x, attn_weights
        else:
            return x


# 保存模型
#def save_model(model, model_path):
    #torch.save(model.state_dict(), model_path)
    #print(f"Model saved at {model_path}")


# 加载模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# 预测功能
def predict(model, data_loader, device):
    model.to('cpu')
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in data_loader:
            X, _ = batch  
            X = X.to('cpu', dtype=torch.float32)  
            y_hat = model(X)
            probs = F.softmax(y_hat, dim=1)  # 获得概率
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取 TIS-seq 类的概率值
    return all_preds, all_probs

def generate_file_names(fasta_file, jobid, resultdir, jsondir):
    base_filename = os.path.basename(fasta_file).split(".")[0]
    return {
        "predictions_file": os.path.join(resultdir, f"{jobid}_predictions.json"), 
        "start_codon_file": os.path.join(resultdir, f"{jobid}_start_codon.json"), 
        "sORF_file": os.path.join(resultdir, f"{jobid}_sORFs.json"),  
        "log_file": os.path.join(jsondir, f"{jobid}_log.json")
    }

# 编码DNA序列为独热编码
def encode(sequence):
    base_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    sequence = sequence.upper()
    return [base_dict.get(base, [0, 0, 0, 0]) for base in sequence]

def extract_start_codons(sequence, upstream=50, downstream=20):
    start_codons = ['ATG', 'GTG', 'TTG']
    fragments = []
    fragment_length = upstream + 3 + downstream
    for i in range(len(sequence) - 3):
        codon = sequence[i:i+3]
        if codon in start_codons:
            start_pos = i
            end_pos = start_pos + 3 + downstream
            fragment = sequence[max(0, start_pos - upstream):end_pos]

            if len(fragment) < fragment_length:
                fragment = fragment.ljust(fragment_length, 'N')  # 用 'N' 填充到固定长度

            fragments.append(fragment)
    return fragments


# 加载FASTA文件并提取DNA序列
def load_fasta(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


# 处理FASTA文件并生成数据加载器
def prepare_data(fasta_file, model, device, batch_size=256, file_names=None):
    base_filename = os.path.splitext(os.path.basename(fasta_file))[0]

    sequences = load_fasta(fasta_file)
    all_fragments = []
    fragment_with_metadata = []  # 现在存储字典而非元组

    for seq_id, sequence in sequences.items():
        fragments = extract_start_codons(sequence)
        for frag in fragments:
            if 'N' in frag:
                continue
            start_pos = sequence.find(frag)
            end_pos = start_pos + len(frag) - 1
            all_fragments.append(frag)
            # 修改为字典存储
            fragment_with_metadata.append({
                "seq_id": seq_id,
                "fragment": frag,
                "start_pos": start_pos,
                "end_pos": end_pos
            })

    if len(all_fragments) == 0:
        raise ValueError("No valid fragments extracted from the input FASTA file. Check the sequences or parameters.")

    # 对每个片段进行编码并转换为模型输入
    encoded_fragments = [torch.tensor(encode(frag), dtype=torch.float32) for frag in all_fragments]
    padded_fragments = pad_sequence(encoded_fragments, batch_first=True)
    X = padded_fragments
    y = torch.zeros(len(X))

    # 创建 DataLoader
    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    predictions, probs = predict(model, test_loader, device)

    predictions_data = []
    for meta, pred, prob in zip(fragment_with_metadata, predictions, probs):
        if pred == 1:
            predictions_data.append({
                "Sequence_ID": meta["seq_id"],
                "Fragment": meta["fragment"],
                "Start_Pos": int(meta["start_pos"]),
                "End_Pos": int(meta["end_pos"]),
                "TIS-seq": int(pred),
                "Probability": float(prob)
            })
    predictions_file = file_names["predictions_file"]
    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)

    return predictions_file

def generate_start_codon_positions(predictions_file, fasta_file, output_file):
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # 处理起始密码子
    start_codon_data = []
    for entry in predictions:
        frag = entry["Fragment"]
        if len(frag) >= 53:
            start_codon_info = {
                "Sequence_ID": entry["Sequence_ID"],
                "Start_Codon": frag[50:53],
                "Position": entry["Start_Pos"] + 50,
                "Probability": entry["Probability"],
                "Genomic_Context": {
                "Upstream_50bp": frag[:50],
                "Start_Codon": frag[50:53],
                "Downstream_20bp": frag[53:73]
                }
            }
            start_codon_data.append(start_codon_info)

    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(start_codon_data, f, indent=2) 
    
    return output_file

def find_sORFs_with_positions(fasta_file, start_codon_file, output_file):
    with open(start_codon_file, 'r') as f:
        start_data = json.load(f)
    
    sORFs_data = []
    sequences = load_fasta(fasta_file)
    stop_codons = ['TAA', 'TGA', 'TAG']

    for entry in start_data:
        seq_id = entry["Sequence_ID"]
        if seq_id not in sequences:
            continue

        start_pos = entry["Position"] - 50 
        sequence = sequences[seq_id]
        downstream_seq = sequence[start_pos + 3 : ]

        for j in range(0, len(downstream_seq)-2, 3):
            codon = downstream_seq[j:j+3]
            if codon in stop_codons:
                sORF_info = {
                    "sORF_ID": f"{seq_id}_sORF_{len(sORFs_data)+1}",
                    "Start": start_pos,                   
                    "End": start_pos + 3 + j + 2,         
                    "Start_Codon": entry["Start_Codon"],    
                    "Stop_Codon": codon,                   
                    "ORF_Length": (j + 3) * 3,            
                    "Sequence": sequence[start_pos : start_pos+3+j+3],  
                    "Probability": entry["Probability"],   
                    "Reading_Frame": (start_pos % 3) + 1
                }
                sORFs_data.append(sORF_info)
                break

    with open(output_file, 'w') as f:
        json.dump(sORFs_data, f, indent=2)

    return output_file


# 命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Predict TIS-seq from FASTA")
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--jobid', type=str, required=True)  
    parser.add_argument('--resultdir', type=str, required=True)
    parser.add_argument('--jsondir', type=str, required=True)  
    return parser.parse_args()


# 主函数
def main():
    args = parse_args()
    
    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.jsondir, exist_ok=True)
    file_names = generate_file_names(
        args.fasta_file, 
        args.jobid,
        args.resultdir,
        args.jsondir
    )
    # 加载模型
    model = TransModel(num_genomic_features=4)
    model = load_model(model, args.model_path)

    # 使用模型进行预测
    device = torch.device('cpu')
    predictions_file = prepare_data(args.fasta_file, model, device, 
                                      args.batch_size, file_names)
    start_codon_file = generate_start_codon_positions(
        predictions_file, 
        args.fasta_file,
        file_names["start_codon_file"]
    )
    sORF_file = find_sORFs_with_positions(
        args.fasta_file,
        start_codon_file,
        file_names["sORF_file"]
    )
        
    with open(file_names["log_file"], 'w') as f:
        json.dump({"finished": 100}, f)

if __name__ == '__main__':
    main()




#python script.py --fasta_file /path/to/test_data.fasta --model_path /path/to/saved_model.pth

