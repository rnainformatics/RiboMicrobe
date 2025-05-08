import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from d2l import torch as d2l

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
    def __init__(self, hidden, dropout = 0.1, max_len = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden = 64, layers = 4, record_attn = False, inpu_dim = 64): 
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
        encoder_layers = TransformerLayer(hidden, 
                                          nhead = 2,
                                          dropout = 0.1,
                                          dim_feedforward = 128,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers, 
                                         layers, 
                                         record_attn = record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)
    
class TransModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=64, record_attn=False):
        super(TransModel, self).__init__()
        print('Initializing TransModel')
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
        x = nn.AdaptiveAvgPool1d(1)(x)  # 全局池化 (batch_size, 1, 1)
        x = x.squeeze(-1)  # 移除多余维度 (batch_size, feature_dim)
        #print("After squeeze:", x.shape)
        x = self.Linear1(x)
        x = F.relu(x)
        if self.record_attn:
            return x, attn_weights
        else:
            return x

# 构建模型
# 创建 TransModel 实例
trans_model = TransModel(num_genomic_features=4)
# 输入数据
X = torch.rand(size=(256, 73, 4))  # (batch_size, length, feature_dim)，根据TransModel要求
# 直接调用模型进行前向传播
output = trans_model(X)

# ==== 数据处理 ====
def encode(sequence):
    base_dict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    return [base_dict.get(base, [0,0,0,0]) for base in sequence.upper()]

def load_dataset(data_path):
    df = pd.read_csv(data_path, sep="\t", header=0)
    data_dict = df.set_index('sequence')['label'].to_dict()
    encoded_data = [(encode(seq), label) for seq, label in data_dict.items()]
    X = torch.tensor([item[0] for item in encoded_data], dtype=torch.float32)
    y = torch.tensor([item[1] for item in encoded_data], dtype=torch.long)
    return TensorDataset(X, y)

# ==== 模型评估 ====
def evaluate_metrics(net, data_iter, device):
    net.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X).argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(y_hat.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    return {
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'accuracy': accuracy_score(all_labels, all_preds)
    }

def compute_auc(net, data_iter, device):
    net.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            probs = F.softmax(net(X), dim=1)[:, 1]
            all_labels.append(y)
            all_probs.append(probs)
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()
    return roc_auc_score(all_labels, all_probs)

# ==== 训练主函数 ====
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.data_path)

    # 数据划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.01 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = TransModel(num_genomic_features=4)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_metrics(model, val_loader, device)
        val_auc = compute_auc(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")

# ==== 参数入口 ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset txt file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
