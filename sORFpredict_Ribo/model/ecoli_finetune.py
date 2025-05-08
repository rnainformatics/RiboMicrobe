import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,auc,roc_curve,confusion_matrix
)
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler ,Subset
from Dataparse import DataProcessor
from test_pretrain import MultiModalModel
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch import optim
from collections import defaultdict
import sys
import os
from datetime import datetime

class H5Dataset(Dataset):
    def __init__(self, h5_path, max_seq_len=1024, is_training=True):
        self.h5_path = h5_path
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        with h5py.File(h5_path, 'r') as hf:
            self.sample_keys = [k for k in hf.keys() if k.startswith('sample_')]
            self.transcript_ids = [hf[k].attrs['transcript_id'] for k in self.sample_keys]
            self.n_samples = len(self.sample_keys)

        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            sample = hf[self.sample_keys[idx]]
            raw_data = sample['data'][:]  # Shape: [L,5]
            label = sample.attrs['label']
            
            if self.is_training:
                if np.random.rand() < 0.5:
                # 随机替换部分序列
                    replace_len = np.random.randint(200, 500)
                    replace_start = np.random.randint(0, self.max_seq_len - replace_len)
                    raw_data[replace_start:replace_start+replace_len] = np.random.rand(replace_len,5)
            
            # 统一长度处理
            L = raw_data.shape[0]
            if L < self.max_seq_len:
                pad = ((0, self.max_seq_len - L), (0,0))
                raw_data = np.pad(raw_data, pad, mode='constant')
            else:
                raw_data = raw_data[:self.max_seq_len]
            
            # 增强2：随机噪声
            if self.is_training:
                noise = np.random.normal(0, 0.2, raw_data.shape)
                raw_data = np.clip(raw_data + noise, 0, 1)
            # 分割特征并调整维度
            seq_input = raw_data[:, :4].T.astype(np.float32)  # [4, L]
            ribo_input = raw_data[:, 4].astype(np.float32)     # [L]
            
            return (
                torch.tensor(seq_input),
                torch.tensor(ribo_input),
                torch.tensor(label, dtype=torch.float32)
            )

class FineTuneModel(nn.Module):
    def __init__(self, pretrained_path=None, freeze_backbone=True):
        super().__init__()  
        self.pretrained = MultiModalModel(
            seq_dim=4, 
            ribo_dim=1, 
            hidden_dim=16
        )
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.pretrained.load_state_dict(state_dict)
        
        # 冻结预训练参数
        if freeze_backbone:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        self.attention_pool = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(), 
            nn.Linear(16, 1),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            #nn.Linear(16, 32),
            nn.LayerNorm(16),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            #nn.Sigmoid()
        )

    def forward(self, seq, ribo):
        # 特征提取
        seq_feat = self.pretrained.seq_encoder(seq) 
        ribo_feat = self.pretrained.ribo_encoder(ribo.unsqueeze(1))
        
        # 特征融合
        combined = torch.cat([seq_feat, ribo_feat], dim=1)
        context = self.pretrained.transformer(combined.permute(0,2,1))
        
        # 注意力池化
        attn_weights = self.attention_pool(context)
        pooled = (context * attn_weights).sum(dim=1)
        
        return self.classifier(pooled).squeeze()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.cuda()
        self.scaler = GradScaler()

        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            [
                optim.lr_scheduler.LinearLR(
                    self.optimizer, 
                    start_factor=0.01,
                    total_iters=5
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=args.epochs-5
                )
            ],
            milestones=[5]
        )
        # 其他参数
        self.clip_grad = 1.0
        self.label_smooth = 0.1
        
        # 类别权重计算
        labels = torch.stack([sample[2] for sample in self.train_loader.dataset])
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        self.pos_weight = torch.tensor([n_neg / (n_pos + 1e-7)]).cuda()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for seq, ribo, labels in self.train_loader:
            seq, ribo, labels = seq.cuda(), ribo.cuda(), labels.cuda()
            smooth_labels = labels * (1 - self.label_smooth) + 0.5 * self.label_smooth
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                outputs = self.model(seq, ribo)
                loss = F.binary_cross_entropy_with_logits(outputs, labels, 
                                                        pos_weight=self.pos_weight)
            
            # 梯度裁剪
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 反向传播时梯度裁剪
            
            # 记录指标
            total_loss += loss.item()
            all_preds.append(outputs.sigmoid().detach().cpu())
            all_labels.append(labels.cpu())
        self.scheduler.step()
        # 计算指标
        train_metrics = self._compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
        train_metrics['loss'] = total_loss / len(self.train_loader)  # 添加loss记录
        return train_metrics

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for seq, ribo, labels in self.val_loader:
                seq, ribo, labels = seq.cuda(), ribo.cuda(), labels.cuda()
                outputs = self.model(seq, ribo)
                # 计算损失
                loss = F.binary_cross_entropy_with_logits(
                    outputs, labels,
                    pos_weight=self.pos_weight
                )
                total_loss += loss.item() * labels.size(0)  # 累加批次损失
                
                all_preds.append(outputs.sigmoid().cpu())
                all_labels.append(labels.cpu())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader.dataset)
        val_metrics = self._compute_metrics(torch.cat(all_preds), 
                                           torch.cat(all_labels))
        val_metrics['loss'] = avg_loss
        return val_metrics

    def _compute_metrics(self, preds, labels):
        preds = preds.numpy()
        labels = labels.numpy()
        
        # 动态阈值调整
        if (labels == 1).sum() == 0:
            return {
                'acc': accuracy_score(labels, np.zeros_like(labels)),
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'roc_auc': 0.5,
                'prauc': 0.5,
                'cm': [[len(labels), 0], [0, 0]]
            }
    
        # 扩展阈值范围到0.1-0.9
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = -1
        best_thresh = 0.5
    
    # 动态选择阈值时跳过全负预测
        for thresh in thresholds:
            current_preds = (preds > thresh).astype(int)
            if current_preds.sum() == 0:
                continue
            current_f1 = f1_score(labels, current_preds)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh
    
        final_preds = (preds > best_thresh).astype(int)
    
        # 确保至少有一个预测结果
        if final_preds.sum() == 0:
            final_preds = (preds > 0.5).astype(int)
    
    # 手动计算precision防止除零
        tn, fp, fn, tp = confusion_matrix(labels, final_preds).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        pr_precision, pr_recall, _ = precision_recall_curve(labels, preds)
        pr_auc_value = auc(pr_recall, pr_precision)
    
        return {
            'acc': accuracy_score(labels, final_preds),
            'precision': precision,
            'recall': recall_score(labels, final_preds),
            'f1': f1_score(labels, final_preds),
            'roc_auc': roc_auc_score(labels, preds),
            'prauc': pr_auc_value,
            'cm': [[tn, fp], [fn, tp]]
        }
    
def grouped_stratified_split(dataset, test_size=0.3):
    """基于转录本的分组分层划分"""
    # 获取所有样本的转录本ID和标签
    #transcript_ids = []
    #labels = []
    with h5py.File(dataset.h5_path, 'r') as hf:
        tid_samples = defaultdict(list)
        for idx, key in enumerate(dataset.sample_keys):
            tid = hf[key].attrs['transcript_id']
            tid_samples[tid].append((idx, hf[key].attrs['label']))
    
    # 生成转录本级标签
    tids, tid_labels = [], []
    for tid, samples in tid_samples.items():
        tids.append(tid)
        has_positive = any(s[1] for s in samples)
        tid_labels.append(1 if has_positive else 0)
    
    # 分层划分转录本
    train_tids, val_tids = train_test_split(
        tids,
        test_size=test_size,
        stratify=tid_labels,
        random_state=42
    )
    
    # 收集样本索引
    train_indices = []
    val_indices = []
    for tid in train_tids:
        train_indices.extend([s[0] for s in tid_samples[tid]])
    for tid in val_tids:
        val_indices.extend([s[0] for s in tid_samples[tid]])
    
    # 统计信息
    train_labels = [s[1] for tid in train_tids for s in tid_samples[tid]]
    val_labels = [s[1] for tid in val_tids for s in tid_samples[tid]]
    print(f"[Data Split] Train positives: {sum(train_labels)}/{len(train_indices)}")
    print(f"[Data Split] Val positives: {sum(val_labels)}/{len(val_indices)}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def main(args):
    # 数据预处理
    processor = DataProcessor(
        genome_path=args.genome,
        annotation_path=args.annotation,
        ribo_bam=args.ribo_bam,
        output_h5=args.output_h5
    )
    processor.run()
    full_dataset = H5Dataset(args.output_h5, args.max_seq_len)

    train_set, val_set = grouped_stratified_split(full_dataset)
    train_set.dataset.is_training = True
    val_set.dataset.is_training = False
    
    # 数据加载器
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, args.batch_size*2, num_workers=4)
    train_labels = [train_set.dataset[i][2].item() for i in train_set.indices]
    val_labels = [val_set.dataset[i][2].item() for i in val_set.indices]

    with open(f"{args.output_dir}/training.log", 'w') as f:
        f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Positive ratio - Train: {sum(train_labels)/len(train_labels):.2%}\n")
        f.write(f"Positive ratio - Val: {sum(val_labels)/len(val_labels):.2%}\n\n")

    # 初始化模型
    model = FineTuneModel(args.pretrained)
    trainer = Trainer(model, train_loader, val_loader, args)
    
    
    # 训练循环
    best_auc = 0
    no_improve_count = 0

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()
        
        log_content = [
            f"\nEpoch {epoch+1}/{args.epochs}",
            f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | "
            f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['roc_auc']:.4f} | PR AUC: {train_metrics['prauc']:.4f}",
            f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f} | PR AUC: {train_metrics['prauc']:.4f}"
        ]
        
        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            torch.save({
               'model_state_dict': model.state_dict(),  # 模型参数
                'config': {
                    'pretrained_path': None,  # 微调时不需要外部预训练模型
                    'freeze_backbone': True   # 必须与模型初始化参数一致
                }
            }, f"{args.output_dir}/best_model.pth")                                    
            best_epoch = epoch + 1
            log_content.append(f"Epoch {best_epoch}: New best AUC {best_auc:.4f}") 
            print(f"Epoch {best_epoch}:New best model saved with AUC: {best_auc:.4f}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            log_content.append(f"No improvement for {no_improve_count}/{args.patience} epochs")
            print(f"No improvement for {no_improve_count}/{args.patience} epochs")
        
        if no_improve_count >= args.patience:
            log_content.append(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
           
        with open(f"{args.output_dir}/training.log", 'a') as f:
            for line in log_content:
                print(line)
                f.write(line + '\n')

        if no_improve_count >= args.patience:
            break
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome', type=str, required=True)
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--ribo_bam', type=str, nargs='+', required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--output_h5', type=str, default="processed_data.h5")
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, 
                   help='Number of epochs to wait before early stopping')
    args = parser.parse_args()

    processor = DataProcessor(
        args.genome,
        args.annotation,
        args.ribo_bam,  # 使用下划线命名
        args.output_h5
    )
    main(args)

#python ecoli_finetune.py --genome ./samples/eco/genome.dna.fa --annotation ./samples/eco/annotation.gtf --ribo_bam ./samples/eco/ribo_bams/SRX505573.bam --pretrained best_model_1_no_0.01_10epoch.pt --output_h5 ./ecoli_label.h5 --output_dir ./ --max_seq_len 1024 --batch_size 32 --lr 1e-4