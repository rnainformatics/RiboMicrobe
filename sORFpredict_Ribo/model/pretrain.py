import os
import h5py
import json
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import Subset
from collections import defaultdict

#数据加载模块
class H5Dataset(Dataset):
    def __init__(self, h5_path, max_seq_len, pool_factor=4, is_training=True):
        self.pool_factor = pool_factor
        self.max_seq_len = max_seq_len // pool_factor
        self.h5 = h5py.File(h5_path, 'r')
        self.samples = []
        self.is_training = is_training

        for species in self.h5.keys():
            if species == "metadata": continue
            species_grp = self.h5[species]
            for cds_id in species_grp.keys():
                if 'data' in species_grp[cds_id]:
                    self.samples.append((species, cds_id))

    def __len__(self): return len(self.samples)

    def _pool_1d(self, arr, factor):
        remainder = len(arr) % factor
        if remainder: arr = arr[:-remainder]
        return arr.reshape(-1, factor).mean(axis=1)

    def _pool_2d_max(self, arr, factor):
        remainder = arr.shape[0] % factor
        if remainder: arr = arr[:-remainder]
        pooled = arr.reshape(-1, factor, 4)
        sum_pooled = pooled.sum(axis=1)
        max_indices = np.argmax(sum_pooled, axis=1)
        one_hot = np.zeros_like(sum_pooled)
        one_hot[np.arange(len(max_indices)), max_indices] = 1
        return one_hot.astype(np.float32)

    def __getitem__(self, idx):
        species, cds_id = self.samples[idx]
        grp = self.h5[species][cds_id]

        data = grp['data'][:]  # [L,5]
        seq = data[:, :4]      # [L,4]
        ribo = data[:, 4]      # [L]

        # 下采样处理
        seq_down = self._pool_2d_max(seq, self.pool_factor)  # [L//4,4]
        ribo_down = self._pool_1d(ribo, self.pool_factor)  # [L//4]

        # 长度对齐
        curr_len = seq_down.shape[0]
        if curr_len > self.max_seq_len:
            seq_final = seq_down[:self.max_seq_len]
            ribo_final = ribo_down[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, bool)
        else:
            pad = self.max_seq_len - curr_len
            seq_final = np.pad(seq_down, ((0,pad),(0,0)), 'constant')
            ribo_final = np.pad(ribo_down, (0,pad), 'constant')
            mask = np.zeros(self.max_seq_len, bool)
            mask[:curr_len] = True
        
        seq_target = np.zeros(seq_final.shape[0], dtype=np.int64)
        valid_len = min(len(seq_down), self.max_seq_len)
        seq_target[:valid_len] = seq_down.argmax(axis=1)[:valid_len]
        seq_target[valid_len:] = -1
        #noise = np.random.normal(0, 0.1, seq_final.shape) * mask[:, None]
        #seq_final = np.clip(seq_final + noise * mask[:, None], 0.0, 1.0)
        #row_sums = seq_final.sum(axis=1, keepdims=True)
        #row_sums = np.where(row_sums == 0, 1.0, row_sums)
        #seq_final = seq_final / row_sums
        #valid_pos = np.where(mask)[0]
        #mask_len = int(len(valid_pos)*0.7)
        #if mask_len > 0:
            #mask_pos = np.random.choice(valid_pos, mask_len, replace=False)
            #seq_final[mask_pos] = np.random.dirichlet([0.05]*4, size=mask_len)
        corrupt_mask = np.zeros_like(mask, dtype=bool)
        if isinstance(self.h5, h5py.File):  # 训练模式
            seq_final = seq_final + np.random.normal(0, 0.5, seq_final.shape) * mask[:, None]

            # 动态mask比例
            mask_ratio = np.random.choice([0.1, 0.3, 0.5])  # 随机选择破坏强度
            valid_pos = np.where(mask)[0]
            mask_len = int(len(valid_pos) * mask_ratio)

            # 多模态破坏
            if mask_len > 0:
                mask_pos = np.random.choice(valid_pos, mask_len, replace=False)
                corrupt_mask[mask_pos] = True
                # 混合噪声类型
                noise_type = np.random.choice(['dirichlet', 'gaussian', 'zeros'])
                if noise_type == 'dirichlet':
                    dirichlet_alpha = np.random.uniform(0.05, 0.3, 4)
                    seq_final[mask_pos] = np.random.dirichlet(dirichlet_alpha, size=mask_len)
                elif noise_type == 'gaussian':
                    gauss_noise = np.clip(np.random.normal(0, 0.3, (mask_len,4)), 0, 1)
                    seq_final[mask_pos] = gauss_noise
                else:
                    seq_final[mask_pos] = 0.25
            # 随机通道丢弃
            channel_drop = np.random.rand() < 0.2
            if channel_drop:
                drop_channel = np.random.randint(0,4)
                seq_final[:, drop_channel] = np.random.uniform(0, 0.1)
        else:  # 验证模式保持中等强度破坏
            seq_final = seq_final + np.random.normal(0, 0.5, seq_final.shape) * mask[:, None]
            mask_ratio = np.random.choice([0.2, 0.4, 0.6])
            valid_pos = np.where(mask)[0]
            mask_len = int(len(valid_pos) * mask_ratio)

            if mask_len > 0:
                mask_pos = np.random.choice(valid_pos, mask_len, replace=False)
                corrupt_mask[mask_pos] = True
                # 使用dirichlet噪声（固定参数）
                noise_type = np.random.choice(['dirichlet', 'gaussian', 'zeros'])
                if noise_type == 'dirichlet':
                    dirichlet_alpha = np.random.uniform(0.05, 0.3, 4)
                    seq_final[mask_pos] = np.random.dirichlet(dirichlet_alpha, size=mask_len)
                elif noise_type == 'gaussian':
                    seq_final[mask_pos] = np.clip(np.random.normal(0, 0.3, (mask_len,4)), 0, 1)
                else:
                    seq_final[mask_pos] = 0.25

            # 保持与训练相同的通道丢弃概率
            if np.random.rand() < 0.2:
                drop_channel = np.random.randint(0,4)
                seq_final[:, drop_channel] = np.random.uniform(0, 0.1)

        # 添加数值稳定性处理
        seq_final = np.clip(seq_final, 1e-6, 1.0-1e-6)  # 防止出现0或1
        seq_final = seq_final / seq_final.sum(axis=1, keepdims=True)  # 重新归一化

        return (
            torch.from_numpy(seq_final).float(),
            torch.from_numpy(ribo_final).float(),
            torch.from_numpy(mask).bool(),
            torch.from_numpy(seq_target).long(),
            torch.from_numpy(corrupt_mask).bool()
        )

def collate_fn(batch):
    seq, ribo, mask, target, corrupt_mask = zip(*batch)
    return torch.stack(seq), torch.stack(ribo), torch.stack(mask), torch.stack(target), torch.stack(corrupt_mask)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels*2, 3, padding=1),  
            nn.BatchNorm1d(channels*2),
            nn.ReLU(),
            nn.Conv1d(channels*2, channels, 1),  
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

#模型模块
class MultiModalModel(nn.Module):
    def __init__(self, seq_dim=4, ribo_dim=1, hidden_dim=16):
        super().__init__()

        # 序列编码器
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(seq_dim, 32, 11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.MaxPool1d(4),
            #nn.AdaptiveAvgPool1d(128),
            nn.Conv1d(32, hidden_dim, 5, padding=2),
            #nn.Dropout(0.5),
            ResBlock(hidden_dim)
        )

        # Ribosome编码器
        self.ribo_encoder = nn.Sequential(
            nn.Conv1d(ribo_dim, 32, 11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.MaxPool1d(4),
            #nn.Dropout(0.5),
            #nn.AdaptiveAvgPool1d(128),
            nn.Conv1d(32, hidden_dim, 5, padding=2),
            #nn.Dropout(0.5),
            ResBlock(hidden_dim)
        )

        # 融合层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
                #dropout=0.5
            ), num_layers=4
        )

        # 上采样层
        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim*2, 4, stride=4),
            nn.ReLU()
        )

        # 预测头
        #self.pretrain_heads = nn.ModuleDict({
            #'seq_recon': nn.Linear(hidden_dim*2, seq_dim),
            #'ribo_pred': nn.Sequential(
                #nn.Linear(hidden_dim*2, 64), nn.GELU(),
                #nn.Linear(64, 1)
            #)
        #})
        self.pretrain_heads = nn.ModuleDict({
            'seq_recon': nn.Sequential(
                nn.LayerNorm(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(16, seq_dim)
            ),
            'ribo_pred': nn.Sequential(
                nn.LayerNorm(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(16, 1)
            )
        })

    def forward(self, seq, ribo):
        # 编码处理
        seq_feat = self.seq_encoder(seq.permute(0,2,1))      # [B,C,L//4]
        ribo_feat = self.ribo_encoder(ribo.unsqueeze(1))    # [B,C,L//4]

        # 特征融合与上采样
        combined = torch.cat([seq_feat, ribo_feat], dim=1) # [B, 2C, L//4]
        combined = combined.permute(0, 2, 1)                # [B,2C,L]

        # Transformer处理
        context = self.transformer(combined)


        return {
            'seq_recon': self.pretrain_heads['seq_recon'](context),
            'ribo_pred': self.pretrain_heads['ribo_pred'](context).squeeze(-1)
        }


#训练评估模块
class MetricTracker:
    def __init__(self):
        self.class_metrics = {'acc':0, 'recall':0, 'f1':0, 'total':0}
        self.reg_metrics = {'mse':0, 'mae':0, 'pearson_r':0, 'total':0}

    def update_class(self, preds, targets, corrupt_mask):
        if corrupt_mask.sum() == 0:
            return
        valid_t = targets[corrupt_mask]
        valid_p = preds.argmax(2)[corrupt_mask]

        valid_indices = (valid_t >= 0) & (valid_t < 4)
        valid_t = valid_t[valid_indices]
        valid_p = valid_p[valid_indices]

        # 计算混淆矩阵
        cm = torch.zeros(4,4, device=preds.device)
        for t, p in zip(valid_t, valid_p):
            cm[t.long(), p.long()] += 1


        # 计算每个类别的指标（避免宏平均偏差）
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)

        # 按样本数加权平均（更合理）
        weight = cm.sum(1) / (cm.sum() + 1e-6)
        self.class_metrics['acc'] += (valid_p == valid_t).float().mean().item()
        self.class_metrics['recall'] += (rec * weight).sum().item()
        self.class_metrics['f1'] += (f1 * weight).sum().item()
        self.class_metrics['total'] += 1

    def update_reg(self, preds, targets, mask):
        valid_p = preds[mask]
        valid_t = targets[mask]

        self.reg_metrics['mse'] += F.mse_loss(valid_p, valid_t).item()
        self.reg_metrics['mae'] += F.l1_loss(valid_p, valid_t).item()

        # 皮尔逊相关系数
        vp = valid_p - valid_p.mean()
        vt = valid_t - valid_t.mean()
        corr = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-6)
        self.reg_metrics['pearson_r'] += corr.item()
        self.reg_metrics['total'] += 1

    def get_metrics(self):
        return {
            'classification': {k: v/self.class_metrics['total'] if k != 'total' else v
                              for k,v in self.class_metrics.items()},
            'regression': {k: v/self.reg_metrics['total'] if k != 'total' else v
                         for k,v in self.reg_metrics.items()}
        }

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()

class PreTrainer:
    def __init__(self, model, device, config, train_loader):
        all_targets = []
        for _, _, mask, target, corrupt_mask in train_loader:
            all_targets.append(target[mask])
        all_targets = torch.cat(all_targets).cpu()
        class_counts = torch.bincount(all_targets, minlength=4)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights /= class_weights.sum()

        # 使用加权交叉熵损失
        self.seq_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=-1
        )
        self.ribo_criterion = nn.HuberLoss()

        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr']*0.1)
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #self.optimizer,
            #max_lr=config['lr'],
            #total_steps=config['epochs'] * len(train_loader),
            #pct_start=0.3,  # 30%时间用于预热
            #anneal_strategy='linear'
        #)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']//2,  # 每半程重置学习率
            eta_min=config['lr']*0.01
        )
        self.model = model.to(device)
        self.log_file = open("training_log.json", "w")
        self.device = device
        self.config = config


        # 初始化损失函数和优化器
        #self.seq_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        #self.ribo_criterion = nn.HuberLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_score = -float('inf')

    def _log_metrics(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()

    def train_epoch(self, loader):
        self.model.train()
        tracker = MetricTracker()
        total_loss = 0.0

        for seq, ribo, mask, seq_target, corrupt_mask in tqdm(loader, desc="Training"):
            seq = seq.to(self.device)
            ribo = ribo.to(self.device)
            mask = mask.to(self.device)
            seq_target = seq_target.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(seq, ribo)
                # 计算损失
                seq_loss = self.seq_criterion(
                    outputs['seq_recon'].view(-1,4),
                    seq_target.view(-1)
                )
                ribo_loss = self.ribo_criterion(
                    outputs['ribo_pred'][mask],
                    ribo[mask]
                )
                loss = self.config['seq_weight']*seq_loss + \
                       self.config['ribo_weight']*ribo_loss
            
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * 0.1 / (1 + epoch)**0.5
                    param.grad += noise
            # 反向传播
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 更新指标
            tracker.update_class(outputs['seq_recon'].detach(), seq_target, corrupt_mask)
            tracker.update_reg(outputs['ribo_pred'].detach(), ribo, mask)
            total_loss += loss.item()


        return total_loss/len(loader), tracker.get_metrics()

    def evaluate(self, loader):
        self.model.eval()
        tracker = MetricTracker()
        total_loss = 0.0

        with torch.no_grad():
            for seq, ribo, mask, seq_target, corrupt_mask in loader:  # 解包四个元素
                seq, ribo, mask, seq_target = seq.to(self.device), ribo.to(self.device), mask.to(self.device), seq_target.to(self.device)
                seq_target = seq_target.to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(seq, ribo)

                # 计算损失
                seq_loss = self.seq_criterion(outputs['seq_recon'].reshape(-1,4),
                                            seq_target.reshape(-1))
                ribo_loss = self.ribo_criterion(outputs['ribo_pred'][mask], ribo[mask])
                total_loss += (self.config['seq_weight']*seq_loss +
                            self.config['ribo_weight']*ribo_loss).item()

                # 更新指标
                tracker.update_class(outputs['seq_recon'], seq_target, corrupt_mask)
                tracker.update_reg(outputs['ribo_pred'], ribo, mask)

        return total_loss/len(loader), tracker.get_metrics()

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file is not None:
            self.log_file.close()

def get_species_samples(dataset):
    species_dict = defaultdict(list)
    for idx in range(len(dataset)):
        species, cds_id = dataset.samples[idx]
        species_dict[species].append(idx)
    return species_dict

#主程序
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq_weight", type=float, default=0.3)
    parser.add_argument("--ribo_weight", type=float, default=0.7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_species_ratio", type=float, default=0.4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备
    full_dataset = H5Dataset(args.h5_path, args.max_seq_len, is_training=True)
    species_dict = get_species_samples(full_dataset)
    species_list = list(species_dict.keys())
    np.random.shuffle(species_list)
    
    # 动态计算划分比例
    total_samples = len(full_dataset)
    train_samples, val_samples = [], []
    current_count = 0
    split_ratio = 1 - args.val_species_ratio
    
    for species in species_list:
        if current_count + len(species_dict[species]) > total_samples * split_ratio:
            break
        train_samples.extend(species_dict[species])
        current_count += len(species_dict[species])
    
    val_samples = [i for i in range(len(full_dataset)) if i not in train_samples]
    
    # 创建子集
    train_set = Subset(full_dataset, train_samples)
    val_set = Subset(full_dataset, val_samples)
    val_set.dataset.is_training = False

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, args.batch_size, collate_fn=collate_fn)

    # 初始化训练器
    config = {
        'seq_weight': args.seq_weight,
        'ribo_weight': args.ribo_weight,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs
    }
    model = MultiModalModel()
    trainer = PreTrainer(model, device, config, train_loader)

    best_val_loss = float('inf')
    no_improve = 0
    patience = 5  # 连续多个epoch验证损失未改善则停止
    best_model_path = "best_model.pt"
    # 训练循环
    for epoch in range(args.epochs):
        train_loss, train_metrics = trainer.train_epoch(train_loader)
        val_loss, val_metrics = trainer.evaluate(val_loader)
        trainer._log_metrics(epoch+1, train_loss, val_loss, train_metrics, val_metrics)    # 记录日志
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # 保存当前最佳模型
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nValidation loss hasn't improved for {patience} epochs. Early stopping...")
                break

        # 打印进度
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f} | No improvement count: {no_improve}/{patience}")
        print("[Train] Seq Acc: {:.2%} | Ribo R: {:.4f}".format(
            train_metrics['classification']['acc'],
            train_metrics['regression']['pearson_r']))
        print("[Val]   Seq Acc: {:.2%} | Ribo R: {:.4f}".format(
            val_metrics['classification']['acc'],
            val_metrics['regression']['pearson_r']))
        print("-" * 60)

        # 定期保存检查点
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, f"checkpoint_epoch{epoch+1}.pt")