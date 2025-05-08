import os
import h5py
import gffutils
import numpy as np
import pysam
import argparse
from pyfaidx import Fasta
from Bio.Seq import Seq
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, genome_path, annotation_path, ribo_bam, output_h5, 
                 neg_ratio=2.0, max_neg_length=2000):
        self.genome_path = genome_path
        self.annotation_path = annotation_path
        self.ribo_bam = ribo_bam
        self.output_h5 = output_h5
        self.chunk_size = 128
        self.n_threads = 2
        self.valid_starts = {'ATG', 'GTG', 'TTG', 'CTG'}
        self.valid_stops = {'TAA', 'TAG', 'TGA'}
        self.NEG_RATIO = neg_ratio
        self.MAX_NEG_LENGTH = max_neg_length  

    def _is_valid_orf(self, seq):
        if len(seq) < 30 * 3: 
            return False
        if seq[:3] not in self.valid_starts:
            return False
        if seq[-3:] not in self.valid_stops:
            return False
        for i in range(3, len(seq)-3, 3):
            if seq[i:i+3] in self.valid_stops:
                return False
        return True

    def _encode_sequence(self, seq):
        mapping = {'A':0, 'T':1, 'C':2, 'G':3}
        encoded = np.zeros((len(seq),4), dtype=np.float32)
        for i, nt in enumerate(seq.upper()):
            if nt in mapping:
                encoded[i, mapping[nt]] = 1.0
        return encoded

    def _process_single_bam(self, args):
        bam_path, features = args
        bam = pysam.AlignmentFile(bam_path)
        coverage_data = np.zeros(len(features), dtype=object)
        
        for feat_idx, feat in enumerate(features):
            try:
                coverage = np.zeros(feat['end']-feat['start']+1, dtype=np.float32)
                for read in bam.fetch(feat['seqid'], feat['start'], feat['end']):
                    if read.is_reverse != (feat['strand'] == '-'):
                        continue
                    for block in read.get_blocks():
                        overlap_start = max(block[0], feat['start'])
                        overlap_end = min(block[1], feat['end'])
                        coverage[overlap_start-feat['start']:overlap_end-feat['start']] += 1
                
                total = max(bam.mapped, 1)
                coverage = (coverage * 1e6) / (total * (feat['end']-feat['start'])/1e3)
                coverage_data[feat_idx] = np.log1p(coverage)
            except Exception as e:
                print(f"Error processing {feat}: {str(e)}")
                coverage_data[feat_idx] = np.zeros(feat['end']-feat['start']+1)
        return coverage_data
    
    def _generate_negative_samples(self, pos_samples):
        """生成受控的负样本"""
        neg_features = []
        for feat in pos_samples:
            # 类型1：反向链负样本（长度限制）
            rev_seq = str(Seq(feat['sequence']).reverse_complement())
            rev_seq = rev_seq[:min(len(rev_seq), self.MAX_NEG_LENGTH)]
            if not self._is_valid_orf(rev_seq):
                neg_features.append({
                    'seqid': feat['seqid'],
                    'start': feat['start'],
                    'end': feat['start'] + len(rev_seq),
                    'strand': '-' if feat['strand'] == '+' else '+',
                    'sequence': rev_seq,
                    'is_negative': True,
                    'transcript_id': feat['transcript_id']
                })
            
            # 类型2：单移码负样本（随机选择1或2位移码）
            shift = np.random.choice([1, 2])
            mutated = feat['sequence'][shift: shift + self.MAX_NEG_LENGTH]
            mutated = mutated[:len(mutated)//3 *3]
            if not self._is_valid_orf(mutated):
                neg_features.append({
                    'seqid': feat['seqid'],
                    'start': feat['start'] + shift,
                    'end': feat['start'] + shift + len(mutated),
                    'strand': feat['strand'],
                    'sequence': mutated,
                    'is_negative': True,
                    'transcript_id': feat['transcript_id']
                })
        
        # 比例控制
        target_neg = int(len(pos_samples) * self.NEG_RATIO)
        if len(neg_features) > target_neg:
            indices = np.random.choice(len(neg_features), target_neg, replace=False)
            neg_features = [neg_features[i] for i in indices]
        elif len(neg_features) < target_neg:
            # 补充随机负样本
            add_num = target_neg - len(neg_features)
            while add_num > 0:
                rand_feat = np.random.choice(pos_samples)
                rand_seq = rand_feat['sequence'][:np.random.randint(90, 2000)]
                if not self._is_valid_orf(rand_seq):
                    neg_features.append({
                        'seqid': rand_feat['seqid'],
                        'start': rand_feat['start'],
                        'end': rand_feat['start'] + len(rand_seq),
                        'strand': rand_feat['strand'],
                        'sequence': rand_seq,
                        'is_negative': True,
                        'transcript_id': feat['transcript_id']
                    })
                    add_num -= 1
        
        valid_neg = []
        for feat in neg_features:
            if 30 * 3 <= len(feat['sequence']) <= self.MAX_NEG_LENGTH:
                valid_neg.append(feat)
        return valid_neg[:target_neg]
    
    def _process_features(self):
        genome = Fasta(self.genome_path)
        db = gffutils.create_db(
            self.annotation_path,
            dbfn=":memory:", force=True, merge_strategy="merge"
        )

        # 处理正样本
        pos_samples = []
        for cds in db.features_of_type('CDS'):
            seq = str(genome[cds.seqid][cds.start-1:cds.end].seq)
            if cds.strand == '-':
                seq = str(Seq(seq).reverse_complement())
            transcript_id = cds.attributes.get('transcript_id', ['unknown'])[0]
            if not transcript_id:
                transcript_id = cds.attributes.get('Parent', ['unknown'])[0].split(':')[0]

            pos_samples.append({
                'seqid': cds.seqid,
                'start': cds.start,
                'end': cds.end,
                'strand': cds.strand,
                'sequence': seq,
                'is_negative': False,
                'transcript_id': transcript_id 
            })

        # 生成受控负样本
        neg_samples = self._generate_negative_samples(pos_samples)
        
        # 合并样本
        all_features = pos_samples + neg_samples
        np.random.shuffle(all_features)  # 打乱顺序

        # 处理BAM文件
        with Pool(self.n_threads) as pool:
            args = [(bam, all_features) for bam in self.ribo_bam]
            coverage_data = list(tqdm(pool.imap(self._process_single_bam, args),
                                   total=len(self.ribo_bam),
                                   desc="Processing BAMs"))

        # 组装数据
        samples = []
        for feat_idx, feat in enumerate(all_features):
            try:
                cov = np.mean([cd[feat_idx] for cd in coverage_data], axis=0)
                #avg_cov = np.mean(cov)
                #label = 0 if feat['is_negative'] else 1
                
                seq_encoded = self._encode_sequence(feat['sequence'])
                
                feature_matrix = np.zeros((len(seq_encoded),5), dtype=np.float32)
                feature_matrix[:,:4] = seq_encoded
                
                if feat['is_negative']:
                    feature_matrix[:,4] = 0
                    label = 0
                else:
                    cov_truncated = cov[:len(seq_encoded)]
                    feature_matrix[:,4] = cov_truncated
                    label = 0 if np.all(cov_truncated == 0) else 1
                
                samples.append({
                    'seqid': feat['seqid'],
                    'start': feat['start'],
                    'end': feat['end'],
                    'strand': feat['strand'],
                    'data': feature_matrix,
                    'label': label,
                    'transcript_id': feat['transcript_id']
                })
            except Exception as e:
                print(f"Error finalizing sample: {str(e)}")
        return samples

    def run(self):
        samples = self._process_features()
        
        with h5py.File(self.output_h5, 'w') as hf:
            # 添加元数据
            meta = hf.create_group("metadata")
            meta.attrs['total_samples'] = len(samples)
            meta.attrs['pos_ratio'] = self.NEG_RATIO
            meta.attrs['max_neg_length'] = self.MAX_NEG_LENGTH

            for idx, sample in enumerate(samples):
                group = hf.create_group(f"sample_{idx}")
                group.create_dataset('data', data=sample['data'], compression="gzip")
                group.attrs.update({
                    'transcript_id': sample['transcript_id'],
                    'seqid': sample['seqid'],
                    'start': sample['start'],
                    'end': sample['end'],
                    'strand': sample['strand'],
                    'label': sample['label']
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genomic Data Processor')
    parser.add_argument('--genome', required=True, help='Genome FASTA file path')
    parser.add_argument('--annotation', required=True, help='Annotation GTF file path')
    parser.add_argument('--ribo_bam', required=True, nargs='+', help='Ribo-seq BAM files')
    parser.add_argument('--output_h5', required=True, help='Output HDF5 file path')
    parser.add_argument('--threads', type=int, default=2, help='Number of processing threads')
    parser.add_argument('--neg_ratio', type=float, default=2.0, 
                       help='Negative sample ratio (default: 2.0)')
    parser.add_argument('--max_neg_length', type=int, default=2000,
                       help='Maximum length for negative samples (default: 2000)')
    
    args = parser.parse_args()
    
    processor = DataProcessor(
        args.genome,
        args.annotation,
        args.ribo_bam,
        args.output_h5,
        neg_ratio=args.neg_ratio,
        max_neg_length=args.max_neg_length
    )
    processor.n_threads = args.threads
    processor.run()