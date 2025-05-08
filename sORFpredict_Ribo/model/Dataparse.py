import os
import h5py
import gffutils
import numpy as np
import pysam
from pyfaidx import Fasta
from Bio.Seq import Seq
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, species_dirs, output_h5="pretrain_data.h5"):
        self.species_dirs = species_dirs
        self.output_h5 = output_h5
        self.chunk_size = 128  # HDF5存储块大小
        self.n_threads = 2      # 并行线程数

    def _encode_sequence(self, seq):
        """DNA序列转单热编码 (A,T,C,G)"""
        mapping = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]}
        encoded = np.zeros((len(seq),4), dtype=np.uint8)
        for i, nt in enumerate(seq.upper()):
            if nt in mapping:
                encoded[i, mapping[nt]] = 1
        return encoded

    def _process_single_bam(self, args):
        """处理单个BAM文件的覆盖度计算（多进程安全函数）"""
        bam_path, cds_features = args
        bam = pysam.AlignmentFile(bam_path)
        total_reads = max(bam.mapped, 1)  # 避免除以零
        
        # 为所有CDS计算覆盖度
        all_coverage = []
        for cds in cds_features:
            chrom, start, end = cds['seqid'], cds['start'], cds['end']
            cds_len = end - start + 1
            coverage = np.zeros(cds_len, dtype=np.float32)
            
            # 计算原始覆盖度
            for read in bam.fetch(chrom, start, end):
                if read.is_unmapped: continue
                for block in read.get_blocks():
                    overlap_start = max(block[0], start)
                    overlap_end = min(block[1], end)
                    if overlap_start >= overlap_end: continue
                    
                    # 转换为局部坐标
                    local_start = overlap_start - start
                    local_end = overlap_end - start
                    coverage[local_start:local_end] += 1
            
            # RPKM标准化并转为对数尺度
            coverage = (coverage * 1e6) / (total_reads * cds_len / 1e3)
            coverage = np.log1p(coverage)  # 使用log(1+x)平滑
            all_coverage.append(coverage)
        
        return np.array(all_coverage)

    def _process_species(self, species_dir):
        """处理单个物种的所有数据"""
        # 1. 解析基因组和注释
        genome = Fasta(os.path.join(species_dir, "genome.dna.fa"))
        db = gffutils.create_db(
            os.path.join(species_dir, "annotation.gtf"),
            dbfn=":memory:",  # 使用内存数据库加速
            force=True,
            merge_strategy="merge",
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )
        
        # 2. 提取CDS特征
        cds_features = []
        for cds in db.features_of_type('CDS'):
            seq = str(genome[cds.seqid][cds.start-1:cds.end].seq)
            if cds.strand == '-':
                seq = str(Seq(seq).reverse_complement())
            
            cds_features.append({
                'seqid': cds.seqid,
                'start': cds.start,
                'end': cds.end,
                'strand': cds.strand,
                'sequence': seq,
                'phase': cds.frame
            })
        
        # 3. 并行处理所有BAM
        bam_files = [
            os.path.join(species_dir, "ribo_bams", f)
            for f in os.listdir(os.path.join(species_dir, "ribo_bams"))
            if f.endswith(".bam")
        ]
        
        with Pool(self.n_threads) as pool:
            args = [(bam, cds_features) for bam in bam_files]
            coverage_data = list(tqdm(
                pool.imap(self._process_single_bam, args),
                total=len(bam_files),
                desc=f"Processing {os.path.basename(species_dir)} BAMs"
            ))
        
        # 4. 生成HDF5数据条目
        samples = []
        for cds_idx, cds in enumerate(cds_features):
            # 序列编码
            seq_encoded = self._encode_sequence(cds['sequence'])
            seq_len = seq_encoded.shape[0]
            
            # 计算所有ribo样本的平均覆盖度 (L,5)
            coverage = np.mean([cov_data[cds_idx] for cov_data in coverage_data], axis=0)
            coverage = coverage.reshape(-1, 1) 
            
            # 组合样本数据
            sample = np.zeros((seq_len, 5), dtype=np.float32)
            sample[:, :4] = seq_encoded
            sample[:, 4] = coverage.squeeze()
            
            samples.append({
                'species': os.path.basename(species_dir),
                'seqid': cds['seqid'],
                'start': cds['start'],
                'end': cds['end'],
                'strand': cds['strand'],
                'data': sample
            })
        
        return samples

    def run(self):
        """重构后的HDF5存储逻辑"""
        with h5py.File(self.output_h5, 'w') as hf:
            # 创建全局元数据组
            meta_group = hf.create_group("metadata")
            meta_group.attrs["n_species"] = len(self.species_dirs)
            
            # 按物种处理
            for species_dir in self.species_dirs:
                species_name = os.path.basename(species_dir)
                species_group = hf.create_group(species_name)
                
                samples = self._process_species(species_dir)
                species_group.attrs["n_cds"] = len(samples)
                
                # 存储每个CDS为独立的变长数组
                for idx, sample in enumerate(tqdm(samples, desc=f"Writing {species_name}")):
                    cds_group = species_group.create_group(f"cds_{idx}")
                    # 存储变长数据 (seq_len, 5)
                    cds_group.create_dataset(
                        "data", 
                        data=sample['data'], 
                        dtype=np.float32,
                        compression="gzip"
                    )
                    # 存储元数据为属性
                    cds_group.attrs["seqid"] = sample['seqid']
                    cds_group.attrs["start"] = sample['start']
                    cds_group.attrs["end"] = sample['end']
                    cds_group.attrs["strand"] = sample['strand']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--species_dirs", nargs="+", required=True, help="Path to species directories")
    parser.add_argument("--output", required=True, help="Output HDF5 filename")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads to use")
    
    args = parser.parse_args()

    processor = DataProcessor(args.species_dirs, args.output)
    processor.n_threads = args.threads
    processor.run()
