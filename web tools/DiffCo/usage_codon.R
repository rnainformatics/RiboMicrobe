suppressPackageStartupMessages(library(GenomicAlignments))
suppressPackageStartupMessages(library(GenomicFeatures))
#suppressPackageStartupMessages(library(biomaRt))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ggplot2))
#library(TxDb.Mmusculus.UCSC.mm10.knownGene)
suppressPackageStartupMessages(library("jsonlite"))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(seqinr))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library("zoo"))
suppressPackageStartupMessages(library("signal"))
suppressPackageStartupMessages(library("parallel"))
suppressPackageStartupMessages(library("plyr"))
suppressPackageStartupMessages(library(Rsamtools))
#suppressPackageStartupMessages(library("ssh"))

codon_usage_count <- function(coverage_cds, d, normalization) {

	# Keep only in-frame reads
	coverage_cds <- coverage_cds[(CDS_coordinate-1)%%3 == 0]
	
	#coverage_cds <- coverage_cds[codon > 15 & codon < length_codon-5]
	coverage_cds <- coverage_cds[codon > 15]
	
	##add seq
	coverage_cds[transcript_seqs, cdsseq:=i.seq, on=.(transcriptID)]
	siteNames <- c("E","P","A","+1","+2","+3")
	names(siteNames) <- -2:3
	mutialinfor <- function(i) {
	   coverage_cds[strand=='+', V1 := mapply(function(seqs,codon) toupper(substr(seqs, codon, codon+2)), cdsseq, utr5_len+tssExtension+CDS_coordinate+(i*3))]
	   aatem <- coverage_cds[,list(V1,coverage)]
	   setnames(aatem, "V1",paste0('position_', siteNames[as.character(i)]))
	}
	require(parallel)
    #aa <- mclapply(-2:3, mutialinfor, mc.cores = 6)
	aa <- lapply(-2:3, mutialinfor)
	coverage_cds[,cdsseq:=NULL]
	
	coverage_cds[, freq := 1]
	# Calculate sum of raw reads by codon sequence
	summary <- list()
	for (i in 1:length(aa)) {
		tem <- aa[[i]]
		pos <- names(tem)[which(names(tem) %like% 'position')]
		if(pos == "position_A") {
			codon_A_sites = tem[,position_A]
		}
		summary[[i]] <- setnames(tem[, sum(coverage), by=eval(as.character(pos))], c('codon', pos))
	}

	codon_usage <- Reduce(function(x,y) merge(x,y,all=T), summary) 
	codon_usage[is.na(codon_usage)] = 0
	codon_usage <- codon_usage[grepl('[ATGC]{3}', codon_usage$codon)]
	
	# Normalize by +1 to +3
	codon_usage[, baseline:=rowMeans(codon_usage[, paste0('position_', '+', c(1:3))])]
	norm_codon_usage<-sapply(codon_usage[, names(codon_usage) %like% 'position', with=F], function(x) x/codon_usage$baseline)
	codon_usage <- cbind(codon_usage[,1], norm_codon_usage)
	

	# Calculate read count for each codon as percentage
	codon_usage <- as.data.table(append(codon_usage, list(aminoacid=as.character(Biostrings::translate(DNAStringSet(codon_usage$codon)))), after = 1))
	
	# Keep CDS with at least 50 codons after removing beggining and end codons
	#coverage_cds <- coverage_cds[(length/3)>(2*d)+50][codon > d & codon < (ceiling(length/3)-d)]
	
	coverage_cds$position_A <- codon_A_sites
	if (normalization == 'gene_avg_density') {
		# Calculate total footprints for each CDS
		stats <- coverage_cds[, sum(coverage), by='transcriptID']
		times <- coverage_cds[, sum(freq), by='transcriptID']
		coverage_cds[stats, total_rpf := i.V1, on='transcriptID']
		coverage_cds[times, times := i.V1, on='transcriptID']
		coverage_cds[, coverage := coverage/(total_rpf/times)]
	}
	
	codon_occuracy_matrix = coverage_cds[,.(list(coverage)),by='position_A']
	setnames(codon_occuracy_matrix,c("position_A","V1"),c("codon","occupacy_metric"))
	codon_usage[codon_occuracy_matrix, occupacy_metric := i.occupacy_metric, on="codon"]
	
	# Output ordered results table
	return(codon_usage)
}


args <- commandArgs(TRUE)
species <- args[1]
resultdir <- args[2]
samplename <- args[3]

load(paste0("/data/",species,".txlens.rda"))
if(species == "sce_R64" | species == "ecoli_k12" | species == "bsu_168" | species == "pfu_dsm_3638" | species == "hsa_NRC1"){
	tssExtension = 25;
} else {
	tssExtension = 0;
}

dataset <- paste0(resultdir, "/temp/coverage_cds.tsv")
cov_list <- fread(dataset, header = T)

txlens <- txlens[cds_len>0]
txlens[, maxlen := max(cds_len), by = gene_id]
txlensMax <- txlens[cds_len==maxlen]
txlensMax <- txlensMax[!duplicated(gene_id)]

transcript_seqs <- read.fasta(paste0("/data/",species,".txdb.fa"), seqtype = 'DNA', as.string = T)
transcript_seqs <- data.table(transcriptID=names(transcript_seqs), seq=as.character(transcript_seqs))

cov_list <- cov_list[transcriptID %in% txlensMax$tx_name]
usage <- codon_usage_count(cov_list, 30, 'gene_avg_density')
usage <-  usage[!is.nan(position_A)]
save(usage, file=paste0(resultdir,'/codon_usage_64_norm.rda'))

if(nrow(cov_list[totalcov>20]) > 0) {
	cov_list <- cov_list[totalcov > 20]
}

usage[,codon:=gsub('T','U',codon)] # Change T for U
usage<-usage[aminoacid!='*'] 

codons <- unique(usage$codon)

for (i in names(usage)[names(usage) %like% 'position']) {
	usage_summary <- usage[,.(codon, aminoacid, eval(as.name(i)))]

	#order by alphabet
	usage_summary = usage_summary[order(usage_summary$codon),]
	usage_summary[codon=="CUG"|codon=="UUG",aminoacid :='L']
	usage_summary[,codon:=factor(codon, level=codon)]
	usage_summary[V3 > 2, V3 := 2]
	usage_summary[V3 < -2, V3 := -2]
	ggplot(usage_summary, aes(x=codon, y=V3)) +
		labs(x='', y='Codon occupancy') +
		geom_point(color="green") +
		theme(axis.text.x=element_text(angle=90,vjust=0.5),legend.position="none") +
		# geom_text(aes(label=mCN_mES$aminoacid)) +
		theme(panel.background = element_blank(), axis.line = element_line(colour ="black", size = 0.8), panel.grid.major.y = element_line(colour = "gray90", size = 0.5, linetype='solid'),
		panel.grid.major.x = element_line(colour = "gray90", size = 0.5, linetype='33'),
		axis.ticks = element_line(colour = "black", size = 0.8, linetype='solid'),
		axis.text=element_text(size=9)) + geom_hline(yintercept = 1.2,linetype=2) +
		scale_x_discrete(labels=paste0(usage_summary$codon, '(', usage_summary$aminoacid, ')')) +
		scale_y_continuous(expand = c(0, 0), limits = c(0.0,2.0))
	ggsave(paste0(resultdir,samplename,'_usage_graph_',i,'.pdf'), width = 12, height = 6)
	ggsave(paste0(resultdir,samplename,'_usage_graph_',i,'.png'), width = 12, height = 6, bg='white')
	
	#order by occupancy
	usage_summary = usage_summary[rev(order(usage_summary$V3)),]
	usage_summary[codon=="CUG"|codon=="UUG",aminoacid :='L']
	usage_summary[,codon:=factor(codon, level=codon)]
	ggplot(usage_summary, aes(x=codon, y=V3)) +
		labs(x='', y='Codon occupancy') +
		geom_point(color="green") +
		theme(axis.text.x=element_text(angle=90,vjust=0.5),legend.position="none") +
		# geom_text(aes(label=mCN_mES$aminoacid)) +
		theme(panel.background = element_blank(), axis.line = element_line(colour ="black", size = 0.8), panel.grid.major.y = element_line(colour = "gray90", size = 0.5, linetype='solid'),
		panel.grid.major.x = element_line(colour = "gray90", size = 0.5, linetype='33'),
		axis.ticks = element_line(colour = "black", size = 0.8, linetype='solid'),
		axis.text=element_text(size=9)) + geom_hline(yintercept = 1.2,linetype=2) +
		scale_x_discrete(labels=paste0(usage_summary$codon, '(', usage_summary$aminoacid, ')')) +
		scale_y_continuous(expand = c(0, 0), limits = c(0.0,2.0))
	ggsave(paste0(resultdir,samplename,'_usage_graph_sorted_',i,'.pdf'), width = 12, height = 6)
	ggsave(paste0(resultdir,samplename,'_usage_graph_sorted_',i,'.png'), width = 12, height = 6, bg='white')
}
usage[,occupacy_metric:=NULL]
fwrite(usage, file=paste0(resultdir, "/usage.txt"), sep="\t")
fwrite(usage, file=paste0(resultdir, "/usage.tsv"))

