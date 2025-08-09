#suppressPackageStartupMessages(library(GenomicAlignments))
#suppressPackageStartupMessages(library(GenomicFeatures))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library("jsonlite"))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(seqinr))
suppressPackageStartupMessages(library(cowplot))
#suppressPackageStartupMessages(library("zoo"))
#suppressPackageStartupMessages(library("signal"))
suppressPackageStartupMessages(library("parallel"))
suppressPackageStartupMessages(library("plyr"))
#suppressPackageStartupMessages(library(Rsamtools))
suppressPackageStartupMessages(library(getopt))
suppressPackageStartupMessages(library(ggpubr))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggthemes))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(Cairo))
suppressPackageStartupMessages(library(circlize))
suppressPackageStartupMessages(library(ggplotify))

options(bitmapType = 'cairo')

spec <- matrix(
c("jobid", "j", 2, "character", "This is jobid!",
  "species",  "s", 2, "character", "This is species!",
  "samplenames",  "n", 2, "character", "This is samplenames!",
  "bia", "i", 2, "character",  "This is bia!",
  "email",  "e", 2, "character",  "This is email!",
  "offset_position",  "o", 2, "character",  "This is offset_position!",
  "fc_cutoff",  "r", 2, "character",  "This is fc_cutoff!",
  "pvalue_cutoff",  "p", 2, "character",  "This is pvalue_cutoff!",
  "help",   "h", 0, "logical",  "This is Help!"),
   byrow=TRUE, ncol=5)

opt <- getopt(spec=spec)
if( !is.null(opt$help) || is.null(opt$species) || is.null(opt$bia) || is.null(opt$samplenames) || is.null(opt$fc_cutoff) || is.null(opt$pvalue_cutoff) || is.null(opt$offset_position) || is.null(opt$jobid)){
    cat(paste(getopt(spec=spec, usage = T), "\n"))
    quit()
}

samplenames <- opt$samplenames
species <- opt$species
jobid <- opt$jobid
offset_position <- opt$offset_position
bia <- opt$bia
fc_cutoff <- opt$fc_cutoff
pvalue_cutoff <- opt$pvalue_cutoff
resultdir <- paste0("./data/",jobid)

dir.create(resultdir)

get_log_status <- function(messages, status){
        logdate <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
        messages <- gsub("\r?\n|\r", "", messages)
        messages <- gsub("\"", "'", messages)
        current <- paste(c(10,20,30,40,50,60,70,80,90,100),collapse =" ")
        writeLines(paste0('{"species":','"',species,'",','"jobid":','"',jobid,'",','"message":','"',messages,'",','"current":','"',status,'",','"offset_position":','"',offset_position,'",','"bia":','"',bia,'",','"sample":','"',samplenames,'"}'), paste0(resultdir,"/status.json"))
        if(status != "Error") {
                messages <- paste0(messages,"\t",logdate,"\n")
        }
        message(messages)
    write(messages, file=paste0(resultdir,"/log.txt"), append=TRUE, sep="\n")
}


grouplabels <- unlist(strsplit(samplenames, "[#]"))
groupAsamplenames <- unlist(strsplit(grouplabels[1], "[;]"))
groupBsamplenames <- unlist(strsplit(grouplabels[2], "[;]"))
typeList1 <- list()
typeList2 <- list()
CodonsampleNames <- c()
column_a <- paste0("position_", bia)
column_b <- paste0("position_.", offset_position)

for (i in 1:length(groupAsamplenames)) {
    groupAsamplename <- groupAsamplenames[i]
    infile <- paste0("./data/", species, "/", groupAsamplename, ".usage.txt")
    message(groupAsamplename)
    peak <- read.table(infile, sep = "\t", head = TRUE)
	peak <- as.data.table(peak)
	peak[,codon_aaID:=paste0(codon, "(", aminoacid, ")")]
    peak_a <- peak[, .(codon_aaID, get(column_a))]
	setnames(peak_a, c("codon_aaID", groupAsamplename))
    typeList1[[groupAsamplename]] <- peak_a
	peak_b <- peak[, .(codon_aaID, get(column_b))]
	setnames(peak_b, c("codon_aaID", groupAsamplename))
    typeList2[[groupAsamplename]] <- peak_b
    CodonsampleNames <- c(CodonsampleNames, groupAsamplename)
}

for (i in 1:length(groupBsamplenames)) {
    groupBsamplename <- groupBsamplenames[i]
    infile <- paste0("./data/", species, "/", groupBsamplename, ".usage.txt")
    message(groupBsamplename)
    peak <- read.table(infile, sep = "\t", head = TRUE)
	peak <- as.data.table(peak)
	peak[,codon_aaID:=paste0(codon, "(", aminoacid, ")")]
    peak_a <- peak[, .(codon_aaID, get(column_a))]
	setnames(peak_a, c("codon_aaID", groupBsamplename))
    typeList1[[groupBsamplename]] <- peak_a
	peak_b <- peak[, .(codon_aaID, get(column_b))]
	setnames(peak_b, c("codon_aaID", groupBsamplename))
    typeList2[[groupBsamplename]] <- peak_b
    CodonsampleNames <- c(CodonsampleNames, groupBsamplename)
}

typeList1 <- lapply(typeList1, function(i) setDT(i, codon_aaID))
merged1 <- Reduce(function(...) merge(..., by = "codon_aaID", all = T), typeList1)
merged1 <- na.omit(merged1)
typeList2 <- lapply(typeList2, function(i) setDT(i, codon_aaID))
merged2 <- Reduce(function(...) merge(..., by = "codon_aaID", all = T), typeList2)
merged2 <- na.omit(merged2)
sampleidslen <- length(CodonsampleNames)



process_column <- function(merged, CodonsampleNames, condition_prefix, groupAsamplenames, groupBsamplenames, resultdir, fc_cutoff, pvalue_cutoff) {	 
  data <- as.data.frame(merged)
  rownames(data) <- data[,1]
  data <- data[,-1]
  table2 <- data.frame(name = CodonsampleNames, condition = c(rep("CTRL",length(groupAsamplenames)), rep("KO",length(groupBsamplenames))))
  group<-table2[,2]
  design <- model.matrix(~0+factor(group))
  colnames(design)=levels(factor(group))
  rownames(design)=colnames(data)
  contrast.matrix<-makeContrasts("KO-CTRL",levels=design)
  ##step1
  fit <- lmFit(data,design)
  ##step2
  fit2 <- contrasts.fit(fit, contrast.matrix)
  fit2 <- eBayes(fit2)
  ##step3 
  tempOutput = topTable(fit2, coef=1, n=Inf)
  nrDEG = na.omit(tempOutput)
  nrDEG$codon_aaID=row.names(nrDEG)
  data$codon_aaID=row.names(data)
  result = merge(data,nrDEG,by="codon_aaID",all.x=T)
  #save(result,file="result.Data")
  result_sub <- subset(result, abs(logFC)>=fc_cutoff & P.Value<pvalue_cutoff)
  xx = c("AveExpr","t","B")
  result_sub <- result_sub[,!names(result_sub) %in% xx]
  colnames(result_sub) <- gsub("\\.","_",colnames(result_sub))
  write.table(result_sub, file = paste0(resultdir, "/diffcodon_level_", condition_prefix, ".txt"))
  result_sub_1 <- cbind(result_sub[,1],round(result_sub[,-1],digits=3))
  colnames(result_sub_1) <- colnames(result_sub)
  jsonlite::write_json(result_sub_1, path = paste0(resultdir, "/diffcodon_level_", condition_prefix, ".json"))

}

#plot
plot_volcano <- function(result, fc_cutoff, pvalue_cutoff, resultdir, condition_prefix) {
    result <- result %>%
	
    mutate(threshold = ifelse(logFC >= fc_cutoff & P_Value < pvalue_cutoff, "Up", 
                                  ifelse(-logFC >= fc_cutoff & P_Value < pvalue_cutoff, "Down", "None")))
    result$threshold <- factor(result$threshold, levels = c("Up", "Down", "None"))
    
    title <- paste0('Cutoff for logFC is ',fc_cutoff,
                    '\nThe number of up peak is ', sum(result$threshold == "Up"),
                    '\nThe number of down peak is ', sum(result$threshold == "Down"))
    
    ggplot(data = result, aes(x = logFC, y = -log10(P_Value))) +
        geom_point(aes(colour = threshold), size = 3, alpha = 0.6) +
        scale_colour_manual(values = c("Up" = "red", "Down" = "green", "None" = "grey")) +
        theme(legend.position = "none") +
        theme_minimal() +
        ggtitle(title) +
        xlab("Log2 (fold change)") + ylab("-log10 p-value") +
        guides(color = guide_legend(' ', label.position = 'right'))
    
    ggsave(paste0(resultdir, "/diffcodon_volcano_", condition_prefix, ".pdf"), width = 10, height = 5)
    ggsave(paste0(resultdir, "/diffcodon_volcano_", condition_prefix, ".png"), width = 10, height = 5, dpi = 90, type = "cairo")
}


plot_Boxplot <- function(merged, resultdir, sampleidslen, condition_prefix) {
    
    merged_long <- melt(merged, id.vars = "codon_aaID", variable.name = "sample", value.name = "diffCodon")
	#print(merged)
	merged_long$codon_aaID <- make.unique(merged_long$codon_aaID)
    t_col <- setNames(colorRampPalette(brewer.pal(sampleidslen,"Set1"))(length(unique(merged_long$sample ))),
                    unique(merged_long$sample )) 
  merged_long$diffCodon_log <- log2(merged_long$diffCodon+1)
  ggboxplot(merged_long, x = "sample", y = "diffCodon_log", color = "sample", palette =t_col, shape = "sample") + 
    xlab("")+ylab("Codon occupancy") + 
    theme(legend.position = "none",
          axis.text.x=element_text(size = 11, angle = 45, hjust = 1,vjust = 1)) 
	ggsave(paste0(resultdir, "/diffcodon_boxplot_", condition_prefix, ".pdf"), width = 10, height = 5)
    ggsave(paste0(resultdir, "/diffcodon_boxplot_", condition_prefix, ".png"), dpi = 90, type = "cairo")
  }

 
plot_Scatter <- function(merged, resultdir, groupAsamplenames, groupBsamplenames, condition_prefix) {
    merged_1 <- merged
    merged_1$FoldChange <- merged_1[,3] / merged_1[,2]
    merged_1 <- merged_1 %>% 
    mutate(threshold = ifelse(FoldChange >= 1.5, "Up", ifelse(FoldChange <= 0.67, "Down", "None")))
    merged_1$threshold <- factor(merged_1$threshold, levels = c("Up", "Down", "None"))
    colnames(merged_1) <- c("codon_aaID", "sample_A", "sample_B", "FoldChange", "threshold")
  
    p <- ggplot(data = merged_1, aes(x = log2(sample_A), y = log2(sample_B))) +
        geom_point(aes(colour = threshold), size = 3, alpha = 0.6) +
		scale_colour_manual(values = c("Up" = "red", "Down" = "green", "None" = "grey")) +
		theme(legend.title = element_blank(), axis.text.y = element_text(angle = 90, hjust = 0.5)) +
		labs(x = groupAsamplenames, y = groupBsamplenames) +
		guides(color = guide_legend(' ', label.position = 'right'))

	ggsave(paste0(resultdir, "/diffcodon_ggscatter_", condition_prefix, ".pdf"), plot = p, width = 5, height = 5)
	ggsave(paste0(resultdir, "/diffcodon_ggscatter_", condition_prefix, ".png"), plot = p, width = 5, height = 5, dpi = 90)
}
 
plot_heatmap <- function(merged, resultdir, condition_prefix) {
    merged_1 <- merged
    rownames(merged_1) <- merged_1$codon_aaID
    merged_1 <- merged_1[,-1]
    col_fun <- colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100)
    merged_scaled <- t(scale(t(merged_1), center = TRUE, scale = TRUE))
    tmp_p <- Heatmap(merged_scaled,
                     cluster_rows = TRUE,
                     row_dend_width = unit(0.5, "cm"),
                     column_dend_height = unit(0.5, "cm"),
                     row_dend_reorder = TRUE,
                     cluster_columns = TRUE,
                     show_column_names = TRUE,
                     column_names_side = "bottom",
                     column_names_gp = gpar(col="black", fontsize = 10, just = "left", rot = 30),
                     show_row_names = FALSE, 
                     col = col_fun,
                     heatmap_legend_param = list(
                         title = "Codon occupancy levels",
                         title_position = "leftcenter-rot",
                         direction = "vertical",
                         legend_height = unit(4, "cm")))
    
	
	pdf(paste0(resultdir,"/diffcodon_heatmap_", condition_prefix, ".pdf"),width=6,height=8)
    draw(tmp_p, heatmap_legend_side = "right")
    dev.off()
    png(paste0(resultdir,"/diffcodon_heatmap_", condition_prefix, ".png"),width=6,height=8,units = "in",res=300)
    draw(tmp_p, heatmap_legend_side = "right")
    dev.off()

}




plot_heatmap2 <- function(merged, resultdir, condition_prefix) {
    merged_1 <- merged
    rownames(merged_1) <- merged_1$codon_aaID
    merged_1 <- merged_1[,-1]
    col_fun <- colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(1000)
    merged_scaled <- t(scale(t(merged_1), center = F, scale = T))
    tmp_p <- Heatmap(merged_scaled,
                     cluster_rows = TRUE,
                     row_dend_width = unit(0.5, "cm"),
                     column_dend_height = unit(0.5, "cm"),
                     row_dend_reorder = TRUE,
                     cluster_columns = TRUE,
                     show_column_names = TRUE,
                     column_names_side = "bottom",
                     column_names_gp = gpar(col="black", fontsize = 10, just = "left", rot = 30),
                     show_row_names = FALSE,
                     col = col_fun,
                     heatmap_legend_param = list(
                         title = "Codon occupancy levels",
                         title_position = "leftcenter-rot",
                         direction = "vertical",
                         legend_height = unit(4, "cm")))
	
	pdf(paste0(resultdir,"/diffcodon_heatmap_", condition_prefix, ".pdf"),width=6,height=8)
    draw(tmp_p, heatmap_legend_side = "right")
    dev.off()
    png(paste0(resultdir,"/diffcodon_heatmap_", condition_prefix, ".png"),width=6,height=8,units = "in",res=300)
    draw(tmp_p, heatmap_legend_side = "right")
    dev.off()
	
}

if (sampleidslen > 2) {
    get_log_status("Contrast matrix ...", 30)
    library("limma")
   
    process_column(merged1, CodonsampleNames, column_a, groupAsamplenames, groupBsamplenames, resultdir, fc_cutoff, pvalue_cutoff)
    process_column(merged2, CodonsampleNames, column_b, groupAsamplenames, groupBsamplenames, resultdir, fc_cutoff, pvalue_cutoff)
    
    result_a <- fread(paste0(resultdir, "/diffcodon_level_",column_a,".txt"))
    result_b <- fread(paste0(resultdir, "/diffcodon_level_",column_b,".txt"))
    
	get_log_status("volcano creating ...", 40)
    plot_volcano(result_a, fc_cutoff, pvalue_cutoff, resultdir, column_a)
    plot_volcano(result_b, fc_cutoff, pvalue_cutoff, resultdir, column_b)
	
	get_log_status("Boxplot creating ...", 50)
	plot_Boxplot(merged1, resultdir, sampleidslen, column_a)
    plot_Boxplot(merged2, resultdir, sampleidslen, column_b)
	
	get_log_status("heatmap creating ...", 80)
	plot_heatmap(merged1, resultdir, column_a)
    plot_heatmap(merged2, resultdir, column_b)
	
	get_log_status("Mean and foldchange stat ...", 100)
	merged1[,mean_A1:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupAsamplenames]
	merged1[,mean_A2:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupBsamplenames]
	merged1[, log2FoldChange := log2(mean_A2/mean_A1)]
	write.table(merged1, file=paste0(resultdir,"/mean_logfc_",column_a,".txt"),row.names=TRUE)
	merged_1 <- cbind(merged1[,1],round(merged1[,-1],digits=3))
	jsonlite::write_json(merged_1, path=paste0(resultdir,"/mean_logfc_",column_a,".json"))
	
	merged2[,mean_A1:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupAsamplenames]
	merged2[,mean_A2:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupBsamplenames]
	merged2[, log2FoldChange := log2(mean_A2/mean_A1)]
	write.table(merged2, file=paste0(resultdir,"/mean_logfc_",column_b,".txt"),row.names=TRUE)
	merged_2 <- cbind(merged2[,1],round(merged1[,-1],digits=3))
	jsonlite::write_json(merged_2, path=paste0(resultdir,"/mean_logfc_",column_b,".json"))

}else if(sampleidslen==2){
	get_log_status("Boxplot creating ...", 30)
	plot_Boxplot(merged1, resultdir, sampleidslen, column_a)
    plot_Boxplot(merged2, resultdir, sampleidslen, column_b)
  
    get_log_status("Scatter plot creating ...", 70)
	plot_Scatter(merged1, resultdir, groupAsamplenames, groupBsamplenames, column_a)
    plot_Scatter(merged2, resultdir, groupAsamplenames, groupBsamplenames, column_b)
  
    get_log_status("Heatmap creating ...", 90)
	plot_heatmap2(merged1, resultdir, column_a)
    plot_heatmap2(merged2, resultdir, column_b)
	
		
	get_log_status("Mean and foldchange stat ...", 100)
	merged_1 <- merged1
	merged_1$log2FoldChange <- log2(merged_1[,3]/merged_1[,2])
	write.table(merged_1, file=paste0(resultdir,"/mean_logfc_",column_a,".txt"),row.names=TRUE)
	merged_2 <- cbind(merged_1[,1],round(merged_1[,-1],digits=3))
	jsonlite::write_json(merged_2, path=paste0(resultdir,"/mean_logfc_",column_a,".json"))

	merged_3 <- merged2
	merged_3$log2FoldChange <- log2(merged_3[,3]/merged_3[,2])
	write.table(merged_3, file=paste0(resultdir,"/mean_logfc_",column_b,".txt"),row.names=TRUE)
	merged_4 <- cbind(merged_3[,1],round(merged_3[,-1],digits=3))
	jsonlite::write_json(merged_4, path=paste0(resultdir,"/mean_logfc_",column_b,".json"))
}
	
pdfFiles <- list.files(resultdir, pattern = "*.pdf$", full.names = T, recursive=T)
for(i in 1:length(pdfFiles)) {
    pdfFile <- pdfFiles[i]
    outHtmlFile <- sub(".pdf",".html",pdfFile)
	html <- paste0("<style>\n.pdfobject-container {\n		width: 100%;\n		max-width: 1000px;\n		height: 1000px;\n		margin: 2em 0;\n}\n.pdfobject { border: solid 1px #666; }\n</style>\n<script src='./assets/global/plugins/pdfobject.min.js'></script>\n<div id='pdf' class='pdfobject-container'></div>\n<script>\nvar options = {\n	pdfOpenParams: {\n		pagemode: 'thumbs',\n		navpanes: 0,\n		toolbar: 0,\n		statusbar: 0,\n		view: 'FitV'\n	}\n};\nvar myPDF = PDFObject.embed('",pdfFile,"', '#pdf', options);\n</script>\n")
	writeLines(html,outHtmlFile)
}


