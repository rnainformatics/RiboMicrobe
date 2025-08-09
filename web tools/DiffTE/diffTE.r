suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(getopt))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(ggpubr))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggthemes))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(Cairo))
suppressPackageStartupMessages(library(circlize))
#library(gridExtra)
suppressPackageStartupMessages(library(ggplotify))

options(bitmapType = 'cairo')

spec <- matrix(
c("jobid", "j", 2, "character", "This is output dir name!",
  "species",  "s", 2, "character", "This is species!",
  "samplenames", "n", 2, "character",  "This is samplenames!",
  # "method",  "m", 2, "character",  "This is method!",
  "fc_cutoff",  "fc", 2, "character",  "This is fc_cutoff!",
  "pvalue_cutoff",  "p", 2, "character",  "This is pvalue_cutoff!",
  "email",  "e", 2, "character",  "This is email!",
  "help",   "h", 0, "logical",  "This is Help!"),
   byrow=TRUE, ncol=5)
opt <- getopt(spec=spec)
if( !is.null(opt$help) || is.null(opt$species) || is.null(opt$samplenames) || is.null(opt$jobid) || is.null(opt$fc_cutoff) || is.null(opt$pvalue_cutoff) || is.null(opt$email)){
    cat(paste(getopt(spec=spec, usage = T), "\n"))
    quit()
}
species <- opt$species
samplenames <- opt$samplenames
# method <- opt$method
jobid <- opt$jobid
fc_cutoff <- opt$fc_cutoff
pvalue_cutoff <- opt$pvalue_cutoff
resultdir <- paste0("./data/",jobid)
dir.create(resultdir)
get_log_status <- function(messages, status){
        logdate <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
        messages <- gsub("\r?\n|\r", "", messages)
        messages <- gsub("\"", "'", messages)
        current <- paste(c(10,20,30,40,50,60,70,80,90,100),collapse =" ")
        writeLines(paste0('{"species":','"',species,'",','"jobid":','"',jobid,'",','"message":','"',messages,'",','"current":','"',status,'",','"sample":','"',samplenames,'"}'), paste0(resultdir,"/status.json"))
        if(status != "Error") {
                messages <- paste0(messages,"\t",logdate,"\n")
        }
        message(messages)
    write(messages, file=paste0(resultdir,"/log.txt"), append=TRUE, sep="\n")
}

grouplabels <- unlist(strsplit(samplenames, "[#]"))
groupAsamplenames <- unlist(strsplit(grouplabels[1], "[;]"))
groupBsamplenames <- unlist(strsplit(grouplabels[2], "[;]"))
typeList <- list()
TEsampleNames <- c()
for(i in 1:length(groupAsamplenames)) {
    groupAsamplename <- groupAsamplenames[i]
    infile <- paste0("./data/",species,"/",groupAsamplename,".TE.txt")
    message(groupAsamplename)
    peak = fread(infile,head=F)
    peak[,peak_id:=paste0(V1)]
    peak <- peak[,.(peak_id,V2)]
        groupAsamplename <- gsub("-","_",groupAsamplename)
    setnames(peak,"V2",groupAsamplename)
    typeList[[groupAsamplename]] <- peak
        TEsampleNames <- c(TEsampleNames,groupAsamplename)
}
for(i in 1:length(groupBsamplenames)) {
    groupBsamplename <- groupBsamplenames[i]
    infile <- paste0("./data/",species,"/",groupBsamplename,".TE.txt")
    message(groupBsamplename)
    peak = fread(infile,head=F)
    peak[,peak_id:=paste0(V1)]
    peak <- peak[,.(peak_id,V2)]
        groupBsamplename <- gsub("-","_",groupBsamplename)
    setnames(peak,"V2",groupBsamplename)
    typeList[[groupBsamplename]] <- peak
        TEsampleNames <- c(TEsampleNames,groupBsamplename)
}
groupAsamplenames <- gsub("-","_",groupAsamplenames)
groupBsamplenames <- gsub("-","_",groupBsamplenames)
typeList <- lapply(typeList, function(i) setkey(i, peak_id))
merged <- Reduce(function(...) merge(..., all = T), typeList)
merged <- na.omit(merged)
sampleidslen <- length(TEsampleNames)
#save(merged,file="merged.RData")
if (sampleidslen >2){
  ### limma ###
  get_log_status("Contrast matrix ...", 30)
  library("limma")
  data <- as.data.frame(merged)
  rownames(data) <- data[,1]
  data <- data[,-1]
  table2 <- data.frame(name = TEsampleNames, condition = c(rep("CTRL",length(groupAsamplenames)), rep("KO",length(groupBsamplenames))))
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
  nrDEG$peak_id=row.names(nrDEG)
  data$peak_id=row.names(data)
  result = merge(data,nrDEG,by="peak_id",all.x=T)
  #save(result,file="result.Data")
  result_sub <- subset(result, abs(logFC)>=fc_cutoff & P.Value<pvalue_cutoff)
  xx = c("AveExpr","t","B")
  result_sub <- result_sub[,!names(result_sub) %in% xx]
  colnames(result_sub) <- gsub("\\.","_",colnames(result_sub))
  write.table(result_sub,file=paste0(resultdir,"/diffTE.level.txt"))
  result_sub_1 <- cbind(result_sub[,1],round(result_sub[,-1],digits=3))
  colnames(result_sub_1) <- colnames(result_sub)
  jsonlite::write_json(result_sub_1, path=paste0(resultdir,"/diffTE.level.json"))
  
  ###Volcano Plot ###
 get_log_status("Volcano Plot creating ...", 40)
 class(fc_cutoff)
 result <- result %>% mutate(threshold = ifelse(logFC >= fc_cutoff & P.Value < pvalue_cutoff, "Up", ifelse(-logFC >= fc_cutoff & P.Value < pvalue_cutoff, "Down", "None")))
 result$threshold <- factor(result$threshold,levels=c("Up","Down","None"))
 title <- paste0('Cutoff for logFC is ',fc_cutoff,
                '\nThe number of up peak is ',sum(result$threshold == "Up"),
                '\nThe number of down peak is ',sum(result$threshold == "Down"))
 #plot
 ggplot(data=result, aes(x=logFC, y=-log10(P.Value)))+
 geom_point(aes(colour = threshold),size = 3, alpha = 0.6)+
 scale_colour_manual(values = c("Up" = "red", "Down" = "green", "None" = "grey"))+
 theme(legend.position = "none")+
 theme_base() + ggtitle(title)+
 xlab("Log2 (fold change)") + ylab("-log10 p-value")+
 guides(color=guide_legend(' ',label.position = 'right'))
 ggsave(paste0(resultdir,"/diffTE.Volcano.pdf"), width=10, height=5)
 ggsave(paste0(resultdir,"/diffTE.Volcano.png"), width=10, height=5, dpi=90, type = "cairo")

  ###Boxplot ###
  get_log_status("Boxplot creating ...", 50)
  merged_long <- melt(merged, id.vars = "peak_id", variable.name = "sample", value.name = "diffTE")
  t_col <- setNames(colorRampPalette(brewer.pal(sampleidslen,"Set1"))(length(unique(merged_long$sample ))),
                    unique(merged_long$sample )) 
  merged_long$diffTE_log <- log2(merged_long$diffTE+1)
  ggboxplot(merged_long, x = "sample", y = "diffTE_log", color = "sample", palette =t_col, shape = "sample") + 
    xlab("")+ylab("Translation efficiency") + 
    theme(legend.position = "none",
          axis.text.x=element_text(size = 11, angle = 45, hjust = 1,vjust = 1))   
  ggsave(paste0(resultdir,"/diffTE.boxplot.pdf"))
  ggsave(paste0(resultdir,"/diffTE.boxplot.png"), dpi=90, type = "cairo")
  
 
  get_log_status("Pheatmap creating ...", 80)
  merged_1 <- merged
  rownames(merged_1) <- merged_1$peak_id
  merged_1 <- merged_1[,-1]
  col_fun = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100)
  merged_scaled <- t(scale(t(merged_1), center = T, scale=T))
  tmp_p <- Heatmap(merged_scaled,
                cluster_rows = TRUE,
                row_dend_width = unit(0.5, "cm"),
                column_dend_height = unit(0.5, "cm"),
                row_dend_reorder = TRUE,
                cluster_columns = TRUE,
                show_column_names = TRUE,
                column_names_side = "bottom",
                column_names_gp = gpar(col="black",fontsize = 10,just="left",rot=30),
                show_row_names = FALSE, 
                col = col_fun,
                heatmap_legend_param = list(
                  title = "Translation efficiency levels",
                  title_position = "leftcenter-rot",
                  #at = c(-1,  1),
                  #labels = c("Low","High"),
                  direction = "vertical", #
                  legend_height = unit(4, "cm")))


  pdf(paste0(resultdir,"/pheatmap_diffTE.pdf"),width=6,height=8)
  draw(tmp_p, heatmap_legend_side = "right")
  dev.off()
  png(paste0(resultdir,"/pheatmap_diffTE.png"),width=6,height=8,units = "in",res=180)
  draw(tmp_p, heatmap_legend_side = "right")
  dev.off()
  
  
  
  get_log_status("Mean and foldchange stat ...", 100)
  merged[,mean_A1:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupAsamplenames]
  merged[,mean_A2:=rowMeans(.SD, na.rm = TRUE),.SDcols = groupBsamplenames]
  merged[, log2FoldChange := log2(mean_A2/mean_A1)]
  write.table(merged, file=paste0(resultdir,"/mean_logfc.txt"),row.names=TRUE)
  merged_1 <- cbind(merged[,1],round(merged[,-1],digits=3))
  jsonlite::write_json(merged_1, path=paste0(resultdir,"/mean_logfc.json"))
}else if(sampleidslen==2){
  ###boxplot###
  get_log_status("boxplot creating ...", 30)
  merged_long <- melt(merged, id.vars = "peak_id", variable.name = "sample", value.name = "diffTE")
  t_col <- setNames(colorRampPalette(brewer.pal(3,"Set1"))(length(unique(merged_long$sample ))),
                    unique(merged_long$sample )) 
  merged_long$diffTE_log <- log2(merged_long$diffTE+1)
  ggboxplot(merged_long, x = "sample", y = "diffTE_log", color = "sample", palette =t_col, shape = "sample") + 
    xlab("")+ylab("Translation efficiency") + 
    theme(legend.position = "none",
          axis.text.x=element_text(size = 11, angle = 45, hjust = 1,vjust = 1)) 
  ggsave(paste0(resultdir,"/diffTE.boxplot.pdf"))
  ggsave(paste0(resultdir,"/diffTE.boxplot.png"), dpi=90, type = "cairo")
  
  ###Scatter plot ###
  get_log_status("Scatter plot creating ...", 70)
  merged_1 <- merged
  merged_1$FoldChange <- merged_1[,3]/merged_1[,2]
  merged_1 <- merged_1 %>% mutate(threshold = ifelse(FoldChange >=1.5, "Up", ifelse(FoldChange <=0.67, "Down", "None")))
  merged_1$threshold <- factor(merged_1$threshold,levels=c("Up","Down","None"))
  name_x <- colnames(merged_1)[2]
  name_y <- colnames(merged_1)[3]
  colnames(merged_1) <- c("peak_id", "sample_A", "sample_B", "FoldChange", "threshold")
  ggplot(data=merged_1, aes(x=log2(sample_A), y=log2(sample_B)))+
    geom_point(aes(colour = threshold),size = 3, alpha = 0.6)+
    scale_colour_manual(values = c("Up" = "red", "Down" = "green", "None" = "grey"))+
    theme(legend.title = element_blank(),
          axis.text.y = element_text(angle = 90,hjust = 0.5))+
    theme_base() + 
    labs(x=name_x, y=name_y)+
    guides(color=guide_legend(' ',label.position = 'right'))
     
  #ggscatter(merged_1, x = colnames(merged_1)[2] , y = colnames(merged_1)[3],
  #          color = "threshold",palette = c("Up" = "red", "Down" = "green", "None" = "grey"),
  #          size = 3, alpha = 0.6, yscale = "log2", xscale = "log2")+ border()+
  #      guides(color=guide_legend(' ',label.position = 'right'))

  ggsave(paste0(resultdir,"/diffTE.ggscatter.pdf"), width=5, height=5)
  ggsave(paste0(resultdir,"/diffTE.ggscatter.png"), width=5, height=5, dpi=90, type = "cairo")

  ###pheatmap###
  get_log_status("Pheatmap creating ...", 90)
  merged_1 <- merged
  rownames(merged_1) <- merged_1$peak_id
  merged_1 <- merged_1[,-1]
  col_fun = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(1000)
  merged_scaled <- t(scale(t(merged_1), center = F, scale=T))
  tmp_p <- Heatmap(merged_scaled,
               cluster_rows = TRUE,
               row_dend_width = unit(0.5, "cm"),
               column_dend_height = unit(0.5, "cm"),
               row_dend_reorder = TRUE,
               cluster_columns = TRUE,
               show_column_names = TRUE,
               column_names_side = "bottom",
               column_names_gp = gpar(col="black",fontsize = 10,just="left",rot=30),
               show_row_names = FALSE,
               col = col_fun,
               heatmap_legend_param = list(
                  title = "Translation efficiency levels",
                  title_position = "leftcenter-rot",
                  #at = c(-1,  1),
                  #labels = c("Low","High"),
                  direction = "vertical", #
                  legend_height = unit(4, "cm")))
  pdf(paste0(resultdir,"/pheatmap_diffTE.pdf"),width=6,height=8)
 
  draw(tmp_p, heatmap_legend_side = "right")
  while (!is.null(dev.list()))  dev.off()
  png(paste0(resultdir,"/pheatmap_diffTE.png"),width=6,height=8,units = "in",res=180)
  
  draw(tmp_p, heatmap_legend_side = "right")
  while (!is.null(dev.list()))  dev.off()
  

  ###pheatmap completed
  ### mean log2FoldChange ###
  get_log_status("Mean and foldchange stat ...", 100)
  merged_1 <- merged
  merged_1$log2FoldChange <- log2(merged_1[,3]/merged_1[,2])
  write.table(merged_1, file=paste0(resultdir,"/mean_logfc.txt"),row.names=TRUE)
  merged_2 <- cbind(merged_1[,1],round(merged_1[,-1],digits=3))
  jsonlite::write_json(merged_2, path=paste0(resultdir,"/mean_logfc.json"))

}


pdfFiles <- list.files(resultdir, pattern = "*.pdf$", full.names = T, recursive=T)
for(i in 1:length(pdfFiles)) {
    pdfFile <- pdfFiles[i]
    outHtmlFile <- sub(".pdf",".html",pdfFile)
	html <- paste0("<style>\n.pdfobject-container {\n		width: 100%;\n		max-width: 1000px;\n		height: 1000px;\n		margin: 2em 0;\n}\n.pdfobject { border: solid 1px #666; }\n</style>\n<script src='./assets/global/plugins/pdfobject.min.js'></script>\n<div id='pdf' class='pdfobject-container'></div>\n<script>\nvar options = {\n	pdfOpenParams: {\n		pagemode: 'thumbs',\n		navpanes: 0,\n		toolbar: 0,\n		statusbar: 0,\n		view: 'FitV'\n	}\n};\nvar myPDF = PDFObject.embed('",pdfFile,"', '#pdf', options);\n</script>\n")
	writeLines(html,outHtmlFile)
}

