library(data.table)
library(edgeR)
countTable <- fread("count.txt", head=T)
glength <- countTable$Length
countTable[, Length := NULL]
countTable <- as.data.frame(countTable)
duplicated_rows <- countTable[duplicated(countTable[, 1]), 1]
print(duplicated_rows)
unique_names <- make.unique(as.character(countTable[, 1]))
rownames(countTable) <- unique_names
count_df <- countTable[6]
y <- DGEList(counts=count_df,genes=data.frame(Length=glength))  
y <- calcNormFactors(y)
normalcount <- rpkm(y)
data.filter <- normalcount[rowSums(normalcount)>0,]
write.table(normalcount, file = " RPKM.txt", sep = "\t", quote = FALSE, row.names = TRUE)
exp_input_table <- read.table("RNAt_RPKM.txt",header=T,row.names=1,sep='\t')
exp_ribo_table <- read.table("Ribo_RPKM.txt",header=T,row.names=1,sep='\t')
common_rows <- intersect(rownames(exp_input_table), rownames(exp_ribo_table))
exp_input_table_common <- exp_input_table[common_rows, ]
exp_ribo_table_common <- exp_ribo_table[common_rows, ]
TE <- exp_ribo_table_common / exp_input_table_common
TE <- as.matrix(TE)
ncol(TE)
new_data <- cbind(common_rows,TE)
TE_filter <- new_data[is.finite(TE) & !is.nan(TE) & new_data[, 2] != 0, ]
write.table(TE_filter, file = " TE.txt", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)