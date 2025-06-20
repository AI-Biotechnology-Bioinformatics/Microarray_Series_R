#### Install and Load Required Packages ####
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c("GEOquery","affy","limma", "AnnotationDbi", "hgu133plus2.db"))
install.packages("tidyverse")
install.packages("caret")
install.packages("FSelectorRcpp")

#Packages need to be installed once but must be loaded in each new R session
# to use their functions 

library(GEOquery)     # For downloading microarray data
library(affy)         # For reading and normalizing CEL files
library(limma)        # For differential expression analysis
library(AnnotationDbi)# For probe ID to gene symbol conversion
library(hgu133plus2.db) # Annotation package for your platform
library(caret)        # For machine learning and cross-validation
library(FSelectorRcpp)    # For information gain-based feature selection

##### Download Raw Data (CEL files) from NCBI GEO ####
# CEL files are large, and downloads may fail even with a good connection. 
#It's recommended to download raw data directly from NCBI GEO

# skip this step if you already downloaded data from NCBI
getGEOSuppFiles("GSE79973") # Download supplementary files

# Untar downloaded Files
untar("GSE79973_RAW.tar", exdir = "data/") # Extract CEL files into 'data/' folder

# Read CEL files into R
raw_data <- ReadAffy(celfile.path = "data/")        

# Perform RMA Normalization
normalized_data <- rma(raw_data)

# Extract normalized expression data
normalized_expr <- exprs(normalized_data)  

# Download series matrix files to access phenotype and feature data
gse_data <- getGEO("GSE79973", GSEMatrix = TRUE)

# Extract phenotype data (sample metadata)
phenotype_data <- pData(gse_data$GSE79973_series_matrix.txt.gz)  

# Prepare Phenotype Data/target column
# Convert "gastric adenocarcinoma" to "cancer" and "gastric mucosa" to "normal"

phenotype_data$labels <- factor( # column with name "labels will created
  phenotype_data$source_name_ch1, 
  levels = c("gastric adenocarcinoma", "gastric mucosa"),
  labels = c("cancer", "normal")
)

# Perform Differential Expression Analysis with limma

# Create design matrix
group <- factor(phenotype_data$labels)
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)  # Ensure column names are "Cancer" and "Normal"

# Fit linear model
fit_1 <- lmFit(normalized_expr, design)

# Define contrasts (Cancer vs Normal)
contrast_matrix <- makeContrasts(Cancer_vs_Normal = cancer - normal, levels = design)

# Apply empirical Bayes moderation
fit_contrast <- contrasts.fit(fit_1, contrast_matrix)
fit_2 <- eBayes(fit_contrast) 

# Extract top DEGs
deg_results <- topTable(fit_2, 
                        coef = "Cancer_vs_Normal",
                        number = Inf, 
                        adjust.method = "BH")

# Summary of results
summary(decideTests(fit_2, lfc = 1))

# Identify significant genes

# Add a column for significance
deg_results$Significant <- ifelse(deg_results$adj.P.Val < 0.05 & abs(deg_results$logFC) > 1, "yes", "no")

# View the results
head(deg_results)

# Visualize Significant vs. Non-Significant Genes

# Create volcano plot
ggplot(deg_results, aes(x = logFC, y = -log10(adj.P.Val), color = Significant)) +
  geom_point(aes(color = Significant), size = 2, alpha = 0.6) +
  scale_color_manual(values = c("yes" = "red", "no" = "gray")) +
  theme_minimal() +
  labs(
    title = "Volcano Plot: Cancer vs Normal",
    x = "log2 Fold Change (logFC)",
    y = "-log10(Adjusted p-value)"
  ) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +  # p-value threshold
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "blue")  # logFC thresholds

# Heatmap of Significant Genes
# Extract expression data for significant genes
significant_genes <- deg_results[deg_results$Significant == "yes", ]
significant_expr <- normalized_expr[rownames(significant_genes), ]

# Create heatmap
pheatmap(significant_expr, scale = "row", clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean", clustering_method = "complete",
         show_rownames = FALSE, main = "Heatmap of Significant Genes")

# Identify upregulated and downregulated genes

# Add columns for up-regulated and down-regulated genes
deg_results$Regulation <- ifelse(
  deg_results$logFC > 1 & deg_results$adj.P.Val < 0.05, "Up",
  ifelse(deg_results$logFC < -1 & deg_results$adj.P.Val < 0.05, "Down", "NotSig")
)

# View the updated results
head(deg_results)

# Create volcano plot with up/down regulation
ggplot(deg_results, aes(x = logFC, y = -log10(adj.P.Val), color = Regulation)) +
  geom_point(aes(color = Regulation), size = 2, alpha = 0.6) +
  scale_color_manual(values = c("Up" = "red", "Down" = "blue", "NotSig" = "gray")) +
  theme_minimal() +
  labs(
    title = "Volcano Plot: Cancer vs Normal",
    x = "log2 Fold Change (logFC)",
    y = "-log10(Adjusted p-value)"
  ) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +  # p-value threshold
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black")  # logFC thresholds

# heatmap 

# Extract expression data for up-regulated and down-regulated genes
up_genes <- deg_results[deg_results$Regulation == "Up", ]
down_genes <- deg_results[deg_results$Regulation == "Down", ]

up_expr <- normalized_expr[rownames(up_genes), ]
down_expr <- normalized_expr[rownames(down_genes), ]

# Create heatmap for up-regulated genes
pheatmap(up_expr, scale = "row", clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean", clustering_method = "complete",
         show_rownames = FALSE, main = "Heatmap of Up-Regulated Genes")

# Create heatmap for down-regulated genes
pheatmap(down_expr, scale = "row", clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean", clustering_method = "complete",
         show_rownames = FALSE, main = "Heatmap of Down-Regulated Genes")

### Feature Annotation to gene IDS
# By using linear model, we identified differentially expressed probesets between cancer and normal tissues.
# For further downstream analysis of microarray data,
# we need to map these probe IDs to gene symbols. 
# This mapping helps to interpret the results in the context of genes rather than probe IDs.

# Map probe IDs with gene symbols
deg_results <-  deg_results %>%
  rownames_to_column(var = "ID") %>%
  inner_join(feature_data, ., by = "ID")

############### Session II: Machine Learning ########

# Prepare data for machine learning tasks

# Encode target variable (group) 0 = normal, 1 = cancer
print(group)

target <- as.factor(ifelse(group == "normal", 0, 1))
print(target)
class(target)

# For machine learning, features should be in columns and samples in rows.
# Our data has genes as rows and samples as columns, 
# so we need to transpose it for compatibility with ML algorithms." 

data <- as.data.frame(t(normalized_expr))
str(data)

# Bind target column with data
data <- cbind(target, data)

# Split data into train (70%) and test (30%) sets
set.seed(123)  # For reproducibility

index <- createDataPartition(data$target, p = 0.7, list = FALSE)
train <- data[index, ]
test <- data[-index, ]

# Perform information gain-based feature selection
genes <- data[, -1] # remove target column only features
target <- data$target # only target column

info_gain <- information_gain(x = genes,
                              y = target,
                              type = "infogain")

# Information gain scores range from 0 to 1
# values closer to 1 indicate higher importance

# remove features with zero importance
sorted_features <- info_gain[info_gain$importance > 0,]

# rearrange features
sorted_features <- sorted_features[order(sorted_features$importance, decreasing = TRUE),]

# Extract column from train and test set based on information gain features
train_genes <- train[, sorted_features$attributes]
train_target <- as.factor(train$target)

test_genes <- test[, sorted_features$attributes]
test_target <- as.factor(test$target)

# Set training parameter
ctrl <- trainControl(method = "cv", # 10-fold cross-validation
                     number = 10,
                     verboseIter = TRUE) 
# Model training
Rf_model <- train(x = train_genes,
                  y = train_target,
                  method = "rf", 
                  trControl = ctrl)
Rf_model@
  

# Model Predictions
  
# Predict on train data
train_predictions <- predict(Rf_model, train_genes)

# compare outcome of train_prediction & train_target
print(train_predictions)
print(train_target)

# Predict on test data
test_predictions <- predict(Rf_model, test_genes)

# compare outcome of test_prediction & test_target
print(test_predictions)
print(test_target)

# compare model prediction results on train & test set

# Confusion matrix and performance metrics
conf_train <- confusionMatrix(train_predictions, train_target)
print(conf_train)

conf_test <- confusionMatrix(test_predictions, test_target)
print(conf_test)

# Save Results

# Save DEG results
write.csv(deg_results, file = "differential_expression_results.csv", row.names = FALSE)

# Save top features based on information gain
write.csv(sorted_features, file = "top_features_information_gain.csv", row.names = FALSE)

# Save random forest model
saveRDS(rf_model, file = "random_forest_model.rds")
