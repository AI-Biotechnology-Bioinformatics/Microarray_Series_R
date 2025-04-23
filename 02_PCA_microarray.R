# Install and load necessary libraries
install.packages("caret")
install.packages("factoextra")

library(caret)      # For model training and evaluation
library(factoextra) # For PCA visualization

# ---------------------------------
# Load and Preprocess Data
# ----------------------------------
# Import dataset 
# already normalized microarray expression data 
# Here's the link to access the dataset: https://github.com/AI-Biotechnology-Bioinformatics/Microarray_Series_R

data <- read.csv("gastric_cancer.csv", row.names = 1)

# Ensure genes are in columns (features) and samples are in rows
str(head(data)[1:5])
dim(data)
View(data)

# Remove empty rows
data <- data[rowSums(data == "") == 0, ]

# Check and remove NA values
sum_na <- sum(is.na(data))
cat("Total NA values in dataset:", sum_na, "\n")
data <- na.omit(data)

# ---------------------------------
# Perform PCA (Exclude Target and Categorical Variables)
# ----------------------------------
pca_data <- data[, -c(1,2)]  # Remove "labels" & "target" column

# Perform PCA 
pca_result <- prcomp(pca_data, scale=TRUE) # Perform PCA

# PCA Summary and Components Analysis
summary(pca_result)

# elements of PCA object
names(pca_result)

# "sdev", Standard Deviation of components
pca_result$sdev

# "rotation" (eigenvectors),provides "loadings" 
# which show how much each original variable 
# contributes to the principal components
pca_result$rotation[1:10]

# "center", is mean of each variable(gene expression value)
pca_result$center

# "scale", St.deviation of variables
pca_result$scale
# "center" & "scale" is used to standardized data

# "x", Principle Component Scores of all observations(samples)
pca_result$x

# Visualizing PCA Results 
# scree plot to explain variance 
fviz_eig(pca_result, addlabels = TRUE) +
  ggtitle("Scree Plot") +
  theme_minimal()

# Individual Sample Plot (PC1 & PC2)
fviz_pca_ind(pca_result, 
             axes = c(1, 2), # PC1 & PC2 
             col.ind = data$target,  
             palette = c("blue", "red"),  
             legend.title = "Group",
             xlab = "PC1 (21%)",
             ylab = "PC2 (11.3%)",
             title = "Individual Plot (PC1 & PC2)") +
  theme_minimal()

# Biplot with Top 30 Contributing Genes
fviz_pca_biplot(pca_result, 
                axes = c(1, 2),  
                col.ind = data$target,  
                palette = c("blue", "red"),  
                repel = TRUE,  
                select.var = list(contrib = 30),  # top 30 genes
                xlab = "PC1 (21%)",
                ylab = "PC2 (11.3%)",
                title = "Biplot - PC1 and PC2") +
  theme_minimal()

## Check the name of the top 30 measurements (genes) 
# that contribute most to PC1
loading_PC1 <- pca_result$rotation[,1]
gene_scores <- abs(loading_PC1) ## get the magnitudes
ranked_genes <- sort(gene_scores, decreasing=TRUE)
top_30_genes <- names(ranked_genes[1:30])

top_30_genes ## show the names of the top 30 genes

pca_result$rotation[top_30_genes,1] ## show the scores with +ve/-ve sign)

# ---------------------------------
# Extract Top Contributing Genes from PC1
# ----------------------------------
loading_PC1 <- pca_result$rotation[,1]
gene_scores <- abs(loading_PC1) # Get absolute scores
ranked_genes <- sort(gene_scores, decreasing=TRUE)

# Select top 30, 50, and 100 genes
pc1_loadings <- data.frame(gene_name = names(ranked_genes),
                           PC1 = ranked_genes)
top30_genes  <- pc1_loadings$gene_name[1:30]
top50_genes  <- pc1_loadings$gene_name[1:50]
top100_genes <- pc1_loadings$gene_name[1:100]

# ---------------------------------
# Train and Evaluate Random Forest Models
# ----------------------------------

target <- as.factor(data$target)
# Function to train and evaluate Random Forest
train_rf_model <- function(data, genes) {
  data_subset <- data[, genes]  # Select genes
  data_subset <- cbind(target, data_subset)  # Add target column
  
  # Split data into training (70%) and test (30%) sets
  train_index <- createDataPartition(data_subset$target, p = 0.7, list = FALSE)
  train_data <- data_subset[train_index, ]
  test_data <- data_subset[-train_index, ]
  
  train_target <- as.factor(train_data$target)
  train_data <- train_data[,-1]  # Remove target column
  test_target <- as.factor(test_data$target)
  test_data <- test_data[,-1]  # Remove target column
  
  # Train Random Forest model with 10-fold cross-validation
  ctrl <- trainControl(method = "cv",
                       number = 10, 
                       verboseIter = TRUE)
  rf_model <- train(x = train_data, 
                    y = train_target, 
                    method = "rf", 
                    trControl = ctrl)
  
  # Evaluate model
  predictions <- predict(rf_model, test_data)
  confusionMatrix(predictions, test_target)
}

# Train models using top 30, 50, 100, and all genes
conf_top30  <- train_rf_model(data, top30_genes)
conf_top50  <- train_rf_model(data, top50_genes)
conf_top100 <- train_rf_model(data, top100_genes)
# Original data with all genes
conf_all <- train_rf_model(data, colnames(data)[-c(1,2)])  # Use all genes

# ---------------------------------
# Compare Model Performance
# ----------------------------------
comparison <- data.frame(
  Model = c("Top 30 Genes", 
            "Top 50 Genes", 
            "Top 100 Genes", 
            "All Genes"),
  Accuracy = c(conf_top30$overall["Accuracy"],
               conf_top50$overall["Accuracy"],
               conf_top100$overall["Accuracy"],
               conf_all$overall["Accuracy"]))
print(comparison)
