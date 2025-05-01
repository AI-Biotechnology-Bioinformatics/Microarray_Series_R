# ===================================================================
#                  AI and Biotechnology/Bioinformatics        
# ===================================================================
# -------------------------------------------------------------------
#                    Microarray Analysis – Part 5
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#            Feature Selection with "Boruta" wrapper method          
# -------------------------------------------------------------------
# ===================================================================

# Load necessary libraries 
library(DataExplorer)  # For quick data exploration and visualization
library(caret)         # For machine learning tasks like data splitting and model training
library(Boruta)        # For feature selection using the Boruta algorithm

# Exploratory data analysis (EDA)

# View the structure of the first 6 columns of the dataset
head(str(data[1:6]))  # Helps understand the data types and layout of initial columns

# Create an introductory plot to summarize dataset features (e.g., types, missing values)
plot_intro(data)

# Prepare the dataset
genes <- data[, -c(1:3)]        # Exclude first 3 columns (non-gene info like row names, ID, sample labels)
target <- data$target           # Extract target variable-numeric factor (1 = cancer, 0 = normal) 

# Split data into training and testing sets (80% train, 20% test)
index <- createDataPartition(target,
                             p = 0.8,
                             list = FALSE)  # Stratified sampling to maintain class distribution

train_genes <- genes[index, ]   # Training features
test_genes <- genes[-index, ]   # Testing features

train_target <- target[index]   # Training labels
test_target <- target[-index]   # Testing labels

# Feature selection using Boruta algorithm on the training data only
boruta <- Boruta(x = train_genes, 
                 y = train_target)  # Identify important features based on random forest importance

# Print summary of Boruta feature selection process
print(boruta)

# Finalize feature selection by resolving tentative features (neither confirmed nor rejected)
boruta_ten <- TentativeRoughFix(boruta)  # Improves feature stability by resolving tentative decisions
print(boruta_ten)

# Extract names of top selected genes (important features only)
top_genes <- getSelectedAttributes(boruta_ten, withTentative = FALSE )

# Subset the training and testing data using only the top selected features
subset_train <- train_genes[, top_genes]
subset_test <- test_genes[, top_genes]

# Set up training control parameters for cross-validation
ctrl <- trainControl(method = "cv",     # Use cross-validation to evaluate model performance
                     number = 3,        # 3-fold cross-validation
                     verboseIter = TRUE) # Print training progress

# Train a Random Forest model using selected features
RF_model <- train(x = subset_train,
                  y = train_target,
                  method = "rf",        # Random forest classifier
                  trControl = ctrl)     # Apply cross-validation control

# Predict labels on the test set using the trained model
pred <- predict(RF_model, subset_test)

# Evaluate model performance using a confusion matrix
conf <- confusionMatrix(pred, test_target)

# Print the confusion matrix (includes accuracy, sensitivity, specificity, etc.)
print(conf)

# follow for more:
# github: https://github.com/AI-Biotechnology-Bioinformatics
# linkedin: https://www.linkedin.com/company/ai-and-biotechnology-bioinformatics/