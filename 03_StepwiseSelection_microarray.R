# ===================================================================
#            AI in Biotechnology & Bioinformatics Series         
# ===================================================================
# -------------------------------------------------------------------
#                    Microarray Analysis – Part 3
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Forward, Backward, and Stepwise Selection using Logistic Regression           
# -------------------------------------------------------------------
# ===================================================================

#-----------------------------------------
#### STEP 1: Load and explore Dataset ####
#-----------------------------------------
# already normalized microarray expression data
# check data structure and dimensions

# load Required Packages
library(caret)
library(dplyr)
library(MASS)

# load dataset 
data <- read.csv("gastric_cancer.csv")
dim(data)

# samples in rows
# gene + target variable in column

head(str(data[1:6]))

#----------------------------------------
##### Step 2: Clean and prepare data ####
#----------------------------------------
# remove unnecessary columns and 
# Select top 100 genes based on variance
# Check correaltion within features

View(data)
# subset gene and target columns
genes <- data[, -c(1:3)] # remove first 3 columns 
class(genes) # must be data frame
target <- as.factor(data$target)  # 1 = cancer, 0 = normal tissues
class(target)

#---------------------------------
#### Variance and Correlation #### 
#---------------------------------

# Note: These steps are optional for smaller datasets 
# — added here just for understanding the method, not as mandatory steps
# If your dataset has a moderate number of features, you can skip these steps

# now select top 100 genes based of variance to reduce dimensionality of data

# here apply() function tell R to 
# apply var function (variance) to each column 
# 2 means columns (R apply variance function by columns because each column represents genes)
# and order(.. decreasing = TRUE), sort genes from highest to lowest variance
# [1:100] selects top 100 columns based on variance
top_var <- order(apply(genes, 2, var), decreasing = TRUE)[1:100]

# subset genes based on variance
top_genes <- genes[, top_var]

# now identify the correlation between these genes
# apply cor() function 

correlation <- cor(top_genes)

# find all highly correlated gene pairs (abs (cor) > 0.9)
high_cor <- which(abs(correlation) > 0.9 &
                    abs(correlation) < 1, arr.ind = TRUE)

# turn into readable table with gene names
high_corr_df <- data.frame(
  Gene_1 = rownames(correlation)[high_cor[,1]],
  Gene_2 = colnames(correlation)[high_cor[,2]],
  Correlatin = correlation[high_cor]
)

View(high_corr_df) 
# highly correalated data
# remove highly correlated genes 
filtered_genes <- top_genes[, -high_cor]
cat("Remaining genes:", ncol(filtered_genes), "\n")

# now with these reduced number of features train logistic model

# first split data into train and test set (80/20 ratio)

# combine target column with genes
clean_data <- cbind(target, filtered_genes)
class(clean_data)

# splitting 80/20 ratio
index <- createDataPartition(clean_data$target,
                             p = 0.8,
                             list = FALSE)
train_data <- clean_data[index,]
test_data <- clean_data[-index,]

train_target <- train_data$target
tesr_target <- test_data$target

#---------------------------------------------------------------
#### Step 3: Build full and null logistic regression models ####
#    Needed for backward, forward, and step wise selection
#----------------------------------------------------------------
# with predictors
full_model <- glm(train_target~.,
                  data = train_data,
                  family = "binomial")

# only intercept/ no predictors
null_model <- glm(train_target~1,
                  data = train_data,
                  family = "binomial")
#--------------------------------------------------------------
##### Step 4: Perform Stepwise feature selection using AIC ####
#--------------------------------------------------------------

# Perform forward feature selection
#         Start with no features and add one at a time
forward_selction <- stepAIC(null_model,
                            scope = formula(full_model),
                            direction = "forward",
                            trace = TRUE)

# Perform backward feature selection 
#         Start with all features and remove one at a time
backward_selection <- stepAIC(full_model,
                              direction = "backward",
                              trace = TRUE)

# Perform stepwise selection (both directions) 
#         Combine forward and backward strategies)
stepwise_selection <- stepAIC(null_model,
                              scope = formula(full_model),
                              direction = "both",
                              trace = TRUE)

#----------------------
#### check results ####
#----------------------

# Result Summary
summary(forward_selction) 
summary(backward_selection) 
summary(stepwise_selection) 

# extract selected features
forward_features <- names(forward_selction$coefficients)[-1] # remove intercept
print(forward_features)

ward_features <- as.data.frame(names(backward_selection$coefficients)[-1]) # remove intercept
print(forward_features)

# save as csv file
write.csv(forward_features, file = "Features_forward_Selection.csv")

# check multicolinearity of model
library(car)
forward_VIF <- vif(full_model)

#--------------------------------------------------------
# Step 5: Make predictions and evaluate model performance
#--------------------------------------------------------
predicted_prob <- predict(backward_selection,
                          newdata = test_data,
                          type = "response")
predicted_classes <- ifelse(predicted_prob > 0.5, 1, 0)

# accuracy 
mean(predicted_classes == test_data$target)

# follow for more:
# github: https://github.com/AI-Biotechnology-Bioinformatics
# linkedin: https://www.linkedin.com/company/ai-and-biotechnology-bioinformatics/
