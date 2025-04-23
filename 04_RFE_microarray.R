

# ===================================================================
#            AI in Biotechnology & Bioinformatics Series         
# ===================================================================
# -------------------------------------------------------------------
#                    Microarray Analysis – Part 4
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#            Recursive Feature Selection using Random forest          
# -------------------------------------------------------------------
# ===================================================================



#-----------------------------------------
#### STEP 1: Load and explore Dataset ####
#-----------------------------------------

# Install required packages for data analysis and modeling
install.packages("DataExplorer")  # for exploratory data analysis
install.packages("caret")         # for machine learning functions
install.packages("dplyr")         # for data manipulation

# Load the libraries
library(DataExplorer)
library(caret)
library(dplyr)

# Import dataset 
# already normalized microarray expression data 
# Here's the link to access the dataset: https://github.com/AI-Biotechnology-Bioinformatics/Microarray_Series_R

data <- read.csv("gastric_cancer.csv")

# Check the number of rows and columns
dim(data)

# View the structure of the first few columns
head(str(data[1:6]))

# Perform exploratory data analysis
introduce(data)            # Summary of dataset
plot_intro(data)           # Visual summary of variables
plot_bar(data$labels)      # Bar plot of sample labels (e.g., cancer vs normal)
plot_correlation(data)     # Correlation between features (only numeric columns)

#-------------------------------
##### Step 2: Prepare data ####
#-------------------------------

# Select only gene expression features (excluding ID and labels)
x <- data %>%
  select(-V1, -labels, -target) %>%
  as.data.frame()

# Select the data labels 
# convert it into numeric factor variable (e.g., 0 = normal, 1 = cancer)
y <- as.factor(data$labels,
               levels = c("cancer", "normal"),
               labels = c(1, 0))
class(y)
levels(y)

# Split the dataset into training (70%) and testing (30%) sets
index <- createDataPartition(y,
                             p = 0.7,
                             list = FALSE)

x_train <- x[index,] # train features
x_test <- x[-index,] # test features

y_train <- y[index] # train target
y_test <- y[-index] # test target


#--------------------------------------------
#### Step 4: Feature Selection with RFE ####
#--------------------------------------------

# Set up the RFE control with 10-fold cross-validation and Random Forest functions
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   verbose = TRUE)

# Apply RFE to select the most important features
# Sizes = number of top features to evaluate, set it accordingly

RFE_features <- rfe(x = x_train,
                    y = y_train,
                    sizes = c(1:5, 10, 30),
                    rfeControl = ctrl)

# View RFE results (e.g., which features performed best)
RFE_features

# Display the list of best performing features
predictors(RFE_features) #OR
RFE_features$optVariables


# Visualize the RFE results (accuracy vs number of features)
library(ggplot2)

ggplot(data = RFE_features,
       metric = "Accuracy")

# Extract top 30 features with their importance scores
top_var <- data.frame(features = row.names(varImp(RFE_features))[1:10],
                            scores = varImp(RFE_features)[1:10, 1])

# Let’s save these selected features so you can use them later
write.csv(top_var, file = "RFE_top_features.csv", row.names = FALSE)

#---------------------------------------------------------------------
#### Step 5: Build Random Forest Model with RFE Selected Features ####
#---------------------------------------------------------------------

# subset train and test set with only RFE selected features
subset_train <- x_train[, imp_variables$features]
subset_test <- x_test[, imp_variables$features]

# Set up training control for cross-validation
rf_ctrl <- trainControl(method = "cv",
                        number = 10,
                        verboseIter = TRUE)

# Train the Random Forest model using the selected features
Rf_model <- train(x = subset_train,
                  y = y_train,
                  method = "rf",
                  trControl = rf_ctrl)

# Predict the target labels on the test set
pred <- predict(Rf_model, subset_test)

# Compare predictions with actual test labels
y_test # check the difference between predicted and actual target (observe model's miss classifications)


# Generate and print the confusion matrix (model evaluation)
conf <- confusionMatrix(pred, y_test)
print(conf)

# Extract the confusion matrix table
conf_table <- as.table(conf$table)

# Convert to data frame
conf_df <- as.data.frame(conf_table)

# Save to CSV
write.csv(conf_df, "Confusion_Matrix_RF.csv", row.names = FALSE)

# ---------------------------------------------------------------------
# 🎯 Practice Task: Recursive Feature Elimination with SVM              
# ---------------------------------------------------------------------

# Bounce Question for You!
# Try running RFE using SVM instead of Random Forest.
# Here's a hint: SVM models are sensitive to the scale of the data.
# So, you'll need to **center and scale** the data before training.

# 📌 Tip:
# Models like Decision Trees and Random Forests **don't care** about feature scaling.
# But algorithms like **SVM** and **KNN** are sensitive to feature values.
# So always remember to scale your data when using these models.

# 👉 Try this:
# Use the 'svmFuncs' inside the 'rfeControl()' function instead of 'rfFuncs'.
# Also, use the 'preProcess = c("center", "scale")' argument to standardize your features.

# Example template to help you start:
ctrl_svm <- rfeControl(functions = svmFuncs,
                       method = "cv",
                       number = 10,
                       verbose = TRUE)

RFE_svm <- rfe(x = x_train,
               y = y_train,
               sizes = c(1:5, 10, 30),
               rfeControl = ctrl_svm,
               preProc = c("center", "scale"))

# Then check your results:
RFE_svm
predictors(RFE_svm)

# follow for more:
# github: https://github.com/AI-Biotechnology-Bioinformatics
# linkedin: https://www.linkedin.com/company/ai-and-biotechnology-bioinformatics/

