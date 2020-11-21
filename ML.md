
```R
# Installa packages
install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth'))

# Load the caret package
library(caret)

# Import dataset
orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

# Structure of the dataframe
str(orange)

# See top 6 rows and 10 columns
head(orange[, 1:10])
```
----------------------------------------------------------------------------------------------------------------------
# Statistics Descriptive
```R
library(skimr)
skimmed <- skim_to_wide(train_data)
skimmed[, c(1:5, 9:11, 13, 15:16)]

```
----------------------------------------------------------------------------------------------------------------------
# Create the knn imputation model on the dataset

```R
library(RANN)  # required for knnImpute
preProcess_missingdata_model <- preProcess(orange, method='knnImpute')
preProcess_missingdata_model
```
**Created from 828 samples and 18 variables

**Pre-processing:
  **- centered (16)
  **- ignored (2)
  **- 5 nearest neighbor imputation (16)
  **- scaled (16)
  
# Use the imputation model to predict the values of missing data points
```R
orange1 <- predict(preProcess_missingdata_model, newdata = orange)
anyNA(orange1)    
```
-------------------------------------------------------------------------------------------------------------------------
# One-Hot Encoding 
### If you have a categorical column as one of the features, 
### it needs to be converted to numeric in order for it to be used by the machine learning algorithms
```R
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars(Purchase ~ ., data=trainData)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

# # Convert to dataframe
trainData <- data.frame(trainData_mat)

# # See the structure of the new dataset
str(trainData)
```
------------------------------------------------------------------------------------------------------------------------

# Randomly order data
```R
set.seed(123)

rows<-sample(nrow(dataset))

random_dataset<-dataset[rows,]
```
-------------------------------------------------------------------------------------------------------------------------
# CreateDataPartition
```R
set.seed(123)

training_samples<-random_dataset$Rate_Mort_Child %>%
  createDataPartition(p=0.8,list=FALSE)
  
train_data<-random_dataset[training_samples,]
test_data<-random_dataset[-training_samples,]

```
------------------------------------------------------------------------------------------------------------------------

# See available algorithms in caret
```R
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames
```

--------------------------------------------------------------------------------------------------------------------------

# 3.- Models
## 

## Linear Regression


-------------------------------------------------------------------------------------------------------------------------
## Multivariate Adaptive Regression Splines (MARS) model by setting the method='earth'.

```R
# Step 1: Define the training control
fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final',       # saves predictions for optimal tuning parameter
    classProbs = T,                  # should class probabilities be returned
    summaryFunction=twoClassSummary  # results summary function
) 
# Step 2: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                        degree = c(1, 2, 3))

# Step 3: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars3 = train(Purchase ~ ., 
                   data=trainData, 
                   method='earth', 
                   metric='ROC', 
                   tuneGrid = marsGrid, 
                   trControl = fitControl)

# Step 4: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)

confusionMatrix(reference = testData$Purchase, 
                data = predicted3, 
                mode='everything', 
                positive='MM')

```


----------------------------------------------------------------------------------------------------------------------------
## K-Neighbors "Regression"
```R
# Fit the model on the training set
set.seed(123)
model_KNN<-train(Rate_Mort_Child~.,data=train_data,
                 method="knn",
                 trControl=trainControl("repeatedcv",number = 5, repeats = 3),
                 preProcess=c("center","scale"),
                 metric=c("RMSE"),
                 tuneLength=10)
                 
# Plot model error RMSE vs different values of k
plot(model_KNN,main="KNN vs RMSE")

# Information stored in the list returned by train()
names(model_KNN)

# Best tuning parameter k that minimize the RMSE
model_KNN$bestTune
metrics_train_KNN<-model_KNN$results[1,2:7]

metricas_train_KNN<-data.frame("metrics_train"="metrics_train_KNN",
                              Mean_RMSE=metrics_train$RMSE,
                              Mean_MAE=metrics_train$MAE,
                              Mean_RSquared=metrics_train$Rsquared,
                              Tasa_Error_Train=metrics_train$RMSE/mean(train_data$Rate_Mort_Child))

# Make predictions on the test data
predictions_KNN<-model_KNN %>% predict(test_data)

# Compute the prediction error RMSE, MAE,R2 & Error_Rate
Metricas_KNN_test<-data.frame("metrics_test"="metrics_test_KNN",
                              RMSE=RMSE(predictions_KNN,test_data$Rate_Mort_Child),
                              MAE=MAE(predictions_KNN,test_data$Rate_Mort_Child),
                              RSquared=caret::R2(predictions_KNN,test_data$Rate_Mort_Child), 
                              Tasa_Error_Test=RMSE(predictions_KNN,test_data$Rate_Mort_Child)/mean(test_data$Rate_Mort_Child))
Metricas_KNN_test

# Plot "Variable Importance"  
ggplot(varImp(model_KNN),15,main="% Importancia Variables")

```

------------------------------------------------------------------------------------------------------------------------------

## Decision Tree
### 1a.- Decision Tree Classification "Pruning the tree" (complexity parameter (cp))
```R
# Fit the model on the training set
library(rpart)
set.seed(123)
model2 <- train(
  diabetes ~., data = train.data, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
  )
  
# Plot model accuracy vs different values of
# cp (complexity parameter)
plot(model2)

# Print the best tuning parameter cp that
# maximizes the model accuracy
model2$bestTune

# Plot the final tree model
par(xpd = NA) # Avoid clipping the text in some device
plot(model2$finalModel)
text(model2$finalModel,  digits = 3)

# Decision rules in the model
model2$finalModel

# Make predictions on the test data
predicted.classes <- model2 %>% predict(test.data)
# Compute model accuracy rate on test data
mean(predicted.classes == test.data$diabetes)

```
------------------------------------------------------------------------------------------------------------------------------------
### 1b.- Decision Tree Regression "Pruning the tree" (complexity parameter (cp))-> Similar Classification except for metrics
```R
# Compute the prediction error RMSE
RMSE(predictions, test.data$medv)
```
### 2.- "Conditional Inference Tree" Classification 
```R
# Fit the model on the training set
library(party)
set.seed(123)
model <- train(
  diabetes ~., data = train.data, method = "ctree2",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(maxdepth = 3, mincriterion = 0.95 )
  )
plot(model$finalModel)

# Make predictions on the test data
predicted.classes <- model %>% predict(test.data)
# Compute model accuracy rate on test data
mean(predicted.classes == test.data$diabetes)
```
------------------------------------------------------------------------------------------------------------------------------
## Random Forest(Bagging)-> Classification
```R
library(randomForest)
# Fit the model on the training set
set.seed(123)
model <- train(
  diabetes ~., data = train.data, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE
  )
# Best tuning parameter
model$bestTune

# Final model
model$finalModel

# Make predictions on the test data
predicted.classes <- model %>% predict(test.data)
head(predicted.classes)

# Compute model accuracy rate
mean(predicted.classes == test.data$diabetes)

#Variable importance
importance(model$finalModel)

# Plot MeanDecreaseAccuracy
varImpPlot(model$finalModel, type = 1)
# Plot MeanDecreaseGini
varImpPlot(model$finalModel, type = 2)

#displays the importance of variables in percentage
varImp(model)

```
-------------------------------------------------------------------------------------------------------------------------------------
### Hyperparameters "nodesize & maxnodes" -> RandomForest
```R
models <- list()
for (nodesize in c(1, 2, 4, 8)) {
    set.seed(123)
    model <- train(
      diabetes~., data = na.omit(PimaIndiansDiabetes2), method="rf", 
      trControl = trainControl(method="cv", number=10),
      metric = "Accuracy",
      nodesize = nodesize
      )
    model.name <- toString(nodesize)
    models[[model.name]] <- model
}
# Compare results
resamples(models) %>% summary(metric = "Accuracy")
```

-----------------------------------------------------------------------------------------------------

## Adaboost-Gradient Boosting (Adaboost)
```R
set.seed(100)

# Train the model using adaboost
model_adaboost = train(Purchase ~ ., 
                 data=trainData, 
                 method='adaboost', 
                 tuneLength=2, 
                 trControl = fitControl)
model_adaboost
```

-------------------------------------------------------------------------------------------------------------------------
## Gradient Boosting (xgbTree)
```R
library(xgboost)
# Fit the model on the training set
set.seed(123)
model <- train(
  diabetes ~., data = train.data, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
  )
# Best tuning parameter
model$bestTune

# Make predictions on the test data
predicted.classes <- model %>% predict(test.data)
head(predicted.classes)

# Compute model prediction accuracy rate
mean(predicted.classes == test.data$diabetes)

# displays the importance of variables in percentage
varImp(model)

```
-------------------------------------------------------------------------------------------------------------------------------
## SVM
```R
set.seed(100)

# Train the model using MARS
model_svmRadial = train(Purchase ~ ., 
                  data=trainData, 
                  method='svmRadial', 
                  tuneLength=15, 
                  trControl = fitControl)
model_svmRadial

```
--------------------------------------------------------------------------------------------------------------------------------

# Run resamples() to compare the models
```R
# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, 
                             RF=model_rf, 
                             XGBDART=model_xgbDART, 
                             MARS=model_mars3, 
                             SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

```
```R
# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)
```
-----------------------------------------------------------------------------------------------------------------------------------------

# Metrics
## MAE
## MAPE
## RMSE
## CORRELATION
## BIAS


# Model Evaluation
## TRAIN TEST SPLIT
## CROSS VALIDATION
### GRID SEARCH
### RANDOMIZED SEARCH
