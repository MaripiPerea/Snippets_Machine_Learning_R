# 1.- Randomly order data
```R
set.seed(123)

rows<-sample(nrow(dataset))

random_dataset<-dataset[rows,]
```

# 2.- CreateDataPartition
```R
set.seed(123)

training_samples<-random_dataset$Rate_Mort_Child %>%
  createDataPartition(p=0.8,list=FALSE)
  
train_data<-random_dataset[training_samples,]
test_data<-random_dataset[-training_samples,]
```

# 3.- Models
## 

## Linear Regression


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

## Random Forest-> Classification
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
