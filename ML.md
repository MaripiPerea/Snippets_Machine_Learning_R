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

# Models
## 

## Linear Regression
## K-Neighbors
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

# Compute the prediction error RMSE, MAE,R2
Metricas_KNN_test<-data.frame("metrics_test"="metrics_test_KNN",
                              RMSE=RMSE(predictions_KNN,test_data$Rate_Mort_Child),
                              MAE=MAE(predictions_KNN,test_data$Rate_Mort_Child),
                              RSquared=caret::R2(predictions_KNN,test_data$Rate_Mort_Child), 
                              Tasa_Error_Test=RMSE(predictions_KNN,test_data$Rate_Mort_Child)/mean(test_data$Rate_Mort_Child))
Metricas_KNN_test


```



## Decision Tree

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
