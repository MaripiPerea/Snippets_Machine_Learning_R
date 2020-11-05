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
model <- train(
  Rate_Mort_Child~., data = train_data, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 10
  )
# Plot model error RMSE vs different values of k
plot(model)
# Best tuning parameter k that minimize the RMSE
model$bestTune
# Make predictions on the test data
predictions <- model %>% predict(test_data)
head(predictions)
# Compute the prediction error RMSE
RMSE(predictions, test_data$Rate_Mort_Child)
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
