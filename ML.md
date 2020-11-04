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
## Linear Regression
## K-Neighbors
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
