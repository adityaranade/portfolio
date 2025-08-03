library(readxl)
library(reshape2)
library(ggplot2)
library(dplyr)
library(ggh4x)
library(ggcorrplot)
library(GGally) # for pairs plot using ggplot framework
# library(car) # to calculate the VIF values
# library(cmdstanr)
# library(bayesplot)
# library(rstanarm)
###############################################################
# Get starbucks data from github repo
# path <- "https://raw.githubusercontent.com//adityaranade//portfolio//refs//heads//main//starbucks//starbucks-menu-nutrition-drinks.csv"
path <- "./gallstone_dataset.xlsx"
data0 <- read_excel(path)
###############################################################
# Data processing
head(data0)
colnames(data0)
data <- data.frame(data0)
colnames(data)

ggplot() +
  geom_tile(data = data, aes(x = Lean.Mass..LM.....,, y = C.Reactive.Protein..CRP., fill = as.factor(Gallstone.Status)), alpha = 0.3) +
  geom_point(data = data, aes(x = Lean.Mass..LM.....,, y = C.Reactive.Protein..CRP., color = as.factor(Gallstone.Status)), size = 3, shape = 21) +
  scale_color_manual(values = c('green4', 'red3')) +
  labs(title = 'SVM Decision Boundary', x = 'Lean.Mass..LM.....,', y = 'C.Reactive.Protein..CRP.') +
  theme_bw() +
  theme(legend.position = "none")

ggplot() +
  geom_tile(data = data, aes(x = Total.Body.Water..TBW., y = C.Reactive.Protein..CRP., fill = as.factor(Gallstone.Status)), alpha = 0.3) +
  geom_point(data = data, aes(x = Total.Body.Water..TBW., y = C.Reactive.Protein..CRP., color = as.factor(Gallstone.Status)), size = 3, shape = 21) +
  scale_color_manual(values = c('green4', 'red3')) +
  labs(title = 'SVM Decision Boundary', x = 'Total.Body.Water..TBW.', y = 'C.Reactive.Protein..CRP.') +
  theme_bw() +
  theme(legend.position = "none")



# Check the first 6 rows of the dataset
data |> head()

# Check the type of data
data |> str()

# Convert the data into appropriate factors or numbers
data$Gallstone.Status <- data$Gallstone.Status |> as.factor()

# Combine only numerical data along with the response
data1 <- data |> dplyr::select(Gallstone.Status,Vitamin.D,
                               Total.Body.Water..TBW.,
                               Lean.Mass..LM.....,
                               C.Reactive.Protein..CRP.)

data1 |> str()

# Check the rows which do not have any entries
sum(is.na(data1)) # No NA values

###############################################################
# EDA
# Data for histogram
melted_data <- melt(data1, id="Gallstone.Status")

# Plot the histogram of all the variables
ggplot(melted_data,aes(value))+
  geom_histogram(aes(),bins = 20)+
  # geom_histogram(aes(y = after_stat(density)),bins = 20)+
  facet_grid2(Gallstone.Status~variable, scales="free")+theme_bw()
###############################################################
# Pairs plot to check the correlation between each pair of variables
ggpairs(data1)
###############################################################
# split the data into training and testing data
seed <- 23
set.seed(seed)

ind <- sample(1:nrow(data1),
              floor(0.75*nrow(data1)),
              replace = FALSE)

# Training dataset
data_train <- data1[ind,]
# Testing dataset
data_test <- data1[-ind,]

# Scaled training and testing data
mean <- apply(data_train[,-1],2,mean)
sd <- apply(data_train[,-1],2,sd)
min <- apply(data_train[,-1],2,min)
max <- apply(data_train[,-1],2,max)
data_train2 <- data_train
data_test2 <- data_test
for (i in (2:ncol(data_train2))){
  for (j in 1:nrow(data_train2)){
    ## Standardization
    #data_train2[j,i] <- (data_train[j,i] - mean[i-1])/sd[i-1]
    
    ## Normalization
    data_train2[j,i] <- (data_train[j,i] - min[i-1])/(max[i-1]-min[i-1])
  }
  for (k in 1:nrow(data_test2)){
    ## Standardization
    #data_test2[k,i] <- (data_test[k,i] - mean[i-1])/sd[i-1]
    
    ## Normalization
    data_test2[k,i] <- (data_test[k,i] - min[i-1])/(max[i-1]-min[i-1])
  }
}
###############################################################
# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=5)
metric <- "Accuracy"

# k grid search
grid <- expand.grid(.k=seq(1,20,by=1))
fit.knn <- train(Gallstone.Status~., data=data_train2, method="knn", 
                 metric=metric, tuneGrid=grid, trControl=trainControl)
knn.k2 <- fit.knn$bestTune # keep this optimal k for testing with stand alone knn() function in next section
print(fit.knn)
plot(fit.knn)

# Predictions on testing data
prediction_knn <- predict(fit.knn, newdata = data_test2)
confusion_matrix_knn <- confusionMatrix(prediction_knn, 
                                        data_test$Gallstone.Status)
confusion_matrix_knn 
# Accuracy is 57.5%
###############################################################
library(MASS)
# Fit an ordinal logistic regression model
model_logistic <- glm(Gallstone.Status ~ .,
             family = binomial(link='logit'),data = data_train2)

# Check the summary of the model
model_logistic |> summary()

# Predictions on the testing data
y_pred_prob_logistic <- predict(model_logistic, data_test2, type = "response")
prediction_logistic <- ifelse(y_pred_prob_logistic > 0.5,1,0) |> as.factor()

# confusion matrix
confusion_matrix_logistic <- confusionMatrix(data_test$Gallstone.Status, 
                                      prediction_logistic)

confusion_matrix_logistic
# Accuracy is 80%

# Compute ROC curve
roc_curve <- roc(data_test2$Gallstone.Status,as.vector(y_pred_prob_logistic))

# Calculate AUC
auc_value <- auc(roc_curve)

# Plot the ROC curve
plot(roc_curve, col = "blue", lwd = 3, main = "ROC Curve")

# # ROC curve
# library(ROCR)
# pr <- prediction(y_pred, data_test$Gallstone.Status)
# prf <- performance(pr, measure = "tpr", x.measure = "fpr")
# plot(prf)

# Add AUC to the plot
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 3)

###############################################################
library(e1071)
model_svm = svm(Gallstone.Status ~ ., 
                 data = data_train2,
                 kernel = 'linear')

print(model_svm)

# Predictions on the testing data
y_pred_svm <- predict(model_svm, data_test2, type = "response")
prediction_svm <- y_pred_svm

# confusion matrix
confusion_matrix_svm <- confusionMatrix(data_test2$Gallstone.Status, 
                                             prediction_svm)

confusion_matrix_svm
# Accuracy is 77.5%
###############################################################
# Compare the different models
comb_models <- data.frame(KNN = confusion_matrix_knn$byClass,
                          Logistic = confusion_matrix_logistic$byClass,
                          SVM = confusion_matrix_svm$byClass) |> round(4)
comb_models
###############################################################