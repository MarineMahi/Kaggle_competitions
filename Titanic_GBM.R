setwd ('C:/data_science/Kaggle/Titanic_competition')

test_data <- read.table('test.csv', header = T, sep = "," , na.strings = "")
train_data <- read.table ('Train.csv', header = T, sep = ",", na.strings = "") 
head(train_data)
head(test_data)

# Create a column "Survived" in the "test" dataset to bind both train_data and test_data together
test_data$Survived <- matrix (data =NA, nrow = nrow(test_data), ncol=1)
test_data <- test_data[,c(1,12,2:11)] # Reorganize columns order to be the same as the one in "train_data"
head(test_data)

data <- rbind(train_data, test_data) # Bind both datase

library(Amelia)
library(ggplot2)
library(GGally)
library(mice)
library(caret)
library(Metrics)
library(gbm)
library(ROCR)


#===============================================================================
# Data exploration, data cleaning, data transformation and feature engineering 
#===============================================================================

# check(data)
str(data)
summary(data)

# Map missing data
missmap (data, legend = T)

# Too many missing values in variable 'Cabin'. Remove this variable for analyses
# Observation with missing value in variable 'Embarked' is removed from the database 
# (WARNING: can be removed because missing value in the train_dataset, could not be removed if was in the test dataset)
# 'Fare' has one missing value that needs to be replaced (because missing value in the test dataset, cannot be removed) 
# 'Age' have missing values that need to be replaced

# ------------ Data cleaning -----------------

#Remove observation with missing values in variable 'Embarked' 
data <- subset (data, data$Embarked != 'NA')

# Replace missing value in 'Fare' with mean 
data$Fare[is.na(data$Fare)] <- mean(data$Fare, na.rm = T)

# Vizualisation variable 'Age'
NoMissingAge <- subset (data,data$Age!='NA')
NoMissingAge$Survived <- as.factor(NoMissingAge$Survived)
ggplot(data = NoMissingAge, aes(x = Age, fill = Survived)) + 
  theme_bw() + 
  geom_histogram (binwidth = 1) 

# Many missing values in 'Age' need to be replaced. 
# Replacing all missing value with mean would change the distribution of data compared to the original distribution
# Mean would stay identical but the variance is likely to change too much.
# Replacing missing values with mean is not appropriate in that case.

# Imputing missing age values using package (mice)
md.pattern(data) # Other way to visualize variables with missing values

# Remove variables that do not seem appropriate to impute missing values in age
Data_age_tempo <- data[, !names(data) %in% c('PassengerId','Survived', 'Name','Ticket','Cabin')]
set.seed(100)
Imputed_Age <- mice(Data_age_tempo)
summary(Imputed_Age)
# Save the output
mice_output <- complete(Imputed_Age)

# Check distribution after replacing missing values
ggplot(data = mice_output ) + 
  theme_bw() + 
  geom_histogram(aes(x = Age), binwidth=1) 

#Replace Age data in the database with imputed data
data$Age <- mice_output$Age

# ---------- Data transformation ---------

head(data)
data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.character(data$Sex)
data$Sex[data$Sex == 'female'] <- c(1)
data$Sex[data$Sex == 'male'] <- c(0)
data$Sex <- as.factor(data$Sex)
data$Embarked <- as.character(data$Embarked)
data$Embarked[data$Embarked == 'C'] <- c(0)
data$Embarked[data$Embarked == 'Q'] <- c(1)
data$Embarked[data$Embarked == 'S'] <- c(2)
data$Embarked <- as.factor(data$Embarked)

# --------- Feature engineering --------------

data$FamilySize = data$SibSp + data$Parch + 1  # Create variable family size

# extraction title from name
is.character(data$Name) # False
data$Name <- as.character(data$Name)
data$Title <- sapply(data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]}) 
data$Title <- sub(' ', '', data$Title) # remove space before title
table(data$Title)

#Group titles into 3 different categories
data$Title[data$Title %in% c('Capt', 'Col', 'Dr', 'Major', 'Rev')] <- 0 # Official
data$Title[data$Title %in% c('Don', 'Dona', 'Jonkheer', 'Lady', 'Master', 'Sir', 'the Countess')] <- 1 # Noble
data$Title[data$Title %in% c('Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms')] <- 2 # Popular
data$Title <- as.factor(data$Title)

# --------- Data visualization -------------

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Pclass, fill = Survived)) 

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Sex, fill = Survived)) 

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = SibSp, fill = Survived)) 

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Parch, fill = Survived)) 

ggplot(data = data) + 
  theme_bw() + 
  geom_histogram(aes(x = Fare, fill = Survived), binwidth=10) 

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Embarked, fill = Survived))

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = FamilySize, fill = Survived))

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Title, fill = Survived)) 


# Visualization relationship between variables
Data_cor <- data[, !names(data) %in% c('PassengerId','Name','Ticket','Cabin')]
ggpairs(Data_cor)


# ===================================================
#            Gradient boosting model 
# ===================================================

# remove variables not used in analyses
train_data <- data[, !names(data) %in% c('PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked')]
head(train_data)


#Remove part of the dataset that contain unseen data that will be used for predictions
train_full <- subset(train_data, train_data$Survived != 'NA')
head(train_full)

# Total number of rows in the data set
n <- nrow(train_full)

# Number of rows for the training set (80% of the dataset)
n_train <- round(0.8 * n) 

# Create a vector of indices which is an 80% random sample
set.seed(1)
train_indices <- sample(1:n, n_train)

# Subset the data frame to training indices only
training <- train_full[train_indices, ]  

# Exclude the training indices to create the test set
validation <- train_full[-train_indices, ]  


# =========================================================
#    Tuning the GBM hyperparameters using package caret 
# =========================================================

# Variable 'survived' has to be changed from 0/1 to Yes/No for the train model to run
training$Survived <- ifelse(training$Survived == 1, "yes", "no")  

# Set up training control
ctrl <- trainControl(method = "repeatedcv",   # repeated cross validation
                     number = 5,	# do 5 repetitions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     allowParallel = TRUE)


# ------------ Tuning hyperparameters --------------------------

# First tuning 
grid1 <- expand.grid(n.trees=c(1000, 5000, 10000, 15000),	        # Num trees to fit
                     shrinkage=c(0.001,0.005, 0.01, 0.05, 0.1),  # Try different values for learning rate
                     interaction.depth = c(1, 5, 10),
                     n.minobsinnode = c(3,5,10))		

set.seed(1955)  # set the seed

gbm.tune1 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid1)

gbm.tune1$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
# 5000                 5     0.001             3

# Second tuning
grid2 <- expand.grid(n.trees=c(3000, 5000, 7000, 9000),	        # Num trees to fit
                     shrinkage=c(0.001), 
                     interaction.depth = c(3, 5, 7),
                     n.minobsinnode = c(1,2,3))		

set.seed(1957)  # set the seed

gbm.tune2 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid2)

gbm.tune2$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#28    9000                 7     0.001              1
plot(gbm.tune2)
res <- gbm.tune2$results
print(gbm.tune2)

# Third tuning
grid3 <- expand.grid(n.trees=c(7500, 8000, 9000, 9500),	        # Num trees to fit
                     shrinkage=c(0.001), 
                     interaction.depth = c(6, 7, 8, 9),
                     n.minobsinnode = c(1,2,3))		

set.seed(1959)  # set the seed

gbm.tune3 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid3)

gbm.tune3$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#7    9000                 6     0.001              2
plot(gbm.tune3)
res <- gbm.tune3$results
print(gbm.tune3)

# Fourth tuning
grid4 <- expand.grid(n.trees=c(9000, 9200, 9300, 9400),	        # Num trees to fit
                     shrinkage=c(0.001), 
                     interaction.depth = c(5, 6, 7),
                     n.minobsinnode = c(1,2,3))		

set.seed(2000)  # set the seed

gbm.tune4 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid4)

gbm.tune4$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#   9400                 7     0.001              1
plot(gbm.tune4)
res <- gbm.tune4$results
print(gbm.tune4)

# Fifth tuning
grid5 <- expand.grid(n.trees=c(9350, 9400, 9450),	        # Num trees to fit
                     shrinkage=c(0.001), 
                     interaction.depth = c(5, 6, 7, 8, 9),
                     n.minobsinnode = c(1,2,3))		

set.seed(2001)  # set the seed

gbm.tune5 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid5)

gbm.tune5$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#40    9400                8     0.001              1

plot(gbm.tune5)
res <- gbm.tune5$results
print(gbm.tune5)

# Sixth tuning
grid6 <- expand.grid(n.trees=c(9375:9425),	        # Num trees to fit
                     shrinkage=c(0.001), 
                     interaction.depth = c(5, 6, 7, 8, 9),
                     n.minobsinnode = c(1,2,3))		

set.seed(2001)  # set the seed

gbm.tune6 <- train(Survived~., 
                   data=training,
                   method = "gbm",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid=grid6)

gbm.tune6$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#39    9420                 8     0.001              1
plot(gbm.tune6)
res <- gbm.tune6$results
print(gbm.tune6)


gridFinal <- expand.grid(n.trees=9420,	   
                       shrinkage=c(0.001), 
                       interaction.depth = 8,
                       n.minobsinnode = 1)
preds <- predict(object = train(Survived~., 
                                data=training,
                                method = "gbm",
                                metric = "ROC",
                                trControl = ctrl,
                                tuneGrid=gridFinal),
                 newdata = validation,
                 type = 'raw')

preds <- ifelse(preds == "yes", 1, 0)

# Estimate model performance using ROC curve and AUC
pred <- prediction(preds, validation$Survived)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs,  colorize=T, print.cutoffs.at=seq(0,1,by=0.1), main = "ROC Curve") # plot ROC curve

AUC_perf <- performance(pred,'auc')
AUC = AUC_perf@y.values[[1]] # Estimate AUC from ROC curve (package ROCR)


#==================================================
#       PREDICTION SUR UNSEEN DATASET
#==================================================

test <- subset(data, is.na(data$Survived))
table(test$Title)
test_data <- test[, !names(test) %in% c('PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked', 'celibacy', 'Survived')]
head(test_data)
Model_pred <- predict(object = train(Survived~., 
                                     data=training,
                                     method = "gbm",
                                     metric = "ROC",
                                     trControl = ctrl,
                                     tuneGrid=gridFinal),
                      newdata = test_data,
                      type = 'raw')

Model_pred <- ifelse(Model_pred == "yes", 1, 0)

fitted_data <- cbind(data = test_data, Model_pred)
head(fitted_data)

result <- cbind(test$PassengerId, fitted_data$Model_pred)
head(result)
colnames(result) <- c('PassengerId', 'Survived')
survival_submission <- as.data.frame(result)
write.csv(survival_submission, file = 'result_GBM.csv', row.names = F)


