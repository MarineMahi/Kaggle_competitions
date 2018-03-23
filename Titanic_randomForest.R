
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
library(randomForest)
library(caret)
library(Metrics)



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

# ==========================================================================
#                             Random Forest model 
#===========================================================================

#------------- Tuning random forest hyperparameters using CV ----------

head(data)

# remove variables not used in analyses
train_data <- data[, !names(data) %in% c('PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked', '')]
head(train_data)

#Remove part of the dataset that contain unseen data that will be used for predictions
train_full <- subset(train_data, train_data$Survived != 'NA')
head(train_full)

# Variable 'survived' had to be changed from 0/1 to Yes/No for the train model to run
train_full$Survived <- ifelse(train_full$Survived == 1, "yes", "no")  

# Total number of rows in the data set
n <- nrow(train_full)

# Number of rows for the training set (80% of the dataset)
n_train <- round(0.8 * n) 

# Create a vector of indices which is an 80% random sample
set.seed(1)
train_indices <- sample(1:n, n_train)

# Subset the data frame to training indices only
training <- train_full[train_indices, ]  
head(training)

# Exclude the training indices to create the test set
validation <- train_full[-train_indices, ]  

# Specify the training configuration
ctrl <- trainControl(method = "cv",     # Cross-validation
                     number = 10,      # 10 folds
                     classProbs = TRUE,                  # For AUC
                     summaryFunction = twoClassSummary)  # For AUC

# Create a data frame containing all combinations of mtry
mtry <- seq(2, round(sqrt(ncol(training)))+2, 1)
tunegrid <- expand.grid(mtry = mtry)

#Create an empty list to store models
modellist <- c()

# Cross validate the random forest model using "rf" method; 
for (ntree in c(500, 1000, 1500, 2000, 2500)) {
  set.seed(15)
  model <- train(Survived~., data=training, method="rf", metric = "ROC", trControl=ctrl, tuneGrid = tunegrid, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- model
}

# compare results
modellist
results <- resamples(modellist)
summary(results)
dotplot(results)
# Best model: mtry= 3 and ntree = 500

# test on validation set
pred <- predict(object =train(Survived~., data=training, method="rf", metric = "ROC", trControl=ctrl, .mtry =3, ntree=500), 
                newdata = validation,
                type = "raw")

#Calculate the AUC value
AUC <- auc(actual = ifelse(validation$Survived == "yes", 1, 0), # Compute the AUC (`actual` must be a binary (or 1/0 numeric) vector)
           predicted = pred)

# 0.8096217


##################################################################################
#       PREDICTION ON UNSEEN DATASET
##################################################################################

test <- subset(data, is.na(data$Survived))
test_data <- test[, !names(test) %in% c('PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked', 'Celibacy', 'Survived')]
head(test_data)
Model_pred <- predict(object =train(Survived~., data=training, method="rf", metric = "ROC", trControl=ctrl, .mtry =3, ntree=500), 
                      newdata = test_data,
                      type = "raw")
fitted_data <- cbind(data = test_data, Model_pred)

head(fitted_data)
fitted_data$Model_pred <- as.character(fitted_data$Model_pred)
fitted_data$Model_pred[fitted_data$Model_pred == 'yes'] <- 1
fitted_data$Model_pred[fitted_data$Model_pred == 'no'] <- 0

results <- cbind(test$PassengerId, fitted_data$Model_pred)
head(results)
colnames(results) <- c('PassengerId', 'Survived')
results <- as.data.frame(results)
write.csv(results, file = 'results_randomForests.csv', row.names = F)
