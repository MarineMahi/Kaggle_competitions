
setwd ('C:/data_science/Kaggle/Titanic_competition')

test_data <- read.table('test.csv', header = T, sep = "," , na.strings = "")
train_data <- read.table ('Train.csv', header = T, sep = ",", na.strings = "") 
head(train_data)
head(test_data)

# Create a column "Survived" in the "test" dataset to bind both train_data and test_data together
test_data$Survived <- matrix (data =NA, nrow = nrow(test_data), ncol=1)
test_data <- test_data[,c(1,12,2:11)] # Reorganize columns order to be the same as the one in "train_data"
head(test_data)

data <- rbind(train_data, test_data) # Bind both datasets

# Download libraries
library(Amelia)
library(ggplot2)
library(GGally)
library(mice)
library(car)
library(caret)
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
data$celibacy[data$FamilySize == 1] <- 1 # Create a variable 'Celibacy'. 1 for individuals travelling alone
data$celibacy[data$FamilySize != 1] <- 0 # 0 for individuals travelling with relatives
data$celibacy <- as.factor(data$celibacy)

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
  geom_bar (aes(x = celibacy, fill = Survived))

ggplot(data = data) + 
  theme_bw() + 
  geom_bar (aes(x = Title, fill = Survived)) 


# Visualization relationship between variables
Data_cor <- data[, !names(data) %in% c('PassengerId','Name','Ticket','Cabin')]
ggpairs(Data_cor)


#===================================================================================
#  model development, model selection and threshold selection (via Cross-validation) 
#===================================================================================

head(data)

#Remove test datset that will be used for predictions
train_full <- subset(data, data$Survived != 'NA')
head(train_full)


#---- Model selection based on AIC -----

# Test two logistic models M0 and M0b to determine which variable to keep between 'Family size' and 'celibacy' 
M0 <- glm(data = train_full, Survived ~ Pclass + Sex + Title + Age + Fare + FamilySize, family = binomial )
vif(M0) # check for multicollinearity between variables
#GVIF Df GVIF^(1/(2*Df))
#Pclass     2.019436  2        1.192086
#Sex        1.394754  1        1.180997
#Title      1.676515  2        1.137894
#Age        1.559869  1        1.248947
#Fare       1.586963  1        1.259747
#FamilySize 1.623594  1        1.274203
# all VIF < 3 => no collinearity
summary(M0)
#AIC = 759.94
anova(M0, test = "Chisq") 

M0b <- glm(data = train_full, Survived ~ Pclass + Sex + Age + Title + Fare + celibacy, family = binomial )
summary(M0b)
# AIC = 792.35
anova(M0b, test = "Chisq") 

# variable family size is more relevant than celibacy. Celibacy is removed from subsequent analyses

M1 <- glm(data = train_full, Survived ~ (Age + Fare + Title + FamilySize)*(Sex + Pclass), family = binomial )
summary(M1)
#AIC = 748.88

M2 <- glm(data = train_full, Survived ~ (Pclass + Age + Title + Fare + FamilySize)*Sex, family = binomial )
summary(M2)
#AIC = 747.62

M3 <- glm(data = train_full, Survived ~ (Sex + Age + Title + Fare + FamilySize)*Pclass, family = binomial )
summary(M3)
#AIC= 735.94

m3b <- glm(data = train_full, Survived ~ (Sex + Age + Title + FamilySize)*Pclass + Fare, family = binomial ) 
summary(m3b)
# AIC = 738.38
anova(m3b, test = "Chisq")

m4 <- glm(data = train_full, Survived ~ (Age + Title + Fare + FamilySize)*Pclass, family = binomial )
summary(m4)
#AIC = 1031.9

m5 <- glm(data = train_full, Survived ~ (Sex + Title + Fare + FamilySize)*Pclass, family = binomial )
summary(m5)
#AIC = 737.97

m6 <- glm(data = train_full, Survived ~ (Sex + Age + Title + FamilySize)*Pclass, family = binomial )
summary(m6)
#AIC = 737.36
anova(m6, test = "Chisq")

m7 <- glm(data = train_full, Survived ~ (Sex + Age + Title + Fare)*Pclass, family = binomial )
summary(m7)
# AIc =  763.01

M8 <- glm(data = train_full, Survived ~ (Sex + Age + Fare + FamilySize)*Pclass, family = binomial )
summary(M8)
# AIC = AIC: 772.16

# => most parsimonious models: M3 and m6

# --------------- threshold selection (via Cross validation and ROc curve) ----------------

# Set random seed. 
set.seed(1)
# Shuffle the dataset
n <- nrow(train_full)
train_data_s <- train_full[sample(n),] 

#### ------ Model M3 -------

#Fold the dataset 10 times and calculate AUC, cutoffs and accuracy for each fold
nbfolds = 10
AUCM3 <- matrix(0,nrow = nbfolds, ncol = 1)
CutoffsM3 <- matrix(0,nrow = nbfolds, ncol = 1)
AccM3 <- matrix(0,nrow = nbfolds, ncol = 1)

for (i in 1:nbfolds) {
  indices <- (((i-1) * round((1/nbfolds)*nrow(train_data_s))) + 1):((i*round((1/nbfolds) * nrow(train_data_s))))
  train <- train_data_s[-indices,] # Define the training set to train the model
  val <- train_data_s[indices,] # Define the validation test to test the model
  M3 <- glm(data = train, Survived ~ (Sex + Age + Title + Fare + FamilySize)*Pclass, family = binomial )
  probM3 <- predict(M3, val, type = 'response')
  predM3 <- prediction(probM3, val$Survived)
  perfM3 <- performance(predM3, 'tpr','fpr')
  plot(perfM3, colorize=T, print.cutoffs.at=seq(0,1,by=0.1))
  perfM32 <- performance(predM3,'auc')
  AUCM3[i] = perfM32@y.values[[1]]
  perfM33 <- performance(predM3, measure = "acc")
  ind = which.max( slot(perfM33, "y.values")[[1]] )
  acc = slot(perfM33, "y.values")[[1]][ind]
  AccM3[i] = acc
  #Optimal cutoff (point closest to the TPR of \(1\) and FPR of \(0\))
  cost.perf = performance(predM3, "cost")
  CutoffsM3[i] = predM3@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
  }  

mean_AUCM3 = mean(AUCM3)
# 0.8714949
mean_cutofM3 = mean(CutoffsM3)
# 0.4998192
mean_accM3 <- mean(AccM3)
# 0.8493105


#### ------ Model M6 -------

set.seed(10)

#Fold the dataset 10 times and calculate ROC curve and AUC for each fold
# Initialize the roc vector 

nbfolds = 10
AUCM6 <- matrix(0,nrow = nbfolds, ncol = 1)
CutoffsM6 <- matrix(0,nrow = nbfolds, ncol = 1)
AccM6 <- matrix(0,nrow = nbfolds, ncol = 1)

for (i in 1:nbfolds) {
  indices <- (((i-1) * round((1/nbfolds)*nrow(train_data_s))) + 1):((i*round((1/nbfolds) * nrow(train_data_s))))
  train <- train_data_s[-indices,]
  val <- train_data_s[indices,]
  M6 <- glm(data = train, Survived ~ (Sex + Age + Title + FamilySize)*Pclass, family = binomial )
  probM6 <- predict(M6, val, type = 'response')
  predM6 <- prediction(probM6, val$Survived)
  perfM6 <- performance(predM6, 'tpr','fpr')
  plot(perfM6, colorize=T, print.cutoffs.at=seq(0,1,by=0.1))
  perfM6 <- performance(predM6,'auc')
  AUCM6[i] = perfM6@y.values[[1]]
  perfM6 <- performance(predM6, measure = "acc")
  ind = which.max( slot(perfM6, "y.values")[[1]] )
  acc = slot(perfM6, "y.values")[[1]][ind]
  AccM6[i] = acc
  cost.perf = performance(predM6, "cost")
  CutoffsM6[i] = predM6@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
}  

mean_AUCM6 = mean(AUCM6)
# 0.8707009
mean_cutofM6 = mean(CutoffsM6)
# 0.5438236
mean_accM6 <- mean(AccM6)
# 0.8492978

# Both models give similar AUC and Accuracy. We chose model M3 (lowest AIC, largest AUC and accuracy)

##################################################################################
#       PREDICTION SUR UNSEEN DATASET
##################################################################################

test <- subset(data, is.na(data$Survived))
head(test)
table(test$Title)

M3 <- glm(data = train_full, Survived ~ (Sex + Age + Title + Fare + FamilySize)*Pclass, family = binomial )
summary(M3)
probs <- predict (M3, test, type = 'response')
fitted_data <- cbind(data = test, probs)
predict_survival <- matrix(0,nrow=length(test[,1]),ncol=1)

head(fitted_data)

for (i in 1:nrow(predict_survival))
{
  if (fitted_data[i,16] >= 0.4998192) {predict_survival[i] = 1}
  else predict_survival[i] = 0
}

result_training <- cbind(fitted_data, predict_survival)
head(result_training)
survival_submission <- result_training[,c(1,17)]
head(survival_submission)
colnames(survival_submission) <- c('PassengerId', 'Survived')
survival_submission <- as.data.frame(survival_submission)
write.csv(survival_submission, file = 'survival_submission.csv', row.names = F)


