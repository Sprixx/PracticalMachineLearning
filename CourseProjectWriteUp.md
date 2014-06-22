Course Project Write up for the practical machine learning course on Coursera
========================================================

In this document the reasoning behind the construction of my machine learning algorithms is given and the steps taken are explained.
I will explain how I built the model, how I used cross validation, what I think about the expected out of sample error and why I made the decisions I did.

I will describe all this by taking the reader along my steps I took in R. I will both provide the code and the outcome of each part of code to ensure full reproducability. Next to the code and outcome I will also share my thinking about these steps.

### Loading and Cleaning the Data


```r
pml.training <- read.csv("pml-training.csv")  #This loads the data into R right from the .csv

library(caret)  #Next the caret package is installed which will be used in the construction of the model
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(5)
Trainset <- createDataPartition(y = pml.training$classe, p = 0.3, list = FALSE)  #This command splits the trainingdata into a training set and a test set where 30% of the data goes into the training set.
```

The splitting of the data into a training and test part is done to be able to test the out of sample error (testing on the test set). This is a way to test the out of sample performance of our to be constructed model which is what a model is intended for.
I have chosen to include only 30% in the training set instead of the rule of thumb value of 70%. This is to ensure a short compilation time, and since the total data set is very large (appro 20k value) also 30% will give a very good sammple to train the model on. 
Furthermore Id like to note that the `createDataPartition` function selects the partition not based on the first 30% and the second 70% of the data but selects random subsample without replacement.


```r
Train1 <- pml.training[Trainset, ]  #Here the split of the createDataPartition is executed to form a training set.
Test1 <- pml.training[-Trainset, ]  #Here the split of the createDataPartition is executed to form a test set.


removeNa <- data.frame()  #Next an empty matrix is constructed which is used next.
for (i in 1:160) {
    # This loops through all columns of the training set and determines the
    # fraction of NA's in that column and also determines the fraction of empty
    # values in a column.
    removeNa[i, 1] <- sum(is.na(Train1[, i]))/length(Train1[, i])  #NA's
    removeNa[i, 2] <- sum(Train1[, i] == "")/length(Train1[, i])  #Empty values
}

Train2 <- Train1[, which(removeNa[, 1] < 0.7 & removeNa[, 2] < 0.7)]  #Next a new trainingset is constructed by deleting the columns from the original training set for which there were more than 70% NA's or empty values. 
# It turned out that columns either had NO NA's or missing value or a large
# number so the columns with large numbers of NA's and empty values are
# deleted from the data and therefore are also not taking into account in
# the construction of the model or the prediction.

Test2 <- Test1[, which(removeNa[, 1] < 0.7 & removeNa[, 2] < 0.7)]  #The same columns are removed from the test set to ensure similarity between these data sets.

Train3 <- Train2[, -c(1, 2, 3, 4, 5, 6)]  #Next the variables on the columns 1,2,3,4,6 are also removed since these seemed not usable for the model. The variables: X, user_name, 3x timestamp variables, new_window. These were disgarded due to their lack of interpratational effect compared to the classses. The X for example only indicated the number of the measurement which should not be taking into account when constructing the model.

Test3 <- Test2[, -c(1, 2, 3, 4, 5, 6)]  #Again the same columns are removed from the test set.
```


What I have done is remove all the variables from the data matrix which are not needed either by their lack of intuitive interpretation with regards to the classe variable or due to the large number of missing or NA values. The result is a clean data sets with already alot of variables removed from the data set.
All varaibles in the remaining data set contain numbers and are fully filled.

### The Final Model
In this section I prove the code and outcome of the final model and what it means.

```r
set.seed(5)
RFModel <- train(Train3$classe ~ ., data = Train3, method = "rf", trControl = trainControl(method = "cv", 
    number = 10))
# The code to constructed the final model using the just constructed train
# set. It is constructed using the rf (random forest) method using all
# remaining variables. Furthermore 10-fold cross validation is used for the
# training of the model. This implies that the data is splitted in 10 folds
# from which one is removed to cross validate the result on (10 times).

RFModel  #This outputs the result of the training of the model.
```

```
## Random Forest 
## 
## 5889 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 5300, 5300, 5299, 5301, 5300, 5300, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.005        0.007   
##   30    1         1      0.006        0.007   
##   50    1         1      0.005        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

From the output of the `RFModel` it is concluded that the in sample accuracy is 100% which is great. To test the out of sample error, which is more of interest I do the following:


```r
confusionMatrix(predict(RFModel, Test3), Test3$classe)  #This is the confusion matrix where the prediction of the model is inputted on the test data and set against the actual values from the test data (test data is the 70% not used for training from the training set in this case). 
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906   33    0    0    0
##          B    0 2606   29    1    1
##          C    0   17 2356   33   11
##          D    0    1   10 2216    8
##          E    0    0    0    1 2504
## 
## Overall Statistics
##                                         
##                Accuracy : 0.989         
##                  95% CI : (0.988, 0.991)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.987         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.981    0.984    0.984    0.992
## Specificity             0.997    0.997    0.995    0.998    1.000
## Pos Pred Value          0.992    0.988    0.975    0.991    1.000
## Neg Pred Value          1.000    0.995    0.997    0.997    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.172    0.161    0.182
## Detection Prevalence    0.287    0.192    0.176    0.163    0.182
## Balanced Accuracy       0.998    0.989    0.989    0.991    0.996
```


From the confusion matrix it can directly be concluded that the model performs very well. Almost all cases from the test set (70% of the training set) gets classified correctly (98.9%) imply a very low out of sample error. This is out of sample since this data was not used to train the model on. 
Therefore the model is validated using cross validation (also in the construction of the model): trained on a training set (30% of training.csv) and tested on a test set (70% of the training.csv).

The estimated out of sample error based on this model and data is: 98.9% accuracy and therefore 1.1% error.

### The test set and programming submission.
Finally this model is used to predict the data from the test.csv file such that all entries are predicted in which class they fall. This is done as follows:

```r
pml.testing <- read.csv("pml-testing.csv")  #The test data is loaded. Note that the classes are not included in this test set and the error rate can therefore not be determined for this data set( the answers not present)
TestFinal1 <- pml.testing  #Renaming
TestFinal2 <- TestFinal1[, which(removeNa[, 1] < 0.7 & removeNa[, 2] < 0.7)]  #The same variables as in the training set are removed. The test set needs to be adjusted in exactly the same way as the training set.
TestFinal3 <- (TestFinal2[, -c(1, 2, 3, 4, 5, 6, 60)])  #Finally the first 6 varaibles are again removed and in adition the varaible which replaced the class variable is removed (60, projectid). This is to identify the project but is not used in the model since this was not included in the training set.
# The result is a test set which has exactly the same columns and variables
# as the trainin set used in the training of the model.
predict(RFModel, TestFinal3)  #Next the answers for the test data set are predicted using the model we training in the previous section.
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# These answers are next submitted after construction correct files for them
# of which I have not shown the code. The result is a 100% score for all the
# 20 test cases. Therefore the out of sample error for this very small test
# set could be seen as 0% although it must be noted that this is a very
# small data set so the actual error rate could very easiliy be 99% as we
# found in our our of sample error result.
```


This is the end of the report.
Regards,

Sprixx
