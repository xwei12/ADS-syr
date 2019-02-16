library(dplyr)
library(arules)
house_train<-read.csv('/Users/xinxuanwei/Desktop/ist707/final/train_clean.csv',header = T)
house_test<-read.csv('/Users/xinxuanwei/Desktop/ist707/final/test_clean.csv',header = T)
house<-rbind(house_train,house_test)
library(RWeka)
house_train
  house_new<-house_train[,c('SalePrice','OverallQual','KitchenQual','BsmtQual','OverallCond','BsmtCond','GrLivArea','TotalSF','BsmtExposure','GarageCars','Neighborhood','MSZoning')]
house1<-house_train[,c('SalePrice','OverallQual','KitchenQual','BsmtQual','OverallCond',
                       'BsmtCond','GrLivArea','TotalSF','LotArea','LotFrontage',
                       'BsmtExposure','GarageCars','Neighborhood','MSZoning')]
median(house_new$SalePrice)
salePrice<-c()
for(i in house_new[,1]){
  if(i<=163000){
    salePrice<-c(salePrice,'normal')
  }
  else{
    salePrice<-c(salePrice,'high')
  }
}
house_new<-cbind(house_new,salePrice)
house_new<-house_new[,-1]
house_new$OverallQual<-as.factor(house_new$OverallQual)
house_new$GarageCars<-as.factor(house_new$GarageCars)
house_new$OverallCond<-as.factor(house_new$OverallCond)
house_new$salePrice<-as.factor(house_new$salePrice)



#house_new_train<-house_new[1:1460,]
#house_new_test<-house_new[1461:2919,]
m=J48(salePrice~., data = house_new, control=Weka_control(U=FALSE, M=2, C=0.5))
m1=J48(salePrice~., data = house_new)
e <- evaluate_Weka_classifier(m, numFolds = 10, seed = 1, class = TRUE)
e1<- evaluate_Weka_classifier(m1, numFolds = 10, seed = 1, class = TRUE)
e$confusionMatrix
e1$confusionMatrix
e$details
e1$details
library(partykit)
plot(m,type="simple")

NB<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes") 
nb1<-NB(salePrice~., data = house_new)
e_nb<- evaluate_Weka_classifier(nb1, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
e_nb$confusionMatrix
e_nb$detailsClass

rf <- make_Weka_classifier("weka/classifiers/trees/RandomForest")
# build default model with 100 trees
#rf_model=rf(label~., data=trainset)
# build a model with 10 trees instead
rf_model=rf(salePrice~., data = house_new, control=Weka_control(I=10))
e5 <- evaluate_Weka_classifier(rf_model, numFolds = 3, seed = 1, class = TRUE)
e5$confusionMatrix

svm <- make_Weka_classifier("weka/classifiers/functions/SMO")
svm_model=svm(salePrice~., data = house_new)
e4 <- evaluate_Weka_classifier(svm_model, numFolds = 3, seed = 1, class = TRUE)
e4$confusionMatrix
e4

house$MSSubClass<-as.factor(house$MSSubClass)
house$OverallQual<-as.factor(house$OverallQual)
house$OverallCond<-as.factor(house$OverallCond)
house$OverallQual=dplyr::recode(house$OverallQual, '1'="Qual=1",'2'="Qual=2",'3'="Qual=3",'4'="Qual=4",'5'="Qual=5",'6'="Qual=6",'7'="Qual=7",'8'="Qual=8",'9'="Qual=9",'10'="Qual=10")
house$OverallCond=dplyr::recode(house$OverallCond,'1'="Con=1",'2'="Con=2",'3'="Con=3",'4'="Con=4",'5'="Con=5",'6'="Con=6",'7'="Con=7",'8'="Con=8",'9'="Con=9")
hist(house$YearBuilt)
house_cat<-house[,-c(1, 4, 5, 14, 19, 20, 23,32,36, 37, 38, 39, 40, 42, 43, 47, 50, 51, 54)]
table(house_cat$BsmtQual)
myRules = apriori(house_cat, parameter = list(supp = 0.001, conf = 0.9, maxlen = 3))
pepRules<-subset(myRules,rhs %pin% c('pep'))
summary(pepRules)

table(crime$Area.ID)
crime$Premise.Code<-as.factor(crime$Premise.Code)
length(table(crime$Crime.Code))
b<-na.omit(crime)

c<-crime[,-4:-7]

#crime age: 17-19 are least
barplot(table(crime$Victim.Age))

barplot(table(crime$Victim.Descent))

#crime time: most in 12:00-13:00pm)
barplot(table(floor(crime$Time.Occurred/100)))

