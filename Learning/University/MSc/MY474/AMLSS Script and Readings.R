## MY474 - Applied Machine Learning for Social Sciences

# Lecture 2 

# load data
library(palmerpenguins)
library(VGAM)
data(penguins)
table(penguins$species)

# transform data
cols <- c(1, 5, 6)
penguins <- penguins[complete.cases(penguins), cols]
nrow(penguins)

# Split into training & test
n_test <- floor(nrow(penguins) * 0.3)
idx <- sample(1:nrow(penguins), n_test)
head(idx)

te <- penguins[idx,]
tr <- penguins[-idx,]

nrow(te)
nrow(tr)

# Multinomial Logistic Regression on Train data

logit = vglm(species ~ ., family = "multinomial", tr)
prob <- predict(logit, tr, type = "response")
pred <- apply(prob, 1, which.max)

pred[which(pred == "1")] <- levels(tr$species)[1]
pred[which(pred == "2")] <- levels(tr$species)[2]
pred[which(pred == "3")] <- levels(tr$species)[3]

# Confusion Matrix on Training data
table(tr$species, pred)

ade_tr <- which(tr$species == "Adelie")
chi_tr <- which(tr$species == "Chinstrap")
gen_tr <- which(tr$species == "Gentoo")

mean(pred[ade_tr] == 'Adelie')
mean(pred[chi_tr] == 'Chinstrap')
mean(pred[gen_tr] == 'Gentoo')

# MLR on Test data

logit_te = vglm(species ~ ., family = "multinomial", te)
prob_te <- predict(logit_te, te, type = "response")
pred_te <- apply(prob_te, 1, which.max)

pred_te[which(pred_te == "1")] <- levels(te$species)[1]
pred_te[which(pred_te == "2")] <- levels(te$species)[2]
pred_te[which(pred_te == "3")] <- levels(te$species)[3]

table(te$species, pred_te)

ade_te <- which(te$species == "Adelie")
chi_te <- which(te$species == "Chinstrap")
gen_te <- which(te$species == "Gentoo")

mean(pred_te[ade_te] == 'Adelie')
mean(pred_te[chi_te] == 'Chinstrap')
mean(pred_te[gen_te] == 'Gentoo')

# K-nearest Neighbors

library(class)

# Scaling so our different variables are operating on same distance scales. 
tr_sc <- scale(tr[,-c(1)])
te_sc <- scale(te[,-c(1)])

# k is our smoothing parameter to specify how tightly to fit to the data. 
pred_3 <- knn(tr_sc, tr_sc, tr$species, k=3)
pred_100 <- knn(tr_sc, tr_sc, tr$species, k=100)

table(tr$species, pred_3)

mean(pred_3[ade_tr] == 'Adelie')
mean(pred_3[chi_tr] == 'Chinstrap')
mean(pred_3[gen_tr] == 'Gentoo')


table(tr$species, pred_100)


# Chapter 4 Lab (ISL)

library(ISLR)
data("Smarket")

head(Smarket)

cor(Smarket[,-9])

glm_fits <- glm(Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Smarket ,family="binomial")
summary(glm_fits)

coef(glm_fits)
summary(glm_fits)$coef

glm_probs=predict(glm_fits,type="response")
# probabilities of going up or down
glm_probs[1:10]

# allocated to up or down
glm_pred <- rep("Down",1250)
glm_pred[glm_probs >.5]="Up"

table(glm_pred, Smarket$Direction)
mean(glm_pred== Smarket$Direction) # accuracy of 52.2%

# At first glance, it appears that the logistic regression model is working a little better than random guessing. 
# However, this result is misleading because we trained and tested the model on the same set of 1, 250 observa- tions. 
# In other words, 100 − 52.2 = 47.8 % is the training error rate.

# Create test set

train <- Smarket %>%
  filter(Year < 2005)

test <- Smarket %>%
  filter(Year > 2004)

glm_fit_2 <- glm(Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=train ,family=binomial)

glm_probs_2 <- predict(glm_fit_2, test, type = "response")

glm_pred_2 <- rep("Down", 252)

glm_pred_2[glm_probs_2 >.5]="Up"

table(glm_pred_2, test$Direction)
mean(glm_pred_2== test$Direction) # accuracy of 48.0% WORSE THAN GUESSING. 
# The results are rather disappointing: the test error rate is 52%, which is worse than random guessing! 
# Of course this result is not all that surprising, given that one would not generally expect to be able to use previous days’ returns to predict future market performance.

# Lets remove all other Lag except 1 and 2, as they seemed useless above

glm_fits_3 <- glm(Direction ~ Lag1+Lag2, data=train ,family=binomial)

glm_probs_3 <- predict(glm_fits_3, test, type="response")

glm_pred_3 <- rep("Down",252)
glm_pred_3[glm_probs_3 >.5]="Up"

table(glm_pred_3,test$Direction)
mean(glm_pred_3==test$Direction) # Accuracy of 0.560

# However, the confusion matrix shows that on days when logistic regression predicts an increase in the market, it has a 58% accuracy rate. 
# This suggests a possible trading strategy of buying on days when the model predicts an in- creasing market, and avoiding trades on days when a decrease is predicted. 
# Of course one would need to investigate more carefully whether this small improvement was real or just due to random chance.

# Linear Discriminant Analysis (LDA)


library(MASS)

lda_fit <- lda(Direction ~ Lag1 + Lag2, data = train)
lda_fit

# The LDA output indicates that πˆ1 = 0.492 and πˆ2 = 0.508; 
# in other words, 49.2% of the training observations correspond to days during which the market went down. It also provides the group means; 
# these are the average of each predictor within each class, and are used by LDA as estimates of μk. 
# These suggest that there is a tendency for the previous 2 days’ returns to be negative on days when the market increases, and a tendency for the previous days’ returns to be positive on days when the market declines.

plot(lda_fit)

# The coefficients of linear discriminants output provides the linear combination of Lag1 and Lag2 that are used to form the LDA decision rule.
# In other words, these are the multipliers of the elements of X = x in (4.19). If −0.642 × Lag1 − 0.514 × Lag2 is large, then the LDA classifier will
# predict a market increase, and if it is small, then the LDA classifier will predict a market decline. The plot() function produces plots of the linear discriminants, 
# obtained by computing −0.642 × Lag1 − 0.514 × Lag2 for each of the training observations.

# The predict() function returns a list with three elements. The first ele- ment, class, contains LDA’s predictions about the movement of the market. 
# The second element, posterior, is a matrix whose kth column contains the posterior probability that the corresponding observation belongs to the kth class, computed from (4.10). 
# Finally, x contains the linear discriminants, described earlier.

lda_pred <- predict(lda_fit, test)

names(lda_pred)

lda_class <- lda_pred$class

table(lda_class, test$Direction)

mean(lda_class==test$Direction) # 0.560 accuracy
# As we observed in Section 4.5, the LDA and logistic regression predictions are almost identical.

# Applying a 50 % threshold to the posterior probabilities allows us to recreate the predictions contained in lda.pred$class.

sum(lda_pred$posterior[,1]>=.5) # 70 over 0.5, so predicted as down

sum(lda_pred$posterior[,1]<.5) # 182 under 0.5, so predicted as Up

# Notice that the posterior probability output by the model corresponds to the probability that the market will decrease:

lda_pred$posterior[1:20,1]
lda_class[1:20]

# If we wanted to use a posterior probability threshold other than 50% in order to make predictions, then we could easily do so. 

sum(lda_pred$posterior[,1]>.60)
# No days in 2005 meet that threshold! In fact, the greatest posterior prob- ability of decrease in all of 2005 was 52.02 %.

# Quadratic Discriminant Analysis (QDA)

qda_fit <- qda(Direction  ~ Lag1 + Lag2, data = train)

qda_fit

# The output contains the group means. But it does not contain the coef- ficients of the linear discriminants, 
# because the QDA classifier involves a quadratic, rather than a linear, function of the predictors. 
# The predict() function works in exactly the same fashion as for LDA.


qda_pred <- predict(qda_fit, test)

qda_class <- qda_pred$class

table(qda_class,test$Direction)

mean(qda_class == test$Direction) # 0.599 accuracy

# Interestingly, the QDA predictions are accurate almost 60% of the time, even though the 2005 data was not used to fit the model. 
# This level of accu- racy is quite impressive for stock market data, which is known to be quite hard to model accurately. 
# This suggests that the quadratic form assumed by QDA may capture the true relationship more accurately than the linear forms assumed by LDA and logistic regression. 
# However, we recommend evaluating this method’s performance on a larger test set before betting that this approach will consistently beat the market!

library(class)


# K-Nearest Neighbours 

# y from the other model- fitting functions that we have encountered thus far. Rather than a two-step approach 
# in which we first fit the model and then we use the model to make predictions, knn() forms predictions using a single command. The function requires four inputs.

# 1. A matrix containing the predictors associated with the training data, labeled train.X below.
# 2. A matrix containing the predictors associated with the data for which we wish to make predictions, labeled test.X below.
# 3. A vector containing the class labels for the training observations, labeled train.Direction below.
# 4. A value for K, the number of nearest neighbors to be used by the classifier.

train_X <- cbind(train$Lag1, train$Lag2)
test_X <- cbind(test$Lag1, test$Lag2)
train_Y <- train$Direction

set.seed(1)
knn_pred <- knn(train_X, test_X, train_Y, k=1)

table(knn_pred, test$Direction)
(83+43) /252 # 0.5

# Lets try k=3
knn_pred <- knn(train_X, test_X, train_Y, k=3)
table(knn_pred, test$Direction)

mean(knn_pred == test$Direction) # 0.536

# The results have improved slightly. But increasing K further turns out to provide no further improvements. 
# It appears that for this data, QDA provides the best results of the methods that we have examined so far.

# Applying Knn to Caravan Insurance Data 

data("Caravan")

dim(Caravan)

attach(Caravan)

# The response variable is Purchase, which indicates whether or not a given individual purchases a caravan insurance policy. 
# In this data set, only 6% of people purchased caravan insurance.

summary(Purchase)

348/5822


# Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, 
# the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between 
# the observations, and hence on the KNN classifier, than variables that are on a small scale. For instance, 
# imagine a data set that contains two variables, salary and age (measured in dollars and years, respectively). 
# As far as KNN is concerned, a difference of $1,000 in salary is enormous compared to a difference of 50 years in age. 
# Conse- quently, salary will drive the KNN classification results, and age will have almost no effect. 
# This is contrary to our intuition that a salary difference of $1, 000 is quite small compared to an age difference of 50 years. 

#  A good way to handle this problem is to standardize the data so that all variables are given a mean of zero and a standard deviation of one. 
# Then all variables will be on a comparable scale. The scale() function does just this. 
# In standardizing the data, we exclude column 86, because that is the qualitative Purchase variable.

stand_X <- scale(Caravan [,-86])

var (Caravan [ ,1]) # 165
var (Caravan [ ,2]) # 0.164 (1000 times smaller)
var(stand_X[,1]) # 1 
var(stand_X[,2]) # 1 

# We now split the observations into a test set, containing the first 1,000 observations, and a training set, 
# containing the remaining observations. 
# We fit a KNN model on the training data using K = 1, and evaluate its performance on the test data.

test =1:1000

train.X <- stand_X[-test ,]
test.X <- stand_X[test ,]

train.Y <- Purchase [-test]
test.Y <- Purchase [test]

set.seed(1)
knn.pred <- knn(train.X,test.X,train.Y,k=1)

mean(test.Y!=knn.pred) # 0.118
mean(test.Y!="No") # 0.059

# The KNN error rate on the 1,000 test observations is just under 12%. At first glance, this may ap- pear to be fairly good. 
# However, since only 6% of customers purchased insurance, we could get the error rate down to 6 % by always predicting No
# regardless of the values of the predictors!

# It turns out that KNN with K = 1 does far better than random guessing among the customers that are predicted to buy insurance. 
# Among 77 such customers, 9, or 11.7 %, actually do purchase insurance. This is double the rate that one would obtain from random guessing.

table(knn.pred, test.Y)
9/(68+9)

# Using K = 3, the success rate increases to 19 %, and with K = 5 the rate is 26.7 %. 
# This is over four times the rate that results from random guessing. 
# It appears that KNN is finding some real patterns in a difficult data set!


knn_pred = knn(train.X, test.X, train.Y, k=3)

table(knn_pred, test.Y)

5/26 # 0.192

knn_pred = knn(train.X, test.X, train.Y, k=5)

table(knn_pred, test.Y)

4/15 # 0.267

# As a comparison, we can also fit a logistic regression model to the data. 
# If we use 0.5 as the predicted probability cut-off for the classifier, then we have a problem: 
# only seven of the test observations are predicted to purchase insurance. 
# Even worse, we are wrong about all of these! However, we are not required to use a cut-off of 0.5. 
# If we instead predict a purchase any time the predicted probability of purchase exceeds 0.25, 
# we get much better results: we predict that 33 people will purchase insurance, 
# and we are correct for about 33% of these people. This is over five times better than random guessing!

glm.fits=glm(Purchase ~.,data=Caravan ,family=binomial, subset=-test)

glm.probs=predict(glm.fits,Caravan[test,],type="response")

glm.pred=rep("No",1000)
glm.pred[glm.probs >.5]="Yes"
table(glm.pred,test.Y)

glm.pred=rep("No",1000)
glm.pred[glm.probs >.25]=" Yes"
table(glm.pred,test.Y)

############################################################################ Seminar 2 

# HIGH VARIANCE IS WHEN MODEL CHANGES A LOT FROM SAMPLE TO SAMPLE - VERY SENSITIVE TO NOISE
# LOW VARIANCE IS WHEN MODEL DOES NOT CHANGE A LOT FROM SAMPLE TO SAMPLE - INSENSITIVE TO NOISE
# HIGH BIAS WHEN MODEL IS INFLEXIBLE AND DOES NOT CHANGE A LOT FROM SAMPLE TO SAMPLE
# LOW BIAS WHEN MODEL IS FLEXIBLE AND CHANGES A LOT FROM SAMPLE TO SAMPLE
# PLUS, THERE IS IRREDUCIBLE ERROR FROM THE DATA RELATIVE TO TRUE FORM - Measure of probabilistic noise that's innate to modelling
# Irreducible error is essentially measure of ignorance - something we cannot explain.

# Model Selection & Assessment

# How do we assess whether models do well?
# Cross-validation is one answer. 

# LINEAR MODEL WITH HIGHER BIAS because its simpler and doesnt change so much when trained sample-to-sample
# POLYNOMIAL MODEL WITH HIGHER VARIANCE because its more complex and changes more when trained sample-to-sample
# Higher order polynomials dont generalise great out-of-sample, especially where we extrapolate outside of bounds of training sample



# Chapter 5 Lab (ISL): Cross-Validation and Bootstrap

library(ISLR)
set.seed(1)
train=sample(392,196)

data(Auto)

lm_fit <- lm(mpg ~ horsepower , data=Auto, subset=train)

attach(Auto)

# We now use the predict() function to estimate the response for all 392 observations, 
# and we use the mean() function to calculate the MSE of the 196 observations in the validation set. 
# Note that the -train index below selects only the observations that are not in the training set.

mean((mpg - predict(lm_fit, Auto))[-train]^2) 
# 23.266

# Therefore, the estimated test MSE for the linear regression fit is 23.266. 
# We can use the poly() function to estimate the test error for the quadratic and cubic regressions.

lm_fit2 <- lm(mpg ~ poly(horsepower ,2),data=Auto,subset=train) 
mean((mpg-predict(lm_fit2,Auto))[-train]^2)
# 18.72
lm_fit3 <- lm(mpg ~ poly(horsepower ,3),data=Auto,subset=train) 
mean((mpg-predict(lm_fit3,Auto))[-train]^2)
# 18.79

# If we choose a different training set instead, then we will obtain somewhat different errors on the validation set.

set.seed(2)
train=sample(392,196)

lm_fit <- lm(mpg ~ horsepower ,subset=train)
mean((mpg - predict(lm_fit,Auto))[-train]^2)
# 25.73

lm_fit2 <- lm(mpg ~ poly(horsepower ,2),data=Auto,subset=train) 
mean((mpg-predict(lm_fit2,Auto))[-train]^2)
# 20.43

lm_fit3 <- lm(mpg ~ poly(horsepower ,3),data=Auto,subset=train) 
mean((mpg - predict(lm_fit3,Auto))[-train]^2)
# 20.39

# These results are consistent with our previous findings: a model that predicts mpg using a quadratic function of horsepower 
# performs better than a model that involves only a linear function of horsepower, and there is little evidence in favor of a model 
# that uses a cubic function of horsepower.


# 5.3.2 Leave-One-Out Cross-Validation
# The LOOCV estimate can be automatically computed for any generalized linear model using the glm() and cv.glm() functions. 
# In the lab for Chapter 4, we used the glm() function to perform logistic regression by passing in the family="binomial" argument. 
# But if we use glm() to fit a model without passing in the family argument, then it performs linear regression, 
# just like the lm() function. So for instance,

glm.fit <- glm(mpg ~ horsepower ,data=Auto)
coef(glm.fit)

lm.fit <- lm(mpg ~ horsepower ,data=Auto) 
coef(lm.fit)

#  In this lab, we will perform linear regression using the glm() function rather than the lm() function 
# because the former can be used together with cv.glm(). The cv.glm() function is part of the boot library.

glm.fit <- glm(mpg ~ horsepower ,data=Auto)
cv.err <- cv.glm(Auto,glm.fit)
cv.err$delta

# The cv.glm() function produces a list with several components. The two numbers in the delta vector contain the cross-validation results. 
# In this case the numbers are identical (up to two decimal places) and correspond to the LOOCV statistic given in (5.1).
# Below, we discuss a situation in which the two numbers differ. Our cross-validation estimate for the test error is approximately 24.23.

# We can repeat this procedure for increasingly complex polynomial fits.

cv.error <- rep(0,5)

for (i in 1:5){
  glm.fit=glm(mpg ~ poly(horsepower ,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit)$delta
}

cv.error

# As in Figure 5.4, we see a sharp drop in the estimated test MSE between the linear and quadratic fits,
# but then no clear improvement from using higher-order polynomials.

# 5.3.3 k-Fold Cross-Validation
# The cv.glm() function can also be used to implement k-fold CV. Below we use k = 10, a common choice for k, on the Auto data set. 
# We once again set a random seed and initialize a vector in which we will store the CV errors corresponding to the polynomial fits 
# of orders one to ten.

set.seed(17)
cv.error.10=rep(0,10)

for (i in 1:10){
  glm.fit=glm(mpg ~ poly(horsepower ,i),data=Auto)
  cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]
}

cv.error.10

cv.err <- cv.glm(Auto,glm.fit)

cv.err$delta
# We still see little evidence that using cubic or higher-order polynomial terms leads 
# to lower test error than simply using a quadratic fit.

# Notice that the computation time is much shorter than that of LOOCV. 
# (In principle, the computation time for LOOCV for a least squares linear model should be faster than for k-fold CV, 
# due to the availability of the formula (5.2) for LOOCV; 
# however, unfortunately the cv.glm() function does not make use of this formula.

# We saw in Section 5.3.2 that the two numbers associated with delta are essentially the same when LOOCV is performed. 
# When we instead perform k-fold CV, then the two numbers associated with delta differ slightly. The first is the 
# standard k-fold CV estimate, as in (5.3). 
# The second is a bias- corrected version. On this data set, the two estimates are very similar to each other.


##### The Bootstrap

# Estimating the Accuracy of a Statistic of Interest
# One of the great advantages of the bootstrap approach is that it can be applied in almost all situations. 
# No complicated mathematical calculations are required. Performing a bootstrap analysis in R entails only two steps.

# First, we must create a function that computes the statistic of interest.

# Second, we use the boot() function, which is part of the boot library, 
# to perform the bootstrap by repeatedly sampling observations from the data set with replacement.

# The Portfolio data set in the ISLR package is described in Section 5.2. 
# To illustrate the use of the bootstrap on this data, we must first create a function, alpha.fn(), 
# which takes as input the (X,Y) data as well as a vector indicating which observations should be used to estimate α. 
# The function then outputs the estimate for α based on the selected observations.

alpha.fn <- function(data,index){
  X=data$X[index]
  Y=data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y))) 
}

alpha.fn(Portfolio ,1:100) 

# The next command uses the sample() function to randomly select 100 ob- servations from the range 1 to 100, with replacement. 
# This is equivalent to constructing a new bootstrap data set and recomputing αˆ based on the new data set.

set.seed(1)

alpha.fn(Portfolio,sample(100,100,replace=T)) 
# 0.737

# We can implement a bootstrap analysis by performing this command many times, recording all of the corresponding estimates for α, 
# and computing the resulting standard deviation. However, the boot() function automates this approach. 
# Below we produce R = 1,000 bootstrap estimates for α.

boot(Portfolio ,alpha.fn,R=1000)

# The final output shows that using the original data, αˆ = 0.5758, and that the bootstrap estimate for SE(αˆ) is 0.0886.


# We can also use to measure variability in coefficients from linear regression

# first, build the function

boot.fn <- function(data,index) 
  return(coef(lm(mpg ~ horsepower ,data=data,subset=index))) 

boot.fn(Auto ,1:392)

set.seed(1)
boot.fn(Auto,sample(392,392,replace=T))

boot.fn(Auto,sample(392,392,replace=T))

# They're slightly different!

boot(Auto ,boot.fn ,1000)

# This indicates that the bootstrap estimate for SE(βˆ0) is 0.841, 
# and that the bootstrap estimate for SE(βˆ1) is 0.0073. 
# As discussed in Section 3.1.2, standard formulas can be used to compute the standard errors 
# for the regression coefficients in a linear model. These can be obtained using the summary() function.

summary(lm(mpg ~ horsepower ,data=Auto))$coef

# Interestingly, these are somewhat different from the estimates obtained using the bootstrap. 
# Does this indicate a problem with the bootstrap? In fact, it suggests the opposite. 

# Recall that the standard formulas given in Equation 3.8 on page 66 rely on certain assumptions. 
# For example, they depend on the unknown parameter σ2, the noise variance. We then estimate σ2 using the RSS. 
# Now although the formula for the standard errors do not rely on the linear model being correct, the estimate for σ2 does.
# We see in Figure 3.8 on page 91 that there is a non-linear relationship in the data, 
# and so the residuals from a linear fit will be inflated, and so will σˆ2. 
# Secondly, the standard formulas assume (somewhat unrealistically) that the xi are fixed, 
# and all the variability comes from the variation in the errors εi. 
# The bootstrap approach does not rely on any of these assumptions, 
# and so it is likely giving a more accurate estimate of the standard errors of βˆ0 and βˆ1 than is the summary() function.



# Below we compute the bootstrap standard error estimates and the standard linear regression estimates 
# that result from fitting the quadratic model to the data. 

boot.fn <- function(data,index)
  coefficients(lm(mpg ~ horsepower+I(horsepower^2),data=data, subset=index))

set.seed(1)
boot(Auto ,boot.fn ,1000)
summary(lm(mpg ~ horsepower+I(horsepower^2),data=Auto))$coef
# Since this model provides a good fit to the data (Figure 3.8), 
# there is now a better correspondence between the bootstrap estimates and the standard estimates of SE(βˆ0), SE(βˆ1) and SE(βˆ2).


# Chapter 6.5 Lab (ISL): Best Subset Selection

library(glmnet)
library(ISLR)
library(leaps)

data(Hitters)

Hitters <- na.omit(Hitters)

# The regsubsets() function (part of the leaps library) performs best subset selection 
# by identifying the best model that contains a given number of predictors, where best is quantified using RSS. 
#The syntax is the same as for lm(). 

regfit.full <- regsubsets(Salary ~.,Hitters)

summary(regfit.full)

# An asterisk indicates that a given variable is included in the corresponding model. 
# For instance, this output indicates that the best two-variable model contains only Hits and CRBI. 
# By default, regsubsets() only reports results up to the best eight-variable model. 
# But the nvmax option can be used in order to return as many variables as are desired. 
# Here we fit up to a 19-variable model (all explanatory variables)

regfit.full <- regsubsets(Salary ~.,data=Hitters ,nvmax=19)

reg.summary <- summary(regfit.full)

# The summary() function also returns R2, RSS, adjusted R2, Cp, and BIC. 
# We can examine these to try to select the best overall model.

names(summary(regfit.full))

# For instance, we see that the R2 statistic increases from 32 %, when only one variable is included in the model, 
# to almost 55%, when all variables are included. 
# As expected, the R2 statistic increases monotonically as more variables are included.

reg.summary$rsq

# Plotting RSS, adjusted R2, Cp, and BIC for all of the models at once will help us decide which model to select. 
# Note the type="l" option tells R to connect the plotted points with lines.

par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",
       type="l")
plot(reg.summary$adjr2 ,xlab="Number of Variables ",
       ylab="Adjusted RSq",type="l")

# The points() command works like the plot() command, except that it 
# puts points on a plot that has already been created, instead of creating a new plot. 

which.max(reg.summary$adjr2)

points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)

# In a similar fashion we can plot the Cp and BIC statistics, and indicate the models with the smallest statistic using which.min().

plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')

which.min(reg.summary$cp) 

points(10,reg.summary$cp [10],col="red",cex=2,pch=20)

which.min(reg.summary$bic)

plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",
       type='l')

points(6,reg.summary$bic [6],col="red",cex=2,pch=20)

# The regsubsets() function has a built-in plot() command which can be used to display the selected variables 
# for the best model with a given number of predictors, ranked according to the BIC, Cp, adjusted R2, or AIC.

plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2") 
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

# The top row of each plot contains a black square for each variable selected according to the optimal model associated with that statistic.
# For instance, we see that several models share a BIC close to −150. 
# However, the model with the lowest BIC is the six-variable model that contains only AtBat, Hits, Walks, CRBI, DivisionW, and PutOuts. 


# We can use the coef() function to see the coefficient estimates associated with this model.
coef(regfit.full ,6)

# Chapter 6.6 Lab (ISL): Ridge & LASSO Regression

# The main function in this package is glmnet(), which can be used to fit ridge regression models, lasso models, and more. 
# This function has slightly different syntax from other model-fitting functions that we have encountered thus far in this book. 
# In particular, we must pass in an x matrix as well as a y vector, and we do not use the y ∼ x syntax. 

x <- model.matrix(Salary ~.,Hitters)[,-1] 
y <- Hitters$Salary

# The model.matrix() function is particularly useful for creating x; 
# not only does it produce a matrix corresponding to the 19 predictors but it also automatically transforms 
# any qualitative variables into dummy variables. 
# The latter property is important because glmnet() can only take numerical, quantitative inputs.

# .6.1 Ridge Regression
# The glmnet() function has an alpha argument that determines what type of model is fit.
# If alpha=0 then a ridge regression model is fit, and if alpha=1 then a lasso model is fit. We first fit a ridge regression model.

grid=10^seq(10,-2,length=100)

ridge.mod <- glmnet(x, y, alpha=0, lambda=grid)

# here we have chosen to implement the function over a grid of values ranging from λ = 1010 to λ = 10−2,
# essentially covering the full range of scenarios from the null model containing only the intercept, 
# to the least squares fit. As we will see, we can also compute model fits for a particular value of λ 
# that is not one of the original grid values. 
# Note that by default, the glmnet() function standardizes the variables so that they are on the same scale. 
# To turn off this default setting, use the argument standardize=FALSE.

# Associated with each value of λ is a vector of ridge regression coefficients, stored in a matrix that can be accessed by coef(). \
# In this case, it is a 20×100 matrix, with 20 rows (one for each predictor, plus an intercept) and 100 columns (one for each value of λ).

coef(ridge.mod)
dim(coef(ridge.mod))

# We expect the coefficient estimates to be much smaller, in terms of l2 norm, 
# when a large value of λ is used, as compared to when a small value of λ is used. 

# We can use the predict() function for a number of purposes. 
#For instance, we can obtain the ridge regression coefficients for a new value of λ, say 50:

predict(ridge.mod,s=50,type="coefficients")[1:20,]


# Train/Test split

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh =1e-12)

ridge.pred=predict(ridge.mod,s=4,newx=x[test,]) 

mean((ridge.pred-y.test)^2)

# Chapter 8.3.1 Lab (ISL): Fitting Classification Trees
library(tree)

attach(Carseats)

High <-  ifelse(Sales <=8,"No","Yes")

Carseats <- data.frame(Carseats, High) 

# We now use the tree() function to fit a classification tree in order to predict High using all variables but Sales.

tree.carseats <- tree(High ~., Carseats)

# The summary() function lists the variables that are used as internal nodes
# in the tree, the number of terminal nodes, and the (training) error rate.

summary(tree.carseats)

summary(tree.carseats)







