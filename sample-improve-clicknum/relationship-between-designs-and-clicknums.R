# load data about relationship between designs and clicknumber
dat1 <- read.table("./data/dat1.csv",
                        sep=",",
                        header=TRUE,
                        na.strings=c(''),
                        fill=TRUE)

# extract eplained valuables
df_ex <- dat1[, 1:12]
head(df_ex)

# ******* 相関係数 heatmap ***
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
# calculate cor
cormat <- round(cor(df_ex),2)
head(cormat)
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
melted_cormat
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)

# ******* 相関係数 heatmap ***

# ** make a poisson regression model **
p_model<-glm(click~.,data = dat1,family=poisson)
summary(p_model)

## choose explanation variables
step.p_model<-step(p_model)
summary(step.p_model)

# ** Negative binomial regression model is more suitable because mean < variance.
mean(dat1[,13])
var(dat1[,13])

# *** Negative Binomial ***
# make a negative binomial regression model
library(MASS)
nb_model<-glm.nb(click~.,dat1)
summary(nb_model)

# Which is more suitable, poisson regression or negatibe binomial regression ?
## Residual plot for Poisson regression
p_res <- resid(p_model)
plot(fitted(p_model), p_res, col='steelblue', pch=16,
     xlab='Predicted Offers', ylab='Standardized Residuals', main='Poisson')
abline(0,0)

## Residual plot for negative binomial regression 
nb_res <- resid(nb_model)
plot(fitted(nb_model), nb_res, col='steelblue', pch=16,
     xlab='Predicted Offers', ylab='Standardized Residuals', main='Negative Binomial')
abline(0,0)

## likelihood ratio test; Poisson and Negative binomial
#perform likelihood ratio test for differences in models
install.packages("lmtest")
library(lmtest)
lrtest(nb_model, p_model)

# negative binomial vs poisson
pchisq(2 * (logLik(nb_model) - logLik(p_model)), df = 1, lower.tail = FALSE)


# *** random forest regression ***
install.packages("randomForest") 
library(randomForest) 

rf_model<-randomForest(click~.,dat1,ntree=4000)
print(rf_model)

randomForest::varImpPlot(rf_model, 
                         sort=TRUE, 
                         main="Variable Importance Plot")


# *** support vector machine ***
# SVR

install.packages("e1071")
library("e1071")

svm_model<-svm(click~.,dat1)
print(svm_model)


# ** モデル同士の比較**
#d1-d12まで全て1で表現される「全部盛り」ベクトルxinです。

# predict(dat1.glm,newdata=xin,type="response")
# 
# predict(dat1.glmnb,newdata=xin,type="response")
# 
# predict(dat1.rf,xin)
# 
# predict(dat1.svm,xin)

## Residual plot for negative binomial regression 
nb_res <- resid(nb_model)
plot(fitted(nb_model), nb_res, col='steelblue', pch=16,
     xlab='Predicted Offers', ylab='Standardized Residuals', main='Negative Binomial')
abline(0,0)
