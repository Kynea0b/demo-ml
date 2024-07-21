
# load data about relationship between designs and clicknumber
dat1 <- read.table("./data/dat1.csv",
                   sep=",",
                   header=TRUE,
                   na.strings=c(''),
                   fill=TRUE)

# ** make a poisson regression model **
p_model_glm<-glm(click~.,dat1,family=poisson)
summary(p_model_glm)


p_model_glm2<-step(p_model_glm)


mean(dat1[,13])

var(dat1[,13])

# ** Negative binomial regression model is more suitable because mean < variance.

# make a negative binomial regression model
library(MASS)
nb_model_glmnb<-glm.nb(click~.,dat1)
summary(nb_model_glmnb)


# Which is more suitable, poisson regression or negatibe binomial regression ?
# Poisson regression model
## get residuals from poisson regression model
p_res <- resid(p_model_glm)

## plot between predicted click num and designs
plot(fitted(p_model_glm), p_res, col='steelblue', pch=16,
     xlab='Predicted ClickNums', ylab='Standardized Residuals', main='Poisson')

abline(0,0)


# Negative binomial regression model
## get residuals from binomial regression model
nb_res <- resid(nb_model_glmnb)

plot(fitted(nb_model_glmnb), nb_res, col='steelblue', pch=16,
     xlab='Predicted ClickNums', ylab='Standardized Residuals', main='Negative Binomial')

abline(0,0)

