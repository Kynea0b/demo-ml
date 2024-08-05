# load data about relationship between designs and clicknumber
dat1 <- read.table("./data/dat1.csv",
                        sep=",",
                        header=TRUE,
                        na.strings=c(''),
                        fill=TRUE)

# ** make a poisson regression model **
p_model<-glm(click~.,data = dat1,family=poisson)
summary(p_model)


dat1.glm2<-step(p_model)

# ** Negative binomial regression model is more suitable because mean < variance.
mean(dat1[,13])
var(dat1[,13])

# ** 負の二項分布 **
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
