##################### MACHINE LEARNING COURSE EX5 ##########################

# R script which produces responses for machine learning course

# Load Packages -----------------------------------------------------------

library(R.matlab)
library(tidyverse)
library(optimx)

theme_set(theme_bw())

# Load Data ---------------------------------------------------------------

# set wd path
wd <- "/Users/samtaylor/Documents/Git/Machine_Learning_Course/Machine_Learning_Course/machine-learning-ex5/ex5"

# read dataset
dataset <- readMat("ex5data1.mat")

# Visualise Data ----------------------------------------------------------

cbind(dataset$X,dataset$y) %>% 
  as.data.frame() %>% 
  select(change_in_water_level=V1, water_escaping =V2) %>%
  ggplot(aes(change_in_water_level, water_escaping)) + 
  geom_point(color='red', shape='cross') + geom_smooth(method='lm') +
  ylim(c(-6, 40))
           
# Define Functions --------------------------------------------------------

#linear regression cost function
regressionCost <- function (X, y, theta, lambda) {
  #identify number of training examples
  m <- length(y)
  # add intercept term
  X <- cbind(rep(1,12), dataset$X)
  # calculate cost function with regularisation
  J <- ((1/(2*m)) * sum(((X[, 1] * theta[1] + X[, 2] * theta[2]) - y)^2)) + 
    (lambda/(2*m)) * t(theta[2:length(theta)]) * theta[2:length(theta)]
  return(J)
}

# define linear regression gradient
regressionGrad <- function (X, y, theta, lambda) {
  G <- (lambda/m) * theta
  G[1] <- 0
  grad <- ((1/m) * t(X) %*% (X %*% theta - y)) + G
  return(grad)
}

#test cost function - cost at theta = 1,1 and lambda equal zero should be 303.99 
# with grad of -15.303 and 598.17
regressionCost(X, y, theta=c(1,1), lambda=0)
regressionGrad(X, y, theta =c(1,1), lambda=0)

# Train Linear Classifier -------------------------------------------------

# train linear classifier
cbind(dataset$X,dataset$y) %>% 
  as.data.frame() %>% 
  select(change_in_water_level=V1, water_escaping =V2) %>% 
  lm(water_escaping ~ change_in_water_level, data=.) %>% 
  broom::tidy()

# Plot Learning Curve -----------------------------------------------------


Jtrain <- list()
for (i in 1:length(dataset$X)) {

  # train model to get coefficients
  theta <- cbind(dataset$X[1:i, ],dataset$y[1:i, ]) %>% 
  as.data.frame() %>% 
  select(change_in_water_level=V1, water_escaping =V2) %>% 
  lm(water_escaping ~ change_in_water_level, data=.) %>% 
  coefficients() %>% as.vector()
 
  # calculate cost based on those coefficients
  J <- regressionCost(X=cbind(rep(1, length(dataset$X[1:i,])), dataset$X)[1:i, ],
             y=dataset$y[1:i], theta = theta, lambda=0 ) 
  Jtrain[i] <- J
}





