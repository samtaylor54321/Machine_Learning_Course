######################## Machine Learning Exercise 1 ########################

# Read  Data --------------------------------------------------------------

dataset <- read_csv("/Users/samtaylor/Desktop/dataset.csv", col_names = c("Population","Profit")) 

# Plot Data ---------------------------------------------------------------

ggplot(dataset, aes(x=Population,y=Profit)) + geom_point(colour ='red', shape=4) +
  xlab("Population (10,000s)") + ylab("Profit ($10,000s)") + theme_bw()

# Gradient Descent ----------------------------------------------

# add intercept term to the model
X <- dataset[, 1] %>% mutate(intercept=1)
# ensure that columns are in the correct order for the model
if (colnames(X[, 1]) != 'intercept') {
X <- X[c('intercept','Population')] %>%  as.matrix()
}
# set y vector    
y = dataset[,2] %>% as.matrix()

# set alpha, iterations and theta
alpha =0.01
num_iters = 1500
theta = c(0,0)

# assign cost function
cost_function <- function (X, y, theta) {
  m <- length(y)
  j = list()
  for (i in 1:m) {
    j[i] <- (1/(2*m)) * sum( ((theta[1] + theta[2] * X[i,2]) - y[i])^2)
  }
  cost <- sum(unlist(j))
  return(cost)
}
# test cost function
cost_function(X, y, c(-1,2))

# build gradient descent function
gradient_descent <- function(X, y, theta, alpha, num_iters) {
  # set up params
  m = length(y)
  J <- vector("list", num_iters)
  # loop through num iters of gradient descent.
  for (n in 1:num_iters) {
      # calculate theta(1)
      T1 = list()
      for (i in 1:m) {
        T1[i] <- ((theta[1] * X[i, 1] + theta[2] * X[i, 2]) - y[i]) 
      }
      #calculate theta(2)
      T2 = list()
      for (i in 1:m) {
          T2[i] <- (((theta[1] * X[i, 1] + theta[2] * X[i, 2]) - y[i])*X[i,2]) 
      }
      # summarise the above
      t1 <- sum(unlist(T1))
      t2 <- sum(unlist(T2))
      # update cost function according to learning rate  
      theta[1] <- theta[1] - (alpha/m) * (t1)
      theta[2] <-  theta[2] - (alpha/m) * (t2) 
      # store values
      J[n] <- cost_function(X, y, theta)
    }
  return(list(cost_function = J[num_iters], theta = theta))
}
gd <- gradient_descent(X, y, theta, alpha, num_iters)
# check that the function has run correctly and print values
( 1 * gd$theta[1] + 3.5 * gd$theta[2]) * 10000
( 1 * gd$theta[1] + 7 * gd$theta[2]) * 10000


theta0_vals <- seq(-10, 10, length.out = 100)
theta1_vals <- seq(-1, 4, length.out=100)




