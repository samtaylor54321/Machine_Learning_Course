
################## COURSERA MACHINE LEARNING COURSE EXERCISE 3 #########################

# Read Data ---------------------------------------------------------------

# read data and tidy
dataset <- readMat("machine-learning-ex3/ex3/ex3data1.mat")

X <- dataset$X
y <- dataset$y

# Display Random Sample of Digits -----------------------------------------

# Sample 100 random images
idx <- sample(5000, 100)
img <- X[idx,]

# Create blank matrix to hold all 100 images
canvas <- matrix(1, nrow=10*(20+1), ncol=10*(20+1))
count <- 1
for(i in seq(1+1, 10*(20+1), 20+1)){
  for(j in seq(1+1, 10*(20+1), 20+1)){
    canvas[i:(i+20-1), j:(j+20-1)] <- matrix(img[count,], nrow=20)
    count <- count + 1
  }
}
image(canvas, col=gray(seq(1,0,length=100)))

# Define Functions --------------------------------------------------------

# sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# cost function
costFunction <- function(X, y, theta, lambda) {
  
  m <- length(y)
  # calculate h_theta
  h_theta <- sigmoid(X %*% theta)
  # calculate J (cost function)
  J <- (1/m) * ((sum((-1*y)*log(h_theta))) - (sum((1-y) * log(1-h_theta)))) + lambda/(2*m) * (sum(theta[-1]^2))

  # calculate gradient
  x <-  X * as.vector(h_theta - y)
  k <- list()  
  for (i in 1:dim(x)[2]) {
    k[i] <- sum(x[, i])
  }
  temp <- theta
  temp[1] <- 0
  grad <- (unlist(k) * (1/m)) + (lambda/m)  * temp
  
  return(list(gradient =grad, cost=J))
}

# test cost function
costFunction(X_t, y_t, theta_t, lambda_t)


# predict using multinomial regression
dataset <- as.data.frame(cbind(X, y))

multi_model <- nnet::multinom(V401~., dataset, MaxNWts = 5000)

dataset$pred <- predict(multi_model, newdata = dataset)

multiclass_logistic_regression 

summary(multi_model)

