
################ Machine Learning Course Exerise 4 ###############

# Load Required Packages --------------------------------------------------

library(R.matlab)
library(tidyverse)

# Load Data ---------------------------------------------------------------

# set correct working directory
wd <- "/Users/samtaylor/Documents/Git/Machine_Learning_Course/Machine_Learning_Course"
setwd(wd)

# read in data - check this hasn't been pushed to icloud otherwise it can't be read by R.
dataset <- readMat("machine-learning-ex4/ex4/ex4data1.mat")

# load params and unroll
params <- readMat("machine-learning-ex4/ex4/ex4weights.mat")
nn_params <- as.vector(matrix(c(params$Theta1,params$Theta2), ncol=1))

# set network arcitecture
input_layer_size  <-  400
hidden_layer_size <- 25   
num_labels <- 10          

# Display Data ------------------------------------------------------------

# Sample 100 random images
idx <- sample(5000, 100)
img <- dataset$X[idx,]

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

# Feedforward and Cost Function ----------------------------------------------------

# unroll params 
Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], 
                 nrow = hidden_layer_size, ncol=(input_layer_size+1))

Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))) : length(nn_params)],
       nrow = num_labels, ncol = (hidden_layer_size +1))

# define sigmoid function
sigmoid <- function(z) {
  1 / (1+exp(-z))
}

# define sigmoid gradient functon 
sigmoidGradient <- function (z) {
  sigmoid(z) * (1 - sigmoid(z));
}

sigmoidGradient(c(-1, -0.5, 0, 0.5, 1))

# define NN cost function
nnCostFunction <- function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda) {
  
  # Part One - forward propogation
  
  #feed forward through the network
  m <- dim(X)[1]
  a1 <- cbind(rep(1,m),X)
  z2 <- a1 %*% t(Theta1)
  a2 <- sigmoid(z2)
  a2 <- matrix(c(rep(1, dim(a2)[1]), a2), nrow=dim(a2)[1])
  z3 <- a2 %*% t(Theta2)
  a3 <- sigmoid(z3)
  hThetaX <- a3
  
  # tidy yVec
  yVec <- matrix(rep(0, 50000), ncol=10)
  for (i in 1:m) {
    yVec[i, y[i]] <- 1
  }

  # define cost 
  J <- (1/m) * sum(sum(-1 * yVec * log(hThetaX) - (1-yVec) * log(1-hThetaX))) 
  # define regulator to add  to cost
  regulator <- (sum(sum(Theta1[,-1]^2)) + sum(sum(Theta2[,-1]^2))) * (lambda/(2*m))
  #
  J <- J + regulator
  
  # implement back prop
  
  for (t in 1:m) {
    a1 <- as.vector(c(1, dataset$X[5000, ]))
    z2 <- Theta1 %*% a1
    a2 <- as.vector(c(1,sigmoid(z2)))
    z3 <- Theta2 %*% a2
    a3 <- sigmoid(z3) 
    
    yy <- as.numeric((1:num_labels) == dataset$y[5000])
    
    delta_3 <- a3 - yy
    
    delta_2 <- ((t(Theta2)) %*% delta_3) * as.vector(c(1,sigmoidGradient(z2)))
    delta_2 <- matrix(delta_2[-1])
    
    Theta1_grad <- matrix(rep(0, dim(Theta1)[1] * dim(Theta1)[2]), nrow=dim(Theta1)[1])    
    Theta2_grad <- matrix(rep(0, dim(Theta2)[1] * dim(Theta2)[2]), nrow=dim(Theta2)[1]) 
    
    Theta1_grad <- Theta1_grad + delta_2 %*% a1
    Theta2_grad <- Theta2_grad + delta_3 %*% a2
  }
  
  Theta1_grad <- (1/m) * Theta1_grad + (lambda/m) * matrix(c(rep(0, dim(Theta1)[1]), Theta1[, -1]),nrow=dim(Theta1)[1]) 
  
  Theta2_grad <- (1/m) * Theta2_grad + (lambda/m) * matrix(c(rep(0, dim(Theta2)[1]), Theta2[, -1]),nrow=dim(Theta2)[1]) 
  
  # unroll grad
  
  grad <- matrix(c(Theta1_grad, Theta2_grad), ncol=1)
  
  params <- list(cost=J, grad = grad)
  return(params)

}  

grads <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, dataset$X, dataset$y, lambda = 0 )



