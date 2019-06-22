################ MACHINE LEARNING COURSE -EXERCISE 6 ##################

#Support Vector Machines

# Initial Set Up ------------------------------------------------------------------

# load packages
library(R.matlab)
library(e1071)

# set theme
theme_set(theme_bw())

# set wd
wd <- "/Users/samtaylor/Documents/Git/Machine_Learning_Course/Machine_Learning_Course/machine-learning-ex6/ex6"
setwd(wd)

# read dataset
dataset1 <- readMat("ex6data1.mat")
dataset2 <- readMat("ex6data2.mat")
dataset3 <- readMat("ex6data3.mat")

# Helper Functions --------------------------------------------------------

# define plot function
ex6plot <- function(dataset) {
  cbind(dataset$X,dataset$y) %>% 
    as.data.frame() %>% 
    ggplot(aes(V1, V2)) + 
    geom_point(aes(color=as.factor(V3), 
                   shape=as.factor(V3)), 
               show.legend=F) +
    labs(x='',
         y='') +
    scale_color_manual(values=c('yellow','navy'))  +
    scale_shape_manual(values=c('circle','cross'))
}

# Dataset One -------------------------------------------------------------

# plot data
ex6plot(dataset2)
  
# generate SVM and plot - linear
model_lin <- svm(V3 ~ V2 + V1, data = cbind(dataset1$X,dataset1$y) %>% 
               as.data.frame() , 
             kernel='linear', cost=1, scale=FALSE,
             type='C-classification', probability=T)

plot(model_lin, cbind(dataset1$X,dataset1$y) %>% 
       as.data.frame() )

# write gaussian function
gaussianKernal <-function(x1, x2, sigma) {
  exp(-sum((x1-x2)^2) / (2*sigma^2))
}

# test gaussian function (should return approx 0.3246525)
gaussianKernal(x1=c(1,2,1), x2=c(0,4,-1), sigma=2)

# Dataset Two -------------------------------------------------------------

# plot data
ex6plot(dataset2)

# generate svm and plot
model_rbf <- svm(V3 ~ V2 + V1, data = cbind(dataset2$X,dataset2$y) %>% 
                   as.data.frame() , 
                 kernel='radial',
                 gamma=1,
                 cost=5,
                 type='C-classification', probability=T)

plot(model_rbf, cbind(dataset2$X,dataset2$y) %>% 
       as.data.frame() )

# Dataset 3 ---------------------------------------------------------------

# plot data
ex6plot(dataset3)

# set up data
train <- cbind(dataset3$X,dataset3$y) %>% 
  as.data.frame()

cv_data <- cbind(dataset3$Xval,dataset3$yval) %>% 
  as.data.frame()

# define param space
c <- c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
gamma <- c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
params <- crossing(c, gamma)

# cv model
accuracy_score <- list() 

for (i in 1:dim(params)[1]) {
  # loop through models
  model <- svm(V3 ~ V2 + V1, 
                   data = cbind(dataset3$X,dataset3$y) %>% 
                     as.data.frame() , kernel='radial', 
                   gamma = params$gamma[i],
                   cost=params$c[i],
                   type='C-classification', 
                   probability=T)
  # get accuracy score
  accuracy_score[i] <- mean(dataset3$yval == as.numeric(predict(model, cv_data))-1)
}

# record error and plot
params$accuracy <- unlist(accuracy_score)

params %>% 
  arrange(desc(accuracy))

# plot best model
best_model <- svm(V3 ~ V2 + V1, 
    data = cbind(dataset3$X,dataset3$y) %>% 
      as.data.frame() , kernel='radial', 
    gamma = 3,
    cost=1,
    type='C-classification', 
    probability=T)

plot(best_model, train)




