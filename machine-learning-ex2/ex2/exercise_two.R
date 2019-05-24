########### MACHINE LEARNING COURSE EXERCISE TWO - LOGISTIC REGRESSION ############

# R scripts for exercise 2 (week 3) of coursera machine learning course.

# Read Data & Set Params ---------------------------------------------------------------

# create dataset for plotting

dataset1 <- read_csv("ex2data1.txt", col_names = c('exam_one','exam_two','entry')) #%>% 
            #mutate(entry = ifelse(entry==1,"Admitted", "Not Admitted"))

# create X and y
X <- matrix(c(rep(1,100), dataset1[['exam_one']], dataset1[['exam_two']]), ncol=3) 
y <- dataset1[['entry']]

# set theta 
theta <- c(0,0,0)

# Plot Data ---------------------------------------------------------------

# create plot
ggplot(dataset1, aes(x=exam_one, y=exam_two)) + 
  geom_point(aes(colour=as.factor(entry), shape =as.factor(entry))) + 
  theme_bw() +
  scale_color_manual(values=c('yellow','navy')) +
  labs(color='', shape='', x="Exam One Score", y="Exam Two Score") +
  scale_shape_manual(values=c('circle','cross'))

# Define Functions --------------------------------------------------------

# Define Sigmoid Function
sigmoid <- function(z) {
  1 / (1+exp(-z))
}

# Define Cost Function
costFunction <- function(X, y, theta) {
  j = list()
  k = list()
  for (i in 1:length(y)) {
    j[i] <- -y[i]*log(sigmoid((X[i, 1] * theta[1] + X[i, 2] * theta[2] + X[i, 3] * theta[3])))
    k[i] <- ((1-y[i])*log( 1 - (sigmoid(X[i,1] * theta[1] + X[i, 2] * theta[2] + X[i, 3] * theta[3]))))
  }
  J <- (1/length(y)) * (sum(sum(unlist(j)) - sum(unlist(k))))
  return(J)
}

#test cost function - output should be approximately 0.218
costFunction(theta=c(-24,0.2,0.2), X, y)

#define gradient function
gradFunction <- function(X, y, theta) {
  j=list()
  for (i in 1:length(y)) {
  j[i] <- (sigmoid((X[i, 1] * theta[1] + X[i, 2] * theta[2] + X[i, 3] * theta[3]))) - y[i]
  }
  g <- (unlist(j))
  f <- (X*matrix(rep(g,3), ncol = 3))
  return((1/length(y) *(c(sum(f[, 1]), sum(f[, 2]), sum(f[, 3])))))
}  

#test grad function
gradFunction(X, y, theta = c(-24, 0.2, 0.2))

# Optimising and Plotting Decision Boundary -------------------------------

log_reg <- glm(entry ~ exam_one + exam_two, data = dataset, family='binomial')

slope <- coef(log_reg)[2]/(-coef(log_reg)[3])
intercept <- coef(log_reg)[1]/(-coef(log_reg)[3]) 

ggplot(dataset, aes(x=exam_one, y=exam_two)) + 
  geom_point(aes(colour=as.factor(entry), shape =as.factor(entry))) + 
  theme_bw() +
  scale_color_manual(values=c('yellow','navy')) +
  labs(color='', shape='', x="Exam One Score", y="Exam Two Score") +
  scale_shape_manual(values=c('circle','cross')) +
  geom_abline(slope = slope, intercept = intercept)

# Prediction and Accuracy -------------------------------------------------

sigmoid(predict(log_reg, tibble(intercept = 1, exam_one = 45, exam_two=85)))

# Regularised Logistic Regression -----------------------------------------

dataset2 <- read_csv("ex2data2.txt", col_names = c('test_one','test_two','pass')) 

# Plotting the Data -------------------------------------------------------

ggplot(dataset2, aes(test_one, test_two)) + 
  geom_point(aes(color=as.factor(pass), shape=as.factor(pass))) +
  scale_color_manual(values=c('yellow','navy')) +
  scale_shape_manual(values=c('circle','cross')) + theme_bw() +
  labs(x="Test One Score", y="Test Two Score", color="Passed", shape="Passed")



