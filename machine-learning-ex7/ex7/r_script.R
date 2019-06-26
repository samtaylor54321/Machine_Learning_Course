##############  WEEK 7 EXERCISES - MACHINE LEARNING COURSE ###################

# responses for week 7 exercises in machine learning coursera course.v 

# Set Up ------------------------------------------------------------------

# load required packages
library(R.matlab)
library(cluster)
library(factoextra)
library(tidyverse)
library(ggbiplot)

# setwd and load data
setwd("/Users/samtaylor/Documents/Git/Machine_Learning_Course/Machine_Learning_Course/machine-learning-ex7/ex7")
dataset2 <- readMat("ex7data2.mat") %>% as.data.frame()
dataset1 <- readMat("ex7data1.mat") %>% as.data.frame()

# other setup
theme_set(theme_bw())

# K Means Clustering ------------------------------------------------------

clust <- kmeans(dataset2, centers = 3, nstart = 25)
dataset2$X <- clust$cluster

# visual data
ggplot(dataset2, aes(x=X.1,y=X.2, color=as.factor(X))) + 
  geom_point() +
  geom_point(aes(x=6.03, y=3.00), colour="blue", size=5, shape='triangle') +
  geom_point(aes(x=1.95, y=5.03), colour="blue", size=5, shape='triangle') +
  geom_point(aes(x=3.04, y=1.02), colour="blue", size=5, shape='triangle')

fviz_cluster(clust, data = dataset2)

# PCA ---------------------------------------------------------------------

ggplot(dataset1,aes(x=X.1, y=X.2)) + geom_point()
pca <- prcomp(dataset1, center = TRUE, scale. = TRUE)
summary(pca)



