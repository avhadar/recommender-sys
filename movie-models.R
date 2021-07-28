# title: "Recommender Systems: Predicting movie ratings with MovieLens"
# author: "Alex V. Hadar"
# date: "7/28/2021"


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColor.Brewer", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(Matrix)
library(recosystem)
library(RColorBrewer)
library(tinytex)
# install tex packages
if (tinytex::is_tinytex() == FALSE){
  tinytex::install_tinytex()
}

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                            title = as.character(title),
#                                            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# store the edx (and validation) data for later use
saveRDS(edx, file = "edx.Rda")
saveRDS(validation, file = "vali.Rda")
# clean up
rm(edx, validation)
gc()


##########################################################
# Create training set, test set (from edx set)
##########################################################
edx <- readRDS("edx.Rda")
# keep only the rating information
edx.user.ratings <- edx %>% select(userId, movieId, rating)
# head(edx.user.ratings)

# test set will be 10% of edx data
# set.seed(1, sample.kind="Rounding") # set before, so not needed again; if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx.user.ratings$rating, times = 1, p = 0.1, list = FALSE)
train <- edx.user.ratings[-test_index,]
temp <- edx.user.ratings[test_index,]

# Make sure userId and movieId in test set are also in edx training set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into edx training set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# clean up
rm(test_index, temp, edx.user.ratings, removed)
gc()


##########################################################
# Explore the Movielens data
##########################################################
# general structure: number of rows/columns, data types
str(edx)
head(edx)

# get number of unique users and movies
all.users <- unique(edx$userId)
length(all.users)  # 69878
all.movies <- unique(edx$movieId)
length(all.movies) # 10677

# Exploratory data analysis - Plots
# rating count per value
edx %>% group_by(rating) %>% summarize(count = n()) %>% ggplot(aes(x = rating, y = count)) + geom_line()
edx %>% ggplot(aes(x = as.factor(rating))) + geom_bar(aes(fill = as.factor(rating)))
# number of ratings per user
user.ratings <- edx %>% group_by(userId) %>% summarize(n= n(), avg.rating = mean(rating))
user.ratings %>% ggplot(aes(x = userId, y = n)) + geom_line()
# number of ratings per movie
movie.ratings <- edx %>% group_by(movieId) %>% summarize(n= n(), avg.rating = mean(rating))
movie.ratings %>% ggplot(aes(x = movieId, y = n)) + geom_line()

# Top genres and movies
# explore the genre counts for selected genres
genres <- c("Drama", "Comedy", "Thriller", "Romance", "Adventure", "Sci-Fi")
sapply(genres, function(g){
  sum(str_detect(edx$genres, g))
})

# explore the movie top counts
most_rated <- edx %>% group_by(movieId, title) %>% 
  summarize(count = n()) %>% arrange(desc(count))
head(most_rated, n=10)

# clean up
rm(edx)
rm(movie.ratings, user.ratings)
gc()


##########################################################
# Models
##########################################################
# First model: Mean overall rating
##########################################################
# helper functions for error computation - get metrics for model performance
# Root Mean Square Error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Mean Absolute Error
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}


mu <- mean(train$rating)
mu

rmse.model1 <- RMSE(test$rating, mu)
rmse.model1

mae.model1 <- MAE(test$rating, mu)
mae.model1

rmse_results <- data_frame(method = "Mean value", RMSE = rmse.model1, MAE = mae.model1)


##########################################################
# Second model: Movie effect
##########################################################
# mu <- mean(train$rating) 
movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

rmse.model2 <- RMSE(test$rating, predicted_ratings)
mae.model2 <- MAE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effects Model",
                                     RMSE = rmse.model2, MAE = mae.model2 ))

# rmse_results %>% knitr::kable()


##########################################################
# Third model: Movie and user effects
##########################################################
train%>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# lm is VERY slow here -> do not run this code
# lm(rating ~ as.factor(movieId) + as.factor(userId))
user_avgs <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test%>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

rmse.model3 <- RMSE(test$rating, predicted_ratings)
mae.model3 <- MAE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = rmse.model3, MAE = mae.model3 ))
# rmse_results %>% knitr::kable()


##########################################################
# Fourth and fifth model: Regularized models
##########################################################
# for movie effects
# choose lambda using cross validation
lambdas <- seq(0, 10, 0.25)
mu <- mean(train$rating)
just_the_sum <- train %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(test$rating, predicted_ratings))
})
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)] 
lambda # 1.75 for movie effects

# penalty term -> see above how it is chosen (cross validation)
# use lambda from above directly
mu <- mean(train$rating)
movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

rmse.model4 <- RMSE(test$rating, predicted_ratings)
mae.model4 <- MAE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effects Model",  
                                     RMSE = rmse.model4, MAE = mae.model4 ))
# rmse_results %>% knitr::kable()


# for movie and user effects
# choose lambda using cross validation
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test%>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(test$rating, predicted_ratings))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda # 4.5 for movie and user effects

# penalty term -> see above how it is chosen (cross validation)
# use lambda from above directly
mu <- mean(train$rating)
movie_reg_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

user_reg_avgs <- train %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- test %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

rmse.model5 <- RMSE(test$rating, predicted_ratings)
mae.model5 <- MAE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = rmse.model5, MAE = mae.model5 ))
# rmse_results %>% knitr::kable()

# clean up - remove unneeded variables
rm(predicted_ratings, just_the_sum, movie_avgs, movie_reg_avgs, user_avgs, user_reg_avgs, lambda, lambdas, rmses, mu)
gc()


##########################################################
# Sixth model: Matrix factorization model
##########################################################
# use recosystem library for matrix factorization and predictions
train_set = data_memory(user_index = train$userId, item_index = train$movieId, 
                        rating = train$rating, index1 = TRUE)
test_set  = data_memory(user_index = test$userId, item_index = test$movieId, 
                        rating = NULL, index1 = TRUE)
r = Reco()
# dim  -> number of latent factors: use 20, then tune
# r$train(train_set, opts = c(dim = 20, nthread = 1, niter = 20))
# Return results as R vector
# predicted_ratings <- r$predict(test_set, out_memory())
# get RMSE
# rmse.model6.notune <- RMSE(test$rating, predicted_ratings)
# rmse.model6.notune

set.seed(1, sample.kind="Rounding")
# model tuning for: number of latent factors (dim), learning rate (lrate)
opts = r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 1, niter = 10))
# opts
# opts$min
# use the tuned values for training
r$train(train_set, opts = c(opts$min, nthread = 1, niter = 20))

# Return results as R vector
predicted_ratings <- r$predict(test_set, out_memory())
head(predicted_ratings, 10)
# r$model
# check for NA values
# sum(is.na(predicted_ratings))

# get RMSE
rmse.model6 <- RMSE(test$rating, predicted_ratings)
mae.model6 <- MAE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix Factorization Model",  
                                     RMSE = rmse.model6, MAE = mae.model6))
# rmse_results %>% knitr::kable()


##########################################################
# Model results
##########################################################
# Evaluate models and select the final model
##########################################################
# get RMSE for all models and compare
rmse_results %>% knitr::kable()
# choose the model with the lowest RMSE: the Matrix factorization model

# clean up
rm(train, test)
gc()


##########################################################
# Results on validation data (final model)
##########################################################
# Evaluate chosen model on the validation set
##########################################################
# Use this only if you want to evaluate the previously trained Matrix factorization model on the validation data
# We suggest to use instead the approach described after this commented code section: 
# re-train the model using the full edx data, then evaluate model performance on the validation data

# # load the validation data
# validation <- readRDS("vali.Rda")
# # create the training and test datasets: (optional) use the entire edx dataset for re-training
# # use validation dataset for final test and RMSE 
# test_set  = data_memory(user_index = validation$userId, item_index = validation$movieId, rating = NULL, index1 = TRUE)
# ## Return results as R vector
# predicted_ratings <- r$predict(test_set, out_memory())
# # head(predicted_ratings, 10)
# # check for NA values
# # sum(is.na(predicted_ratings))
# 
# # get RMSE
# rmse.final <- RMSE(validation$rating, predicted_ratings)
# rmse.final # 0.7859647


# tune the model: retrain on full edx dataset, then test on validation data for final RMSE
# load the datasets
edx <- readRDS("edx.Rda")
validation <- readRDS("vali.Rda")
# create the training and test datasets
train_set = data_memory(user_index = edx$userId, item_index = edx$movieId, 
                        rating = edx$rating, index1 = TRUE)
test_set  = data_memory(user_index = validation$userId, item_index = validation$movieId, 
                        rating = NULL, index1 = TRUE)

# clean up - the validation data is still needed for RMSE!!
rm(edx)
gc()

# train the Matrix factorization model on the full edx dataset, with parameter tuning
r = Reco()
set.seed(1, sample.kind="Rounding")
# model tuning for: number of latent factors (dim), learning rate (lrate)
opts = r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 1, niter = 10))
# display the chosen parameter values
opts$min
# use the tuned values for training
r$train(train_set, opts = c(opts$min, nthread = 1, niter = 20))
# Return results as R vector
predicted_ratings <- r$predict(test_set, out_memory())
# head(predicted_ratings, 10)

# r$model
# check for NA values: 
# sum(is.na(predicted_ratings)) # should be 0 - no missing predictions

##########################################################
# Generate final RMSE
##########################################################
# get RMSE
rmse.final.retrained <- RMSE(validation$rating, predicted_ratings)
mae.final.retrained <- MAE(validation$rating, predicted_ratings)
# display the final metrics for Matrix factorization
rmse.final.retrained # 0.7825732
mae.final.retrained

# clean up
rm(validation)
gc()

