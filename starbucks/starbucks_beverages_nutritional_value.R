library(reshape2)
library(ggplot2)
library(ggh4x)
library(ggcorrplot)
library(GGally) # for pairs plot using ggplot framework
library(car) # to calculate the VIF values
library(cmdstanr)
library(bayesplot)
library(rstanarm)
###############################################################
# Get starbucks data from github repo
path <- "https://raw.githubusercontent.com/adityaranade/starbucks/refs/heads/main/data/starbucks-menu-nutrition-drinks.csv"
data0 <- read.csv(path, header = TRUE)
data0 <- read.csv("./../data/starbucks-menu-nutrition-drinks.csv")
###############################################################
# Data processing
colnames(data0) <- c("name", "calories", "fat", 
                     "carbs", "fiber","protein", 
                     "sodium")

# Check the first 6 rows of the dataset
data0 |> head()

# Check the type of data
data0 |> str()

# convert the data to numberic second row onwards
data0$calories <- as.numeric(data0$calories)
data0$fat <- as.numeric(data0$fat)
data0$carbs <- as.numeric(data0$carbs)
data0$fiber <- as.numeric(data0$fiber)
data0$protein <- as.numeric(data0$protein)
data0$sodium <- as.numeric(data0$sodium)

# Check the type of data again
data0 |> str()

# Check the rows which do not have any entries
ind.na <- which(is.na(data0[,2]))
length(ind.na) # 85 NA values

# Check the rows which has NA values
data0[ind.na,]

# exclude the rows which has NA values 
data_nonlog <- data0[-ind.na,]
# Since some of the variables are 0, add 1 to all the datapoints
data_log <- data0[-ind.na,] 
data_log[,-1] <- (data_log[,-1] + 1 ) |> log()
# Select which data to use for the analysis
# data <- data_nonlog
data <- data_log
###############################################################
# Data for histogram
melted_data <- melt(data, id.vars="name")

# Plot the histogram of all the variables
ggplot(melted_data,aes(value))+
  geom_histogram(aes(y = after_stat(density)),bins = 20)+
  facet_grid2(~variable, scales="free")+theme_bw()

###############################################################
# correlation plot of all the variables
corr <- round(cor(data[,-1]), 1)
p.mat <- cor_pmat(mtcars) # correlation p-value
# Barring the no significant coefficient
ggcorrplot(corr, hc.order = TRUE,
           type = "lower", p.mat = p.mat)
# All positive correlation
###############################################################
# Pairs plot which plots the paired 
# scatterplots along with histogram
library(GGally)
ggpairs(data,columns = 2:ncol(data),
        lower = list(continuous = "smooth"))
###############################################################
# Principal component analysis
pc <- prcomp(data[,-(1:2)],
             center = TRUE,
             scale. = TRUE)
attributes(pc)
###############################################################
# Check the factor loadings of the principal components
print(pc)
# Check the summary of the principal components
summary(pc)
# first PC explains approximately 69 % of variation 
# whereas the second PC explains around 17 % of variation
# and they cumulatively explain around 85% of variation
###############################################################
# Check if the multicollinearity issue has been resolved
ggpairs(pc$x,columns = 1:5,
        lower = list(continuous = "smooth"))
# it has been resolved
###############################################################
library(ggbiplot)
g <- ggbiplot(pc,
              obs.scale = 1,
              var.scale = 1,
              # groups = training$Species,
              ellipse = TRUE,
              circle = TRUE,
              ellipse.prob = 0.68)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')
print(g)
###############################################################
# split the data into training and testing data
seed <- 23
set.seed(seed)

ind <- sample(floor(0.8*nrow(data)),
              replace = FALSE)

# Training dataset
data_train <- data[ind,-1]
# Testing dataset
data_test <- data[-ind,-1]
###############################################################
# Multiple linear regression using raw data
model <- lm(calories ~ fat + carbs + fiber + protein + sodium, data = data_train)
summary(model)

# Prediction on the testing dataset
y_pred <- predict(model, data_test)

# Create a observed vs. predicted plot
ggplot(NULL,aes(y_pred,data_test$calories))+geom_point()+
  labs(y = "Observed", x="Predicted")+theme_minimal()+geom_abline()


# Calculate RMSE
rmse <- (y_pred-data_test$calories)^2 |> sum() |> sqrt()
rmse # 102.33

# Check the variance inflation factor
vif_values <- vif(model)
vif_values

# Check the assumptions of the regression model
# par(mfrow = c(2, 2))
# plot(model)
###############################################################
data_pc <- cbind(data[,1:2],pc$x)
# training data
data_pc_train <- data_pc[ind,-1]
# testing data
data_pc_test <- data_pc[-ind,-1]

# Multiple linear regression using PC
model_pc <- lm(calories ~ PC1 + PC2 + PC3, data = data_pc_train)
summary(model_pc)

# Prediction on the testing dataset
y_pred_pc <- predict(model_pc, data_pc_test)

# Create a observed vs. predicted plot
ggplot(NULL,aes(y_pred_pc,data_test$calories))+geom_point()+
  labs(y = "Observed", x="Predicted")+theme_minimal()+geom_abline()

# Calculate RMSE
rmse <- (y_pred_pc-data_pc_test$calories)^2 |> sum() |> sqrt()
rmse # 63.73

# Check the variance inflation factor
vif_values_pc <- vif(model_pc)
vif_values_pc
###############################################################
# Now we can Gaussian process models
###############################################################
# STAN model
# Read the STAN file
file_stan <- "GP_1d.stan"

# Compile stan model
model_stan <- cmdstan_model(stan_file = file_stan,
                            cpp_options = list(stan_threads = TRUE))
model_stan$check_syntax()
###############################################################
# Use raw data predictions
x1 <- data[ind,-(1:2)]
y1 <- data[ind,2]
x2 <- data[-ind,-(1:2)]
y2 <- data[-ind,2]
###############################################################

###############################################################

###############################################################
# Use principal components for predictions
x1 <- pc$x[ind,1] |> as.matrix()
y1 <- data[ind,2]
x2 <- pc$x[-ind,1] |> as.matrix()
y2 <- data[-ind,2]

standata <- list(K = ncol(x1),
                    N1 = nrow(x1),
                    X1 = x1,
                    Y1 = y1,
                    N2 = nrow(x2),
                    X2 = x2,
                    Y2 = y2)
###############################################################
# Start with optimized values (Penalized likelihood)
fit_optim <- model_stan$optimize(data = standata,
                                 seed = seed,
                                 threads =  10)

# fit_optim$output()

fsum_optim <- as.data.frame(fit_optim$summary())

# The optimized parameter would be 
par_ind <- 2:4
opt_pars <- fsum_optim[par_ind,];opt_pars

start_parameters <- rep(list(list(lambda = opt_pars[1,2],
                              sigma = opt_pars[2,2],
                              tau = opt_pars[3,2])),4)
###############################################################
# Run the MCMC with optimized values as the starting values
# Run MCMC
fit <- model_stan$sample(
  data = standata,
  init = start_parameters,
  seed = seed,
  iter_warmup = 500,
  iter_sampling = 500,
  chains = 4,
  parallel_chains = 4,
  refresh = 100,
  threads =  8)

#Summary
fit$summary()

#Check the diagnostic summary to confirm convergence
fit$diagnostic_summary()
fsum <- as.data.frame(fit$summary())

# Save the model
fit$save_object(file = paste0("fit_model.rds"))

# Plot posterior distribution of parameters
# bayesplot::color_scheme_set("gray")
bayesplot::mcmc_dens(fit$draws(c("lambda","sigma","tau")))

#Trace plots
bayesplot::color_scheme_set("brewer-Spectral")
bayesplot::mcmc_trace(fit$draws(c("lambda","sigma","tau")))

# # Replications to check if posterior data is correct
# y_reps <- fit$draws("y_rep", format = "matrix")
# pp_check <- pp_check(y1,y_reps,ppc_dens_overlay)
# pp_check

# Prediction
y_observed <- y2 #observed
y_predicted <-  fsum[max(par_ind)+(1:length(y2)),c(2)] #predicted

ovp_1d <- ggplot(NULL,aes(y_predicted,y_observed))+geom_point()+
  labs(y = "Observed", x="Predicted")+theme_minimal()+geom_abline()

ovp_1d

rmse_log = sqrt(mean((y_observed-y_predicted)^2))
rmse_log
# raw data rmse = 147.9828
# All 5 PC data rmse = 54.94
# First 4 PC data rmse = 52.61
# First 3 PC data rmse = 67.02
# First 2 PC data rmse = 73.77
# First PC data rmse = 55.36

# Check the assumptions of the regression model
# par(mfrow = c(2, 2))
# plot(model_pc)
###############################################################
# To be added
# Gaussian process regression