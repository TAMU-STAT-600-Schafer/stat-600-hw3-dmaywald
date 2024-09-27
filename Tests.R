# This is a script to save your own tests for the function
source("FunctionsLR.R")



# Make sure the libraries utilized in Tests.R are installed
requiredPackages = c('profvis', 'microbenchmark', 'testthat')

for (pack in requiredPackages) {
  if (!require(pack, character.only = TRUE))
    install.packages(pack)
  library(pack, character.only = TRUE)
}

####### Speed tests for functions used in FunctionsLR.R ########################

library(microbenchmark)

# Manually calculated frobenius norm to compare with norm(., "F") function
frob_norm <- function(X) {
  temp = 0
  for (i in 1:nrow(X)) {
    for (j in 1:ncol(X)) {
      temp = temp + X[i, j] ^ 2
    }
  }
  return(as.numeric(sqrt(temp)))
}

frob_norm_2 <- function(X) {
  sqrt(sum(diag(crossprod(X, X))))
}


X_nrow = 2000
X_ncol = 100

X <- matrix(rnorm(n = X_nrow * X_ncol, mean = 0,sd = 10), nrow = X_nrow)


# Make sure the functions return the same value
test_that("Test Equality of Frobenius Norm Function", {
  expect_equal(frob_norm(X), norm(X, "F"))
  expect_equal(frob_norm_2(X), norm(X, "F"))
})


# See which function is faster

res <- microbenchmark(frob_norm(X), frob_norm_2(X), norm(X, "F")) # norm(X, "F") seems to be the fastest.



########################## Check code for correctness ##########################
# Load the letter data
#########################
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")

Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")

Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

X <- cbind(rep(1, nrow(X)), X)
Xt <- cbind(rep(1, nrow(Xt)), Xt)

# Source the LR function
source("FunctionsLR.R")

out1 <- LRMultiClass(X, Y, Xt, Yt, eta = .1, lambda = 1)
out2 <- LRMultiClass2(X, Y, Xt, Yt, eta = .1, lambda = 1)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out1$objective, type = 'o')
plot(out1$error_train, type = 'o')
plot(out1$error_test, type = 'o')


Rprof(gc.profiling = TRUE)
invisible(LRMultiClass(X, Y, Xt, Yt))
Rprof(NULL)
summaryRprof()$by.total


Rprof(gc.profiling = TRUE)
invisible(LRMultiClass2(X, Y, Xt, Yt))
Rprof(NULL)
summaryRprof()$by.total


######################## Try my own data #######################################

p = 30
K = 20
ntrain = 2000
ntest = 500

# Make true solution beta_star
beta_star <- matrix(sample(-10:10, size = (p + 1) * (K), replace = T), nrow = p + 1)

# Make X random normal matrices
Xtrain <- cbind(rep(1, ntrain), matrix(rnorm(ntrain * p), nrow = ntrain))
Xtest <- cbind(rep(1, ntest), matrix(rnorm(ntest * p), nrow = ntest))

# Calculate probabilities for Y's classification
expXtrainB <- exp(Xtrain %*% beta_star) # no error for now
Ptrain <- expXtrainB / rowSums(expXtrainB)


expXtestB <- exp(Xtest %*% beta_star) # no error for now
Ptest <- expXtestB / rowSums(expXtestB)


# Initialize Y's
Ytrain <- vector(mode = "double", length = ntrain)
Ytest <- vector(mode = "double", length = ntest)



# Assign classifications to Y based on probabilities calculated above
for (i in 1:ntrain) {
  Ytrain[i] <- sample(x = 0:(K - 1), size = 1, prob = Ptrain[i, ])
  # Ytrain[i] <- which.max(Ptrain[i,])
  
}

for (i in 1:ntest) {
  Ytest[i] <- sample(x = 0:(K - 1), size = 1, prob = Ptest[i, ])
  # Ytest[i] <- which.max(Ptest[i,])
}

# Run LRMultiClass function(s)
out <- LRMultiClass(Xtrain, Ytrain, Xtest, Ytest)
out2 <- LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest)

# Microbenchmark 2 different versions of my function for speed comparisons
library(microbenchmark)

microbenchmark(
  LRMultiClass(Xtrain, Ytrain, Xtest, Ytest),
  LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest),
  times = 10
)

# Summary profiles of 2 different versions of function
Rprof(gc.profiling = TRUE)
invisible(LRMultiClass(Xtrain, Ytrain, Xtest, Ytest))
Rprof(NULL)
summaryRprof()$by.total


Rprof(gc.profiling = TRUE)
invisible(LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest))
Rprof(NULL)
summaryRprof()$by.total

# Visual profile of 2 different versions of function
library(profvis)
profvis(LRMultiClass(Xtrain, Ytrain, Xtest, Ytest))
profvis(LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest))

################## Different values of lambda, eta #############################

# Generate different values of lambda, where maximum lambda is evaluated on the norm ||t(X)%*%Y/n||_inf 
lambdas = c(seq(0, .5, length.out = 2), seq(1, max(abs((t(Xtrain) %*% Ytrain)/ntrain)), length.out = 3))
etas = seq(.1, .9, length.out = 5)

# Examine convergence for different values of lambda and eta
for (lambda in lambdas){
  for (eta in etas){
    out <- LRMultiClass(Xtrain, Ytrain, Xtest, Ytest, eta = eta, lambda = lambda)
    
    # plot(out$objective, type = 'o', main = paste("L = ", lambda, ", Eta = ", eta))
    # plot(out$error_train, type = 'o', main = paste("L = ", lambda, ", Eta = ", eta))
    plot(out$error_test, type = 'o', main = paste("L = ", lambda, ", Eta = ", eta))
    
  }
} 
# For this data, it seems that eta can be pushed fairly high (up to .8-.9). 
# However, the best value of Lambda was lambda = 0 (in terms of error/objective values)
# This is as expected since Lambda = 0 corresponds to the least squares solution.


#################### Different Data Sizes ######################################

# Generate different values of p and K
ps = c(1,2,20,30)
Ks = c(1,5,10,30)
ntrain = 1000
ntest = 250

# For each different p and K, generate data as before, run LRMultiClass function and
# examine convergence
for (p in ps) {
  for (K in Ks) {
    beta_star = matrix(sample(-10:10, size = (p + 1) * (K), replace = T), nrow = p + 1)
    Xtrain <- cbind(rep(1, ntrain), matrix(rnorm(ntrain * p), nrow = ntrain))
    Xtest <- cbind(rep(1, ntest), matrix(rnorm(ntest * p), nrow = ntest))
    expXtrainB <- exp(Xtrain %*% beta_star + rnorm(ntrain, mean = 0, sd = 1))
    Ptrain <- expXtrainB / rowSums(expXtrainB)
    Ytrain <- vector(mode = "double", length = ntrain)
    
    expXtestB <- exp(Xtest %*% beta_star + rnorm(ntest, mean = 0, sd = 1)) 
    Ptest <- expXtestB / rowSums(expXtestB)
    Ytest <- vector(mode = "double", length = ntest)
    
    for (i in 1:ntrain) {
      Ytrain[i] <- sample(x = 0:(K - 1),
                          size = 1,
                          prob = Ptrain[i, ])
      # Ytrain[i] <- which.max(Ptrain[i,])
      
    }
    
    for (i in 1:ntest) {
      Ytest[i] <- sample(x = 0:(K - 1),
                         size = 1,
                         prob = Ptest[i, ])
      # Ytest[i] <- which.max(Ptest[i,])
    }
    
    out <- LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest)
    # plot(out$objective, type = 'o', main = paste("p = ", p, ", K = ", K))
    # plot(out$error_train, type = 'o', main = paste("p = ", p, ", K = ", K))
    plot(out$error_test, type = 'o', main = paste("p = ", p, ", K = ", K))
  }
}


################## Check for mismatched input dimensions ###################

p = 20
K = 30
ntrain = 2000
ntest = 500


# Generate Data
beta_star = matrix(sample(-10:10, size = (p + 1) * (K), replace = T), nrow = p + 1)
Xtrain <- cbind(rep(1, ntrain), matrix(rnorm(ntrain * p), nrow = ntrain))
Xtest <- cbind(rep(1, ntest), matrix(rnorm(ntest * p), nrow = ntest))

# Generate probabilities
expXtrainB <- exp(Xtrain %*% beta_star + rnorm(ntrain, mean = 0, sd = 1))
Ptrain <- expXtrainB / rowSums(expXtrainB)

expXtestB <- exp(Xtest %*% beta_star + rnorm(ntest, mean = 0, sd = 1)) 
Ptest <- expXtestB / rowSums(expXtestB)

# Generate samples of Y with probabilities
Ytrain <- vector(mode = "double", length = ntrain)
Ytest <- vector(mode = "double", length = ntest)

for (i in 1:ntrain) {
  Ytrain[i] <- sample(x = 0:(K - 1),
                      size = 1,
                      prob = Ptrain[i, ])
  # Ytrain[i] <- which.max(Ptrain[i,])
  
}

for (i in 1:ntest) {
  Ytest[i] <- sample(x = 0:(K - 1),
                     size = 1,
                     prob = Ptest[i, ])
  # Ytest[i] <- which.max(Ptest[i,])
}

# Does sd of random initialization matter?
beta_init = matrix(rnorm((p + 1) * K, mean = 0, sd = 1), nrow = p + 1) 
out <- LRMultiClass2(Xtrain, Ytrain, Xtest, Ytest, beta_init = beta_init)

library(testthat)

test_that("Expect errors/warnings in mismatched dimensions", 
          {
            expect_error(LRMultiClass(Xtrain, Ytrain, Xtest[-1,], Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain, Ytrain, Xtest, Ytest[-1], beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain, Ytrain, Xtest[,-1], Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain, Ytrain, Xtest[,-2], Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain, Ytrain[-1], Xtest, Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain[-1,], Ytrain, Xtest, Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain[,-1], Ytrain, Xtest, Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain[,-2], Ytrain, Xtest, Ytest, beta_init = beta_init))
            expect_error(LRMultiClass(Xtrain, Ytrain, Xtest, Ytest, beta_init = beta_init[-1,]))
            
            # Function is able to run if column dimension of beta_init doesn't match implied number
            # of classes within Ytrain. I gave a warning if this happens.
            expect_warning(LRMultiClass(Xtrain, Ytrain, Xtest, Ytest, beta_init = beta_init[,-1])) 
          })

