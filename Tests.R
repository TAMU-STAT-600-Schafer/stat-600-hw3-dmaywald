# This is a script to save your own tests for the function
source("FunctionsLR.R")



# Make sure the libraries utilized in Tests.R are installed
requiredPackages = c('profvis',
                     'microbenchmark',
                     'testthat')

for (pack in requiredPackages) {
  if (!require(pack, character.only = TRUE))
    install.packages(pack)
  library(pack, character.only = TRUE)
}

####### Speed tests for functions used in FunctionsLR.R ########################

library(microbenchmark)



# Manually calculated frobenius norm to compare with norm(., "F") function
frob_norm <- function(X){
  temp = 0
  for (i in 1:nrow(X)) {
    for (j in 1:ncol(X)){
      temp = temp + X[i,j]^2
    }
  }
  return(as.numeric(sqrt(temp)))
}

frob_norm_2 <- function(X){
  sqrt(sum(diag(crossprod(X,X))))
}


X_nrow = 2000
X_ncol = 100

X <- matrix(rnorm(n = X_nrow*X_ncol, mean = 0, sd = 10), nrow = X_nrow)


# Make sure the functions return the same value
test_that("Test Equality of Frobenius Norm Function",
          {
            expect_equal(frob_norm(X), norm(X,"F"))
            expect_equal(frob_norm_2(X), norm(X,"F"))
          }         
)


# See which function is faster

res <- microbenchmark(
  frob_norm(X),
  frob_norm_2(X),
  norm(X, "F")
) # norm(X, "F") seems to be the fastest. 



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

X <- cbind(rep(1,nrow(X)), X)
Xt <- cbind(rep(1,nrow(Xt)), Xt)

# Source the LR function
source("FunctionsLR.R")

out <- LRMultiClass(X, Y, Xt, Yt, eta = .5, lambda = 1)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o')
plot(out$error_test, type = 'o')



### Try my own data

p = 20
K = 3
ntrain = 6000
ntest = 1500
beta_star = matrix(sample(-10:10, size = (p+1)*(K), replace = T), nrow = p+1)
Xtrain = cbind(rep(1,ntrain),matrix(rnorm(ntrain*p), nrow = ntrain))
Xtest = cbind(rep(1,ntest),matrix(rnorm(ntest*p), nrow = ntest))
expXtrainB <- exp(Xtrain%*%beta_star) # no error for now
Ptrain <- expXtrainB/rowSums(expXtrainB)
Ytrain <- vector(mode = "double", length = ntrain)

expXtestB <- exp(Xtest%*%beta_star) # no error for now
Ptest <- expXtestB/rowSums(expXtestB)
Ytest <- vector(mode = "double", length = ntest)

for (i in 1:ntrain) {
  Ytrain[i] <- sample(x = 0:(K-1), size = 1, prob = Ptrain[i,])
  
}

for (i in 1:ntest) {
  Ytest[i] <- sample(x = 0:(K-1), size = 1, prob = Ptest[i,])
}