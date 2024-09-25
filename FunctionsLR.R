# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  
  if(!(all(X[,1] == rep(1,nrow(X))))){
    stop("First column of X is not a column of all 1s. Stopping Execution")
  }
  
  if(!(all(Xt[,1] == rep(1,nrow(Xt))))){
    stop("First column of Xt is not a column of all 1s. Stopping Execution")
  }
  
  # Check for compatibility of dimensions between X and Y
  if(!(nrow(X) == nrow(Y))){
    stop("X and Y do not have the same number of rows. Check dimension compatability")
  }
  
  # Check for compatibility of dimensions between Xt and Yt
  if(!(nrow(Xt) == nrow(Yt))){
    stop("Xt and Yt do not have the same number of rows. Check dimension compatability")
  }
  
  # Check for compatibility of dimensions between X and Xt
  if(!(ncol(X) == ncol(Xt))){
    stop("X and Xt do not have the same number of columns. Check dimension compatability")
  }
  
  # Check eta is positive
  if(eta <= 0){
    stop("Eta needs to be strictly positive")
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    stop("lambda needs to be non-negative")
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. 
  # If not NULL, check for compatibility of dimensions with what has been already supplied.
  if(is.null(beta_init)){
    warning("K not explicitly given. Inferred by K = max(Yt)+1")
    beta_init <- matrix(rnorm(ncol(X)*(max(Yt)+1)), nrow = ncol(X), ncol = (max(Yt)+1)) #How do we get K?
  }
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  # A lot of extraneous calculations. Needs better implementation
  expXtB <- exp(crossprod(t(X),beta_init))
  Pk <- expXtB/rowSums(expXtB)
  logPk <- log(Pk)
  K = max(Yt)+1
  
  temp = 0
  for(i in 0:(K-1)){
    temp = temp + sum(logPk[Y==i,(i+1)])
  }

  obj = -1*temp + (lambda/2)*norm(beta_init)^2
  
  objective = rep(obj, numIter+1)
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}