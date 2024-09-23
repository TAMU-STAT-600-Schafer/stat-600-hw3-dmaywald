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




