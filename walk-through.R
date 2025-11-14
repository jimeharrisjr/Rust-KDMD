library(pracma)

kdmd<-function(x){
  if (!is.matrix(x)) stop("X must be matrix")
  structure(as.matrix(x), class = c("kdmd", 'matrix', 'array'))
}
p=1; comp=NA
m<-matrix(ncol=100,nrow=3)
x<-1:100
m[1,]<-sin(x*pi/10)
m[2,]<-sin(1+x*pi/10)
m[3,]<-sin(2+x*pi/10)
writeLines("Begin run",'outputfile.txt')
data<-m
dims <- dim(data)
if (!(dims[1] > 1 &
      dims[2] > 1))
  stop ('matrix must have two dimensions (2x2 or greater)')
if (p <= 0 |
    p > 1)
  stop (paste('p=', p, 'p value must be within the range (0,1]'))
x <- data[, -ncol(data)]
y <- data[, -1]
wsvd <- base::svd(x)
if (!is.na(comp)) {
  r <- as.integer(comp)
  if (r <= 1) {
    warning('component values below 2 are not supported. Defaulting to 2')
    r <- 2
  }
  if (r > length(wsvd$d)) {
    warning('Specified number of components is greater than possible. Ignoring extra')
    r <- length(wsvd$d)
  }
} else {
  if (p == 1) {
    r <- length(wsvd$d)
  } else {
    sv <- (wsvd$d ^ 2) / sum(wsvd$d ^ 2)
    r <- max(which(cumsum(sv) >= p)[1], 2)
  }
}
file_connection <- file("outputfile.txt", open = "a")
writeLines("U Matrix output:", file_connection)
u <- wsvd$u
writeLines(capture.output(print(u)), file_connection)
writeLines("V Matrix output:", file_connection)
v <- wsvd$v
writeLines(capture.output(print(v)), file_connection)
writeLines("d output:", file_connection)
d <- wsvd$d
writeLines(capture.output(print(d)), file_connection)
# If there are components close to zero, override r value and remove them
maxcomp<-max(which(d>3e-15))
if (maxcomp<r){warning("rank of SVD lower than rank value selected, overriding")}
r=min(maxcomp,r)
Atil <- t(u[,1:r]) %*% y
Atil <- Atil %*% v[,1:r]
Atil <- Atil %*% diag(1 / d[1:r])
writeLines("Atil output:", file_connection)
writeLines(capture.output(print(Atil)), file_connection)

eig <- eigen(Atil)
writeLines("Phi output:", file_connection)
Phi <- eig$values
writeLines(capture.output(print(Phi)), file_connection)
writeLines("Q output:", file_connection)
Q <- eig$vectors
writeLines(capture.output(print(Q)), file_connection)
writeLines("Psi output:", file_connection)
Psi <- y %*% v[, 1:r] %*% diag(1 / d[1:r]) %*% (Q)
writeLines(capture.output(print(Psi)), file_connection)
writeLines("x output:", file_connection)
x <- Psi %*% diag(Phi) %*% pracma::pinv(Psi)
writeLines(capture.output(print(x)), file_connection)
writeLines("A matrix output:", file_connection)
A <- kdmd(x)
writeLines(capture.output(print(A)), file_connection)


## Predict functions
if (!'kdmd' %in% class(A)) {
  stop('A must be a kdmd object generated with getAMatrix')
}
if (!is.matrix(data) &
    !is.data.frame(data))
  stop('data must be a numeric matrix or a data frame')
data <- as.matrix(data)
len_predict <- 100 # predict 100 more points
t <- dim(data)[2]
N <- dim(data)[1]
ynew <- data
for (st in 0:(len_predict - 1)) {
  b <- Re(A %*% matrix(ynew[, (t + st)], ncol = 1))
  ynew <- cbind(ynew, b)
}
#writeLines("Predicted y matrix output:", file_connection)
#writeLines(ynew, file_connection)
close(file_connection)
# Plot the output
plot(ynew[1,], type='l')
lines(ynew[2,], type='l')
lines(ynew[3,], type='l')

