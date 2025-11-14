library(pracma)

m<-matrix(ncol=100,nrow=3)
x<-1:100
m[1,]<-sin(x*pi/10)
m[2,]<-sin(1+x*pi/10)
m[3,]<-sin(2+x*pi/10)
data<-m

x <- data[, -ncol(data)]
y <- data[, -1]
wsvd <- base::svd(x)

cat("SVD singular values:\n")
print(wsvd$d)

r <- length(wsvd$d)
cat("r (components):", r, "\n")

u <- wsvd$u
v <- wsvd$v
d <- wsvd$d

Atil <- t(u[,1:r]) %*% y
Atil <- Atil %*% v[,1:r]
Atil <- Atil %*% diag(1 / d[1:r])

cat("Atil matrix:\n")
print(Atil)

eig <- eigen(Atil)
cat("Eigenvalues:\n")
print(eig$values)

Phi <- eig$values
Q <- eig$vectors
Psi <- y %*% v[, 1:r] %*% diag(1 / d[1:r]) %*% (Q)

cat("Final Koopman matrix:\n")
x <- Psi %*% diag(Phi) %*% pracma::pinv(Psi)
print(x)
