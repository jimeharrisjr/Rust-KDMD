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

cat("SVD d values:\n")
print(sprintf("%.15e", wsvd$d))

threshold <- max(dim(x)) * .Machine$double.eps * wsvd$d[1]
cat("Threshold:", threshold, "\n")
effective_r <- sum(wsvd$d > threshold)
cat("Effective rank:", effective_r, "\n")
