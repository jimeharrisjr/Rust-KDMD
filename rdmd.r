
kdmd<-function(x){
  if (!is.matrix(x)) stop("X must be matrix")
  structure(as.matrix(x), class = c("kdmd", 'matrix', 'array'))
}

getAMatrix<-function(data,p=1, comp=NA){
  if (!is.matrix(data) & !is.data.frame(data)) stop('data must be a numeric matrix or a data frame')
  data<-as.matrix(data)
  dims<-dim(data)
  if (!(dims[1]>1 & dims[2]>1)) stop ('matrix must have two dimensions (2x2 or greater)')
  if (p<=0 | p>1) stop (paste('p=',p,'p value must be within the range (0,1]'))
  x<-data[,-ncol(data)]
  y<-data[,-1]
  wsvd<-base::svd(x)
  if (!is.na(comp)){
    r<-as.integer(comp)
    if (r<=1){warning('component values below 2 are not supported. Defaulting to 2'); r<-2}
    if (r>length(wsvd$d)){
      warning('Specified number of components is greater than possible. Ignoring extra')
      r<-length(wsvd$d)
    }
  } else {
    if (p==1){
      r<-length(wsvd$d)
    } else {
      sv<-(wsvd$d^2)/sum(wsvd$d^2)
      r<-max(which(cumsum(sv)>=p)[1],2)
    }
  }
  u<-wsvd$u
  v<-wsvd$v
  d<-wsvd$d
  Atil<-crossprod(u[,1:r],y)
  Atil<-crossprod(t(Atil),v[,1:r])
  Atil<-crossprod(t(Atil),diag(1/d[1:r]))
  eig<-eigen(Atil)
  Phi<-eig$values
  Q<-eig$vectors
  Psi<- y %*% v[,1:r] %*% diag(1/d[1:r]) %*% (Q)
  x<-Psi %*% diag(Phi) %*% pracma::pinv(Psi)
  A<- kdmd(x)
  return(A)
}

predict.kdmd<-function(A,data,l=1){
  if (!'kdmd' %in% class(A)){ stop('A must be a kdmd object generated with getAMatrix')}
  if (!is.matrix(data) & !is.data.frame(data)) stop('data must be a numeric matrix or a data frame')
  data<-as.matrix(data)
  len_predict<-l
  t<-dim(data)[2]
  N<-dim(data)[1]
  ynew<-data
  for (st in 0:(len_predict-1)){
    b<-Re(crossprod(t(A),matrix(ynew[,(t+st)], ncol=1)))
    ynew<-cbind(ynew,b)
  }
  return(ynew)
}
