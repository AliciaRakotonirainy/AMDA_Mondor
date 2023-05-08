library(rTensor)
library(abind)
library(stringr)
library(dplyr)
library(MASS)
library(caret)
art = read.csv("data/art.csv", sep=";", row.names = 1)
port = read.csv("data/port.csv", sep=";", row.names = 1)
vein = read.csv("data/vein.csv", sep=";", row.names = 1)
tard = read.csv("data/tard.csv", sep=";", row.names = 1)
labels = read.csv("data/label.csv", sep=";", row.names = 1)

# preprocess the data

reorder <- function(data) {
  # reorder samples so that we have first all the CCK, then all CHC, then all Mixtes
  reorder.samples = c(rownames(data)[142], 
                      rownames(data)[1:106],
                      rownames(data)[143:145], 
                      rownames(data)[107:141], 
                      rownames(data)[146:147])
  data = data[reorder.samples,]
  return(data)
}

preprocess <- function(data) {
  
  data = reorder(data)
  
  # preprocessing : make all values as floats
  data[data == "#N/A"] = NA
  data <- data %>% 
    mutate_all(funs(str_replace(., ",", ".")))
  data <- as.data.frame(sapply(data, as.numeric))
  
  # replace NA values by column mean
  for(i in 1:ncol(data)){
    data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
  }
  
  # scale the data
  data = scale(data)
  
  # extend the dimension
  data = t(data.matrix(data))
  dim(data) = c(1, dim(data)[1], dim(data)[2])
  return(data)
}

art = preprocess(art)
port = preprocess(port)
vein = preprocess(vein)
tard = preprocess(tard)

X = abind(art, port, vein, tard, along=1)
X = as.tensor(X)

# RGCCA

library(RGCCA)

labels = reorder(labels)
labels_onehot = labels
labels_onehot$CCK = 0
labels_onehot$CHC = 0
labels_onehot$Mixtes = 0
labels_onehot$CCK[which(labels_onehot$x == "CCK")] = 1
labels_onehot$CHC[which(labels_onehot$x == "CHC")] = 1
labels_onehot$Mixtes[which(labels_onehot$x == "Mixtes")] = 1
labels_onehot = subset(labels_onehot, select = -c(x) )
# reorder samples so that we have first all the CCK, then all CHC, then all Mixtes
reorder.samples = c(rownames(labels_onehot)[142], 
                    rownames(labels_onehot)[1:106],
                    rownames(labels_onehot)[143:145], 
                    rownames(labels_onehot)[107:141], 
                    rownames(labels_onehot)[146:147])
labels_onehot = labels_onehot[reorder.samples,]

X_data = list()
X_data$art = art
X_data$port = port
X_data$vein = vein
X_data$tard = tard
X_data$y = labels_onehot

f1_list = c()

for (t in c(0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
  C = matrix(c(0, 0, 0, 0, 1, 
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 0), 5, 5)
  
  tau = c(t, t, t, t, 0)
  
  fit.rgcca = rgcca(X_data, connection = C, tau = tau, init = "random", 
                    ncomp = 2, tol = 1e-16, 
                    verbose = TRUE)
  
  # train a classifier on the 4 first components of the 4 blocks
  latent_data = data.frame(art1 = fit.rgcca$Y$art[,1],
                           port1 = fit.rgcca$Y$port[,1],
                           vein1 = fit.rgcca$Y$vein[,1],
                           tard1 = fit.rgcca$Y$tard[,1],
                           y = labels)
  
  
  X.lda <- lda(y ~ ., data=latent_data)
  X.lda.values <- predict(X.lda)
  # prediction based on LDA
  # pct of correct classification (but imbalanced with CHC...)
  
  f1_list = c(f1_list, compute_f1(X.lda.values$class, factor(labels)))
}
print(f1_list)

# plots of LDA

ldahist(data = X.lda.values$x[,1], g=y)
plot(X.lda.values$x[,1],X.lda.values$x[,2], col=y)

plot(fit.rgcca, type= "sample", comp = 1, block = c(1,4),
     title = "Factorial plan of RGCCA", resp = labels$x)
boot_out = rgcca_bootstrap(fit.rgcca, 200)
print(boot_out)
plot(boot_out, display_order = FALSE)

# sGCCA

C = matrix(c(0, 0, 0, 0, 1, 
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             0, 0, 0, 0, 1,
             1, 1, 1, 1, 0), 5, 5)

f1_list = c()
for (t in c(0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
  tau = c(t, t, t, t, 0)
  
  for (s in c(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9)){
    fit.sgcca = rgcca(X_data, connection = C, method = "sgcca", sparsity = c(s, s, s, s, 1), 
                      ncomp = c(2, 2, 2, 2, 1), scheme = "centroid", verbose = TRUE)
    
    # train a classifier on the 4 first components of the 4 blocks
    latent_data = data.frame(art1 = fit.sgcca$Y$art[,1],
                             port1 = fit.sgcca$Y$port[,1],
                             vein1 = fit.sgcca$Y$vein[,1],
                             tard1 = fit.sgcca$Y$tard[,1],
                             y = labels)
    
    
    X.lda <- lda(y ~ ., data=latent_data)
    X.lda.values <- predict(X.lda)
    # prediction based on LDA
    # pct of correct classification (but imbalanced with CHC...)
    macrof1 = compute_f1(X.lda.values$class, factor(labels))
    f1_list = c(f1_list, macrof1)
  }
}
print(f1_list)

ldahist(data = X.lda.values$x[,1], g=y)
plot(X.lda.values$x[,1],X.lda.values$x[,2], col=y)

compute_f1 <- function(y_pred, y_true){
  cm <- as.matrix(confusionMatrix(y_pred, y_true))
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  diag = diag(cm)  # number of correctly classified instances per class 
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  f1[is.nan(f1)] <- 0
  macrof1 = mean(f1)
  return(macrof1)
}


print(macrof1)


# TUCKER
lim_tucker = tucker_lim(X, c(4,20))
est <- ttl(lim_tucker$Z,lim_tucker$U,ms=1:2)
fnorm(X - est)

nfeatures = lim_tucker$Z@modes[1] *  lim_tucker$Z@modes[2]
tucker_df = data.frame(matrix(ncol = nfeatures, nrow = 147))
colnames(tucker_df) = 1:nfeatures

for (n in 147) {
  slice = lim_tucker$Z@data[,,n]
  slice = as.vector(slice)
  
}


library(tsne)
library(plotly)
data("iris")

features <- subset(iris, select = -c(Species)) 
set.seed(0)
tsne <- tsne(features, initial_dims = 2)
tsne <- data.frame(tsne)
pdb <- cbind(tsne,iris$Species)
options(warn = -1)
fig <-  plot_ly(data = pdb ,x =  ~X1, y = ~X2, type = 'scatter', mode = 'markers', split = ~iris$Species)

fig <- fig %>%
  layout(
    plot_bgcolor = "#e5ecf6"
  )

fig

# PARAFAC

cp_decomp <- cp(as.tensor(X), num_components = 1)
str(cp_decomp$U)

# unfortunately we do not see differences between the 3 classes 
# (see similar analysis : https://www.alexejgossmann.com/tensor_decomposition_CP/)
plot(1:147, cp_decomp$U[[1]], type="S", main="First CP Component - Individuals")
abline(v=24, col="blue")
abline(v=111, col="blue")

plot(1:107, cp_decomp$U[[2]], type="S", main="Second CP Component - Radiomic Features")
plot(1:4, cp_decomp$U[[3]], type="S", main="Third CP Component - Time")

# Tucker

library(psych)

factors = cp(as.tensor(X), num_components = 5)$U
scores <- list()

for (i in 1:length(dim(X))) {
  scores[[i]] <-tcrossprod(factors[[i]], X) %o% kronecker(factors[-i], c(1,1))
  
}

# i = 1
# t(factors[[i]]) = 5 x 147

# 107 features x 4 temps pour chaque patient

sc <- function(x) {
  return(t(factors$U[[1]]) %*% x)
}

