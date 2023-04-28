library(rTensor)
library(abind)
library(stringr)
library(dplyr)

art = read.csv("data/art.csv", sep=";", row.names = 1)
port = read.csv("data/port.csv", sep=";", row.names = 1)
vein = read.csv("data/vein.csv", sep=";", row.names = 1)
tard = read.csv("data/tard.csv", sep=";", row.names = 1)
labels = read.csv("data/label.csv", sep=";", row.names = 1)

# preprocess the data

preprocess <- function(data) {
  # reorder samples so that we have first all the CCK, then all CHC, then all Mixtes
  reorder.samples = c(rownames(data)[142], 
                      rownames(data)[1:106],
                      rownames(data)[143:145], 
                      rownames(data)[107:141], 
                      rownames(data)[146:147])
  data = data[reorder.samples,]
  
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
  data = data.matrix(data)
  dim(data) = c(dim(data)[1], dim(data)[2], 1)
  return(data)
}

art = preprocess(art)
port = preprocess(port)
vein = preprocess(vein)
tard = preprocess(tard)

X = abind(art, port, vein, tard, along=3)

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


#sGCCA
install.packages("rgcca")
install.packages("FactoMineR")

library(RGCCA)
library(FactoMineR)

X1 <- X[,,1]
X2 <- X[,,2]
X3 <- X[,,3]
X4 <- X[,,4]

######################################
# Glmnet

install.packages("glmnet")
library(glmnet)

combined_data <- cbind(X1, X2, X3, X4)

labels <- labels$x # utilisez les étiquettes fournies dans le fichier 'label.csv'


data_with_labels <- cbind(labels, combined_data)

# Divisez les données en ensembles d'apprentissage et de test
set.seed(123) # Pour la reproductibilité
train_idx <- sample(1:nrow(data_with_labels), 0.7 * nrow(data_with_labels))

train_data <- data_with_labels[train_idx, ]
test_data <- data_with_labels[-train_idx, ]

# Séparez les caractéristiques (features) et les étiquettes (labels) pour les ensembles d'apprentissage et de test
train_x <- as.matrix(train_data[, -1])
train_y <- as.factor(train_data[, 1])

test_x <- as.matrix(test_data[, -1])
test_y <- as.factor(test_data[, 1])

cat("train_x dimensions:", dim(train_x), "\n")
cat("train_y dimensions:", length(train_y), "\n")



set.seed(123)
cvfit <- cv.glmnet(train_x, train_y, family = "multinomial", type.measure = "class", nfolds = 5)

predicted_probs <- predict(cvfit, test_x, type = "response", s = "lambda.min")
predicted_labels <- colnames(predicted_probs)[apply(predicted_probs, 1, which.max)]

confusion_matrix <- table(predicted_labels, test_y)
cat("Confusion matrix:\n")
print(confusion_matrix)
cat("\nAccuracy:", mean(predicted_labels == test_y), "\n")
### 58%

######################################
# Random Forest

install.packages("randomForest")
library(randomForest)

# Grille d'hyperparamètres à tester
ntree_grid <- c(100, 200, 300, 500)  # Nombre d'arbres
mtry_grid <- c(2, 4, 6, 8, 10, 20)  # Nombre de variables à essayer à chaque fractionnement

set.seed(123)  # Pour la reproductibilité

best_mtry_list <- list()
min_error_list <- list()

for (ntree_value in ntree_grid) {
  best_mtry <- tuneRF(train_x, as.factor(train_y), ntreeTry = ntree_value, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = TRUE, doBest = FALSE, mtryStart = mtry_grid[1])
  best_mtry_list[[as.character(ntree_value)]] <- best_mtry
  min_error_list[[as.character(ntree_value)]] <- min(best_mtry[, 2])
}

min_error <- min(unlist(min_error_list))
best_ntree_value <- as.numeric(names(min_error_list)[which.min(unlist(min_error_list))])
best_mtry_value <- best_mtry_list[[as.character(best_ntree_value)]][which.min(best_mtry_list[[as.character(best_ntree_value)]][, 2]), 1]

rf_model_optimized <- randomForest(train_x, train_y, ntree = best_ntree_value, mtry = best_mtry_value, importance = TRUE)

predicted_labels_rf_optimized <- predict(rf_model_optimized, test_x)
cat("\nAccuracy:", mean(predicted_labels_rf_optimized == test_y), "\n")
#### 51%

######################################
# SVM

install.packages("e1071")
library(e1071)

cost_grid <- c(0.1, 1, 10, 100)  # Paramètre de coût (C)
gamma_grid <- c(0.1, 1, 10)  # Paramètre gamma

set.seed(123)  # Pour la reproductibilité

tune_result <- tune(svm, train_x, as.factor(train_y), kernel = "radial", ranges = list(cost = cost_grid, gamma = gamma_grid), tunecontrol = tune.control(cross = 5))
best_cost <- tune_result$best.parameters$cost
best_gamma <- tune_result$best.parameters$gamma

svm_model <- svm(train_x, as.factor(train_y), kernel = "radial", cost = best_cost, gamma = best_gamma)

predicted_labels_svm <- predict(svm_model, test_x)
cat("\nAccuracy:", mean(predicted_labels_svm == test_y), "\n")
#### 58%
