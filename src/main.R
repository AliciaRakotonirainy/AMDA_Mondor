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
