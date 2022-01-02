library(MASS)
library(fastICA)
library(rgr)
library(openxlsx)
xx <-read.csv("")
xx.mat <- as.matrix(xx)
View(xx.mat)
xx.mat[] <- xx.mat[] * 10000
temp <- alr(xx.mat, )
temp
View(temp)
View(temp)
write.xlsx(temp, "")
