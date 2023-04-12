library(tidyverse)
library(readxl)

BMI_data <- read_excel("for_R.xlsx")
print(BMI_data)
wt <- BMI_data$Weight
IYY <- BMI_data$IYY
IYY_rat <- BMI_data$IYYRatio
IXX <- BMI_data$IXX
IXX_rat <- BMI_data$IXXRatio

BMI <- BMI_data$BMI
BMI_R <- BMI_data$BMI_Ratio
plot(wt,IXX_rat, pch = 15,col = "blue",xlab = "Body Mass (kg)",ylab = "IXX")
#axis(side = 1, lwd = 2)
#axis(side = 2, lwd = 2)
#print(IXX_rat)