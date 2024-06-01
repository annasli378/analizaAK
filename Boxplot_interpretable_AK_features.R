# Imports
library(tidyverse)
library(ggpubr)
library(rstatix)
library(dplyr)

# Paths
save_path = "./";
path_ = './cechy_boxploty.csv'
Y_path = "./IDvsSTAGE.csv"
# Load data
CECHY <- read.csv(path_);
STAGE <- read.csv(Y_path);
Y <- STAGE[,2, drop=FALSE]

total_data <- cbind(Y,CECHY)
total_data <- total_data %>%  reorder_levels(STAGE, order = c(1, 2, 3))

colname1 <- names(total_data)[1]
colname2 <- names(total_data)[2:9]

# Dunn's test for all the features 
data <- lapply(colname2, function(x) {
  rstatix::dunn_test(total_data, reformulate(colname1, x),  
                     p.adjust.method = "bonferroni")
})

# Loop for printing all boxplots
my.variables <- colnames(CECHY)
for(i in 1:length(my.variables)) {
  my_name = my.variables[i]
  krus <- total_data %>% kruskal_test(CECHY[,i] ~ STAGE)
  eta2 <- total_data %>% kruskal_effsize(CECHY[,i] ~ STAGE)
  pwc <- data[[i]]
  
  pwc <- pwc %>% add_xy_position(x = "STAGE") # , outlier.shape = NA
  
  filename=paste(my_name,".png") # , color = "black", fill = "gray"
  tmp_box = ggboxplot(total_data, x = "STAGE", y = my_name, color = "black", fill = "gray" )+
    stat_pvalue_manual(pwc, hide.ns = TRUE) +
    labs(
      title = get_test_label(krus, detailed = TRUE),
      caption = get_pwc_label(pwc)
    )
  ggsave(tmp_box, file=filename, width = 14, height = 12, units = "cm")
  
}
