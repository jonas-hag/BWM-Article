library(dplyr)

esca_1 <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_ESCA_old.csv")
esca_2 <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_ESCA_2_4_3.csv")
esca_3 <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_ESCA_4_4_3.csv")

blca <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_BLCA.csv")

fill_1 <- blca %>% 
  filter(train_pattern == 2, test_pattern == 4, repetition == 2) %>% 
  mutate(across(-c(X, path, train_pattern, test_pattern, repetition, approach), ~NA),
         path = "/BWM-Article/Data/Raw/ESCA.Rda")

fill_2 <- blca %>% 
  filter(train_pattern == 4, test_pattern == 4, repetition == 2) %>% 
  mutate(across(-c(X, path, train_pattern, test_pattern, repetition, approach), ~NA),
         path = "/BWM-Article/Data/Raw/ESCA.Rda")

esca_total <- rbind(esca_1, fill_1, esca_2, fill_2, esca_3)
write.csv(esca_total, "Docs/Evaluation_Results/PL_Approach/PL_Eval_ESCA.csv")

paad_1 <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_PAAD_old.csv")
paad_2 <- read.csv("Docs/Evaluation_Results/PL_Approach/PL_Eval_PAAD_4_4_5.csv")

fill_3 <- blca %>% 
  filter(train_pattern == 4, test_pattern == 4, repetition == 4) %>% 
  mutate(across(-c(X, path, train_pattern, test_pattern, repetition, approach), ~NA),
         path = "/BWM-Article/Data/Raw/PAAD.Rda")

paad_total <- rbind(paad_1, fill_3, paad_2)
write.csv(paad_total, "Docs/Evaluation_Results/PL_Approach/PL_Eval_PAAD.csv")
