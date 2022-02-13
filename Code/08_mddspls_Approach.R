# [0] SetWD, load packages, define variables and functions                  ----
# 0-1 Set WD
# currently use default (as it is an Rproj)

# 0-2 Load packages
library(checkmate)
library(ddsPLS)
library(caret)
library(pROC)
library(combinat)

# 0-2b set options
# it is important to return to standard before R 4.0.0 because otherwise
# ddsPLS won't work due to a weird bug
options(stringsAsFactors = TRUE)

# 0-3 Define variables

# 0-4 Define functions
# 0-4-1 Load functions from 'code/01_Create_BWM_Pattern"
source("./Code/01_Create_BWM_Pattern.R")

# 0-4-2 Function to evaluate the mdd-sPLS-Apprach
eval_mddspls_approach <- function(path = './Data/Raw/BLCA.Rda', frac_train = 0.75, split_seed = 1312,
                                  block_seed_train = 1234, block_seed_test = 1312, train_pattern = 2, 
                                  train_pattern_seed = 12, test_pattern = 2) {
  "
   Evaluate the PL-Approach on the data 'path' points to. 
   TODO: add description
   Evaluate the predicitons with common metrics,and return all results in a DF w/ all the
   settings for the evaluation  (e.g. path, seeds, train_pattern, settings for RF, ...).
   In case the approach can no be applied (e.g. no common blocks in train- & test-set), return a 
   DF with the settings of the evaluation, but w/ '---' for the metrics!
   
   Args:
      > path               (str): Path to a dataset - must contain 'Data/Raw'
      > frac_train       (float): Fraction of observations for the train-set - ]0;1[
      > split_seed         (int): Seed for the split of the data to train & test
      > block_seed_train   (int): Seed for the shuffeling of the block-order in train
      > block_seed_test    (int): Seed for the shuffeling of the block-order in test
      > train_pattern      (int): Seed for the induction of the pattern for train
                                  (obs. are assigned to different folds!)
      > train_pattern_seed (int): Pattern to induce into train (1, 2, 3, 4, 5)
      > test_pattern       (int): Pattern to induce into test (1, 2, 3, 4)
   
   Return:
      > A DF with the settings of the experiment (path to the data, train pattern, ...), 
        hte common blocks between train- & test-set, as well as the settings of the RF 
        (ntree, mtry, ...) and the results of the evaluation (AUC; Brier-Score; Accuracy)
  "
  # [0] Check Inputs
  #     --> All arguments are checked in the functions 'get_train_test()' &
  #         'get_predicition()' that werde loaded from 'Code/01_Create_BWM_Pattern.R'
  
  # [1] Load & prepare the data 
  # 1-1 Load the data from 'path', split it to test- & train & induce block-wise 
  #     missingness into both of them according to 'train_pattern' & 'test_pattern'
  train_test_bwm <- get_train_test(path = path,                             # Path to the data
                                   frac_train = frac_train,                 # Fraction of data used for Training (rest for test)
                                   split_seed = split_seed,                 # Seed for the split of the data into test- & train
                                   block_seed_train = block_seed_train,     # Seed to shuffle the block-order in train
                                   block_seed_test = block_seed_test,       # Seed to shuffle the block-order in test
                                   train_pattern = train_pattern,           # Pattern to introduce to train
                                   train_pattern_seed = train_pattern_seed, # Seed for the introduction of the BWM into train
                                   test_pattern = test_pattern)             # Pattern for the test-set
  
  # 1-2 Get the observed blocks of test- & train-set
  # --1 Get all test blocks (whether observed or not) in the correct order
  #     into a list
  test_blocks <- list()
  for (curr_block in train_test_bwm$Test$block_names) {
    
    # --1 Which Index has the current block
    curr_block_idx <- which(train_test_bwm$Test$block_names == curr_block)
    
    # --2 Get the corresponding columns to 'curr_block'
    curr_block_cols <- which(train_test_bwm$Test$block_index == curr_block_idx)
    
    # --3 add it to test_blocks
    test_blocks[[curr_block]] <- train_test_bwm$Test$data[,curr_block_cols]
  }
  
  # --2 Get all train blocks (whether observed or not) in the correct order
  #     into a list
  train_blocks <- list()
  for (curr_test_block in names(test_blocks)) {
    
    # --1 Get the index of the current test-block from the train-set
    curr_train_block_idx <- which(train_test_bwm$Train$block_names == curr_test_block)
    
    # --2 Extract the corresponding columns from train for current train-block 
    curr_train_block_cols <- which(train_test_bwm$Train$block_index == curr_train_block_idx)
    
    # --3 add it to train_blocks
    train_blocks[[curr_test_block]] <- train_test_bwm$Train$data[, curr_train_block_cols]
  }
  # --5 extract the response variable
  ytarget <- train_test_bwm$Train$data$ytarget
  ytarget_test <- train_test_bwm$Test$data$ytarget
  
  # --6 store names
  names_train_blocks <- names(train_blocks)
  
  # 1-3 In case 'train_blocks' is an empty list, return the result-DF, but w/o metrics!
  if (length(names(train_blocks)) <= 0) {
    return(data.frame("path"               = path, 
                      "frac_train"         = frac_train, 
                      "split_seed"         = split_seed, 
                      "block_seed_train"   = block_seed_train,
                      "block_seed_test"    = block_seed_test, 
                      "block_order_train_for_BWM" = paste(train_test_bwm$Train$block_names, collapse = ' - '),
                      "block_order_test_for_BWM"  = paste(train_test_bwm$Test$block_names, collapse = ' - '),
                      "train_pattern"      = train_pattern, 
                      "train_pattern_seed" = train_pattern_seed, 
                      "test_pattern"       = test_pattern, 
                      "common_blocks"      = '---',
                      "AUC"                = '---',
                      "Accuracy"           = '---', 
                      "Sensitivity"        = '---', 
                      "Specificity"        = '---', 
                      "Precision"          = '---', 
                      "Recall"             = '---', 
                      "F1"                 = '---', 
                      "BrierScore"         = '---'))
  }
  
  # 1-7 store block information for later
  train_block_names <- train_test_bwm$Train$block_names
  test_block_names <- train_test_bwm$Test$block_names
  rm(train_test_bwm)
  
  # [2] Train & evaluate mdd-sPLS on the data
  # 2-1 Train a mdd-sPLS model on the 'train' data
  mddspls_cv <- perf_mddsPLS(Xs = train_blocks,
                             Y = as.factor(ytarget),
                             n_lambda = 10,
                             R = 1,
                             NCORES = 1,
                             mode = "logit",
                             plot_result = FALSE,
                             kfolds = 10,
                             weight = TRUE)
  
  # 2-2 Train the model with the best parameters
  best_model <- mddsPLS(Xs = train_blocks,
                        Y = as.factor(ytarget),
                        lambda = mddspls_cv$Optim$optim_para_all$Lambdas[1],
                        R = mddspls_cv$Optim$optim_para_all$R[1],
                        mode = "logit",
                        weight = TRUE)
  
  # 2-3 make predictions
  predictions <- predict(object = best_model,
                         newdata = test_blocks)$probY
  # the predictions have probabilities for both classes, only use the probability
  # for class 1
  predictions <- as.vector(predictions[, 2])
  
  # 2-4 Get the predicted class
  classes_predicted <- factor(as.numeric(predictions >= 0.5), levels = c(0, 1))
  
  # [3] Calculate the metrics based on the true & predicted labels
  # 3-1  Confusion Matrix & all corresponding metrics (Acc, F1, Precision, ....)
  metrics_1 <- caret::confusionMatrix(classes_predicted, 
                                      factor(ytarget_test, 
                                             levels = c(0, 1)),
                                      positive = "1")
  
  # 3-2 Calculate the AUC
  AUC <- pROC::auc(factor(ytarget_test, levels = c(0, 1)), 
                   predictions, quiet = T)
  
  # 3-3 Calculate the Brier-Score
  brier <- mean((predictions - ytarget_test)  ^ 2)
  
  # [4] Return the results as DF
  return(data.frame("path"               = path, 
                    "frac_train"         = frac_train, 
                    "split_seed"         = split_seed, 
                    "block_seed_train"   = block_seed_train,
                    "block_seed_test"    = block_seed_test, 
                    "block_order_train_for_BWM" = paste(train_block_names, collapse = ' - '),
                    "block_order_test_for_BWM"  = paste(test_block_names, collapse = ' - '),
                    "train_pattern"      = train_pattern, 
                    "train_pattern_seed" = train_pattern_seed, 
                    "test_pattern"       = test_pattern, 
                    "common_blocks"      = paste(names_train_blocks, collapse = ' - '),
                    "AUC"                = AUC,
                    "Accuracy"           = metrics_1$overall['Accuracy'], 
                    "Sensitivity"        = metrics_1$byClass['Sensitivity'], 
                    "Specificity"        = metrics_1$byClass['Specificity'], 
                    "Precision"          = metrics_1$byClass['Precision'], 
                    "Recall"             = metrics_1$byClass['Recall'], 
                    "F1"                 = metrics_1$byClass['F1'], 
                    "BrierScore"         = brier))
}

# [1] Run the experiments                                                    ----
# 1-1 Initalize a empty DF to store the results
pl_res <- data.frame()

# 1-2 Define a list with the paths to the availabe DFs
df_paths <- paste0("./Data/Raw/", list.files("./Data/Raw/"))

# 1-3 Create a list of seeds for each single evaluation-setting
set.seed(1234)
count    <- 1
allseeds <- base::sample(1000:10000000, 
                         size = length(df_paths) * length(c(1, 2, 3, 4, 5)) * 
                           length(c(1, 2, 3, 4)) * length(c(1, 2, 3, 4, 5)))

# 1-4 Evaluate a RF on all the possible combinations of block-wise missingness
#     patterns in train- & test-set for all DFs in 'df_paths'. Each is evaluated
#     5-times.
for (curr_path in df_paths) {
  for (curr_train_pattern in c(1, 2, 3, 4, 5)) {
    for (curr_test_pattern in c(1, 2, 3, 4)) {
      for (curr_repetition in c(1, 2, 3, 4, 5)) {
        
        # Print Info to current evaluation!
        cat('-----------------------------------------------\n',
            "Current Path:          >", curr_path, '\n',
            "Current Train Pattern: >", curr_train_pattern, '\n',
            "Current Test Patter:   >", curr_test_pattern, '\n',
            "Current Repetition:    >", curr_repetition, '\n')
        
        
        # Get initial seed for the current combination evaluation-settings
        int_seed <- allseeds[count]
        count    <- count + 1
        
        # Set seed & draw points from uniform distribution
        set.seed(int_seed)    
        seeds <- round(runif(4, 0, 100000))
        
        # Use these 'seeds' to set the four necessary seeds for the evaluation:
        #     1. Seed to split data into test & train
        curr_split_seed <- seeds[1]
        
        #     2. Seed to shuffle the block order in 'train'
        curr_block_seed_train <- seeds[2]
        
        #     3. Seed to shuffle the block order in 'test'
        curr_block_seed_test <- seeds[3]
        
        #     4. Seed for the train-pattern (assignment of obs. in train to folds)
        curr_train_pattern_seed <- seeds[4]
        
        # Run the evaluation with current settings
        curr_res <- tryCatch(eval_mddspls_approach(path               = curr_path, 
                                                   frac_train         = 0.75, 
                                                   split_seed         = curr_split_seed,
                                                   block_seed_train   = curr_block_seed_train, 
                                                   block_seed_test    = curr_block_seed_test,
                                                   train_pattern      = curr_train_pattern,
                                                   train_pattern_seed = curr_train_pattern_seed, 
                                                   test_pattern       = curr_test_pattern),
                             error = function(c) {
                               data.frame("path"               = curr_path, 
                                          "frac_train"         = 0.75, 
                                          "split_seed"         = curr_split_seed, 
                                          "block_seed_train"   = curr_block_seed_test,
                                          "block_seed_test"    = curr_block_seed_test, 
                                          "block_order_train_for_BWM" = '---',
                                          "block_order_test_for_BWM"  = '---',
                                          "train_pattern"      = curr_train_pattern, 
                                          "train_pattern_seed" = curr_train_pattern_seed, 
                                          "test_pattern"       = curr_test_pattern,
                                          "common_blocks"      = "---",
                                          "AUC"                = '---',
                                          "Accuracy"           = '---', 
                                          "Sensitivity"        = '---', 
                                          "Specificity"        = '---', 
                                          "Precision"          = '---', 
                                          "Recall"             = '---', 
                                          "F1"                 = '---', 
                                          "BrierScore"         = '---')
                             }
        ) 
        # Add the, 'int_seed', 'curr_repetition' & 'SingleBlock' to 'curr_res'
        curr_res$int_seed   <- int_seed
        curr_res$repetition <- curr_repetition
        curr_res$approach   <- 'Blockwise'
        
        # Add the results of the setting to 'BW_res' & save it
        pl_res <- rbind(pl_res, curr_res)
        write.csv(pl_res, './Docs/Evaluation_Results/MDDSPLS_Approach/MDDSPLS_Eval.csv')
      }
    }
  }
}