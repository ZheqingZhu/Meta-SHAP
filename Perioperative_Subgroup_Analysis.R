# ==============================================================================
# Multilevel Meta-Regression Analysis for Perioperative Immunotherapy Subgroups
# Author: Zheqing Zhu (Shanxi University)
# Description: This script performs multilevel meta-regression to estimate 
#              adjusted hazard ratios across clinical subgroups.
# ==============================================================================

# --- 1. Dependencies ---
required_packages <- c("dplyr", "readr", "purrr", "metafor", "ggplot2", "forcats", "stringr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
lapply(required_packages, library, character.only = TRUE)

# --- 2. Environment Setup ---
# USERS: Set your working directory or ensure data is in the script folder
# setwd("path/to/your/project") 

# Define input/output paths (Using relative paths for portability)
input_file <- "data_pcr_mpr.csv" 
output_dir <- "./output"

if(!dir.exists(output_dir)) dir.create(output_dir)

# --- 3. Data Preprocessing ---
df_raw <- read_csv(input_file, show_col_types = FALSE)

df_clean <- df_raw %>%
  filter(Category != "Overall", !is.na(HR), HR > 0) %>%
  # Filter out rows with biologically implausible or extreme variance
  filter(Upper_CI < 100) %>% 
  mutate(
    log_hr = log(HR),
    # Calculate sampling variance (SE based on 95% CI)
    se = (log(Upper_CI) - log(Lower_CI)) / (2 * 1.96),
    vi = se^2
  ) %>%
  filter(vi > 0)

# --- 4. Multilevel Meta-Regression Model ---
# Model structure: Fixed effects for Subgroups, 
# Random effects nested as Trial/Category to account for structural dependency.
subgroups <- unique(df_clean$Subgroup)
results <- map_dfr(subgroups, function(s) {
  
  sub_data <- df_clean %>% filter(Subgroup == s)
  
  # Ensure sufficient studies exist for the subgroup
  if(nrow(sub_data) < 2) return(NULL)
  
  tryCatch({
    model <- rma.mv(
      yi = log_hr, 
      V = vi, 
      random = ~ 1 | Trial/Category,
      data = sub_data,
      method = "REML"
    )
    
    data.frame(
      Subgroup = s,
      Adjusted_HR = exp(model$beta),
      Adj_CI_Lower = exp(model$ci.lb),
      Adj_CI_Upper = exp(model$ci.ub),
      P_Value = model$pval,
      N_Studies = model$k
    )
  }, error = function(e) return(NULL))
})

# --- 5. Export and Visualization ---
write_csv(results, file.path(output_dir, "Subgroup_Meta_Regression_Results.csv"))

# Forest Plot (Figure 1 Configuration)
forest_p <- ggplot(results, aes(x = Adjusted_HR, y = fct_reorder(Subgroup, desc(Adjusted_HR)))) +
  geom_point(shape = 15, size = 3, color = "#0073C2FF") + 
  geom_errorbarh(aes(xmin = Adj_CI_Lower, xmax = Adj_CI_Upper), height = 0.2, color = "#2E9FDF") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "#D55E00") + 
  scale_x_log10(breaks = c(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)) + 
  theme_classic() +
  labs(x = "Adjusted Hazard Ratio (95% CI)", y = "Clinical Subgroups") +
  theme(axis.text = element_text(color = "black"))

ggsave(file.path(output_dir, "Figure1_Forest_Plot.pdf"), forest_p, width = 8, height = 6)