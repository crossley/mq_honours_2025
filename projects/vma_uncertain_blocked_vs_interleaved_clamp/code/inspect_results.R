library(data.table)
library(ggplot2)
library(ez)

rm(list=ls())

d <- fread("../data_summary/d_for_anova.csv")

ezANOVA(d, dv=.(emv), wid=.(subject), between=.(condition),
        within =.(phase_2, su_prev), type=3, detailed =
            TRUE)

