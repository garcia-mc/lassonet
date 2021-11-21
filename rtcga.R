# Install the main RTCGA package


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("RTCGA")

# Install the clinical  data packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("RTCGA.clinical")

# and mRNA gene expression
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("RTCGA.mRNA")

library(RTCGA)
infoTCGA()

?infoTCGA

library(RTCGA.clinical)
?clinical
browseVignettes("RTCGA")

library(RTCGA.mRNA)
?mRNA

library(RTCGA.mRNA)
?mRNA

dim(BRCA.mRNA)
BRCA.mRNA[1:5, 1:5]


## Extracting Survival Data
library(RTCGA.clinical)
survivalTCGA(BRCA.clinical, OV.clinical, extract.cols = "admin.disease_code") -> brov

# first munge data, then extract survival info
library(dplyr)
BRCA.clinical %>%
  filter(patient.drugs.drug.therapy_types.therapy_type %in%
           c("chemotherapy", "hormone therapy")) %>%
  rename(therapy = patient.drugs.drug.therapy_types.therapy_type) %>%
  survivalTCGA(extract.cols = c("therapy"))  -> BRCA.survInfo.chemo

# first extract survival info, then munge data                  
survivalTCGA(BRCA.clinical, 
             extract.cols = c("patient.drugs.drug.therapy_types.therapy_type"))  %>%
  filter(patient.drugs.drug.therapy_types.therapy_type %in%
           c("chemotherapy", "hormone therapy")) %>%
  rename(therapy = patient.drugs.drug.therapy_types.therapy_type) -> BRCA.survInfo.chemo


proba=survivalTCGA(BRCA.clinical, 
             extract.cols = c("patient.drugs.drug.therapy_types.therapy_type")) 


datos=BRCA.clinical







