#%%

import os

import numpy as np
import pandas as pd
from tabula import read_pdf

#%%

# loading data
pdf_path = os.path.join("data", "raw", "products_with_category.pdf")
data_categories = read_pdf(pdf_path, pages="all", Stream=True, guess=False)

#%%

# Removing whitespace around values just in cae
data_categories.columns = ["labels", "level1"]
data_categories["labels"] = data_categories.labels.str.strip()
data_categories["level1"] = data_categories.level1.str.strip()

#%%

# Fixing spelling mistake in smartwatch
data_categories.loc[(data_categories.level1 == "smartwhatch"), "level1"] = "smartwatch"
data_categories["level1"].value_counts()

#%%

# saving data
processed_path = os.path.join("data", "processed", "product_categories.csv")
data_categories.to_csv(processed_path, index=False)
