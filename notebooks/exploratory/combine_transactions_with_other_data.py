#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import datetime
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
sns.set(style="whitegrid", color_codes=True)
InteractiveShell.ast_node_interactivity = "all"

#%%

# Loading the data
raw_path = os.path.join("data", "raw")
# for trans we do not want the header
data_trans = pd.read_csv(
    os.path.join(raw_path, "trans.csv"), skiprows=1, header=None, sep="\n"
)
data_trans = data_trans[0].str.split(",", expand=True)
data_orders = pd.read_csv(
    os.path.join(raw_path, "orders_translated.csv"), sep=";", decimal=","
)
data_items = pd.read_csv(os.path.join(raw_path, "lineitems.csv"), sep=";", decimal=",")

# Adding total price to items that takes into account the number of items
data_items["total_price"] = data_items["unit_price"] * data_items["product_quantity"]

# correcting the datetime columns
data_orders.created_date = pd.to_datetime(data_orders.created_date)
data_items.date = pd.to_datetime(data_items.date)


# Keeping only the orders with state "Completed"
data_orders.query("state == 'Completed'", inplace=True)

#%%
# Aggregating the item data to orders and keeping only important columns
data_items_agg = (
    data_items[["id_order", "total_price", "product_quantity"]]
    .groupby(["id_order"])
    .agg({"total_price": ['sum'], "product_quantity": ['sum', 'count']})
)

#%%

# Lets combine the orders and items with an inner join
data_orders_items = data_orders.join(data_items_agg, how="left", on="id_order")

data_order_items.columns = [
    "id_order",
    "created_date",
    "state",
    "total_paid",
    "total_items_price",
    "total_items_quantity",
]


#%%


#%%
