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
    .agg(
        {
            "total_price": [("total_price", "sum")],
            "product_quantity": [
                ("product_quantity", "sum"),
                ("n_unique_products", "count"),
            ],
        }
    )
)

data_items_agg.columns = data_items_agg.columns.get_level_values(1)

#%%

# Lets combine items to completed orders with a left join
data_orders_items = data_orders.join(data_items_agg, how="left", on="id_order")

data_orders_items.columns = [
    "id_order",
    "created_date",
    "state",
    "total_paid",
    "total_items_price",
    "total_items_quantity",
    "n_unique_products",
]

# dropping orders where there are less than 2 unique products
data_orders_items.query("n_unique_products >= 2", inplace=True)

# fixing datatypes
data_orders_items[
    "total_items_quantity"
] = data_orders_items.total_items_quantity.astype(int)
data_orders_items["n_unique_products"] = data_orders_items.n_unique_products.astype(int)

#%%

# The dataframe we now have has the same number of rows that the transaction data has
# and if we look at the number of unique products, it seems that the ordering
# of the transactions is identical. This is not unexpected as the transaction dataset was
# created from order and items dataset just like we did here.

data_orders_items.head()
data_orders_items.shape

#%%

# combining the columns to the transaction data saving the data to processed folder
data_trans_enriched = pd.concat([data_trans, data_orders_items], axis=1)
processed_path = os.path.join("data", "processed", "trans_enriched.csv")
data_trans_enriched.to_csv(processed_path, sep=";", index=False)
