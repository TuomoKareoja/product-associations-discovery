#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import copy
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
sns.set(style="whitegrid", color_codes=True)
InteractiveShell.ast_node_interactivity = "all"

#%%

raw_path = os.path.join("data", "raw")
clean_path = os.path.join("data", "clean")

# Loading Electonindex data
data_orders = pd.read_csv(
    os.path.join(raw_path, "orders_translated.csv"), sep=";", decimal=","
)
data_items = pd.read_csv(os.path.join(raw_path, "lineitems.csv"), sep=";", decimal=",")
data_categories = pd.read_csv(os.path.join(clean_path, "product_categories.csv"))

# loading Blackwell data
data_blackwell = pd.read_csv(
    os.path.join(raw_path, "existingproductattributes2017.csv")
)


#%%

# Cleaning electronindex data

# Keeping only the orders with state "Completed"
data_orders.query("state == 'Completed'", inplace=True)

# removing whitespace from sku for a clean join with categories
data_items["sku"] = data_items.sku.str.strip()

# Keeping only interesting columns
data_orders.drop(columns=["state", "created_date", "total_paid"], inplace=True)
data_items.drop(columns=["id", "product_id", "date"], inplace=True)

# recoding the categories in Electronindex data to match those of Blackwell
new_categories = {
    "accessories": "Accessories",
    "smartphone": "Smartphone",
    "tablet": "Tablet",
    "display": "Display",
    "laptop": "Laptop",
    "other": "Other",
    "extended warranty": "ExtendedWarranty",
    "pc": "PC",
    "smartwatch": "Smartwatch",
    "service": "Service",
    "camera": "Camera",
    "software": "Software",
    "printer": "Printer",
}

data_categories.columns = ["sku", "category"]
data_categories.replace(dict(category=new_categories), inplace=True)

# Lets combine items to completed orders with an inner join to keep only items
# from completed orders
data_orders_items = data_orders.join(
    data_items.set_index("id_order"), how="inner", on="id_order"
)

# Adding the product categories
data_electronidex = data_orders_items.join(
    data_categories.set_index("sku"), how="left", on="sku"
)

# replacing missing categories with "unknown"
data_electronidex.category.fillna("Unknown", inplace=True)

# Dropping the Extended warranties as the information from these is
# in general not that interesting
data_electronidex.query("category != 'ExtendedWarranty'", inplace=True)

# dropping the now unnecesary id_order column
data_electronidex.drop(columns=["id_order"], inplace=True)

#%%

# No missing values
data_electronidex.isnull().sum()


# Checking data quality of product quantity and unit price

# There are no suprising values in product quantity
data_electronidex.product_quantity.min()
data_electronidex.product_quantity.max()
data_electronidex.product_quantity.mean()
data_electronidex.product_quantity.median()


# There are no suprising values in product quantity
data_electronidex.unit_price.min()
data_electronidex.unit_price.max()
data_electronidex.unit_price.mean()
data_electronidex.unit_price.median()
sns.boxplot(data_electronidex.unit_price)

# the maximum price seems weird
data_electronidex[data_electronidex.unit_price == data_electronidex.unit_price.max()]

# But there is only one observation with this product and the category is unknown. We let this stand
data_electronidex[
    data_electronidex.sku
    == max(
        data_electronidex[
            data_electronidex.unit_price == data_electronidex.unit_price.max()
        ]["sku"]
    )
]

#%%

# calculating total price of items taking into account the amount of items
data_electronidex["price"] = (
    data_electronidex["product_quantity"] * data_electronidex["unit_price"]
)

# dropping the now unnecessary sku and unit price columns
data_electronidex.drop(columns=["sku", "unit_price"], inplace=True)


#%%

# Cleaning Blackwell data

# keeping only interesting columns
data_blackwell = data_blackwell[["ProductType", "Price", "Volume", "ProfitMargin"]]

# Adding combined price and profit from all purchases
data_blackwell["Price_total"] = data_blackwell["Price"] * data_blackwell["Volume"]
data_blackwell["Profit_total"] = (
    data_blackwell["Price"] * data_blackwell["Volume"] * data_blackwell["ProfitMargin"]
)
data_blackwell["Profit_per_unit"] = (
    data_blackwell["Price"] * data_blackwell["ProfitMargin"]
)

# Dropping the Extended warranties as the information from these # seems false and is also in general not that interesting
data_blackwell.query("ProductType != 'ExtendedWarranty'", inplace=True)

# Dropping original price and profit margin
data_blackwell.drop(columns=["Price", "ProfitMargin"], inplace=True)


#%%

# Aggregating

data_electronidex_sales = data_electronidex.groupby(["category"], as_index=False)[
    ["product_quantity", "price"]
].sum()

data_blackwell_sales = data_blackwell.groupby(["ProductType"], as_index=False)[
    "Volume", "Price_total", "Profit_total"
].sum()

#%%

data_electronidex_sales["product_quantity"] = (
    data_electronidex_sales["product_quantity"]
    .divide(data_electronidex_sales.product_quantity.sum())
    .multiply(100)
)

data_electronidex_sales["price"] = (
    data_electronidex_sales["price"]
    .divide(data_electronidex_sales.price.sum())
    .multiply(100)
)

data_blackwell_sales["Volume"] = (
    data_blackwell_sales["Volume"]
    .divide(data_blackwell_sales.Volume.sum())
    .multiply(100)
)

data_blackwell_sales["Price_total"] = (
    data_blackwell_sales["Price_total"]
    .divide(data_blackwell_sales.Price_total.sum())
    .multiply(100)
)

data_blackwell_sales["Profit_total"] = (
    data_blackwell_sales["Profit_total"]
    .divide(data_blackwell_sales.Profit_total.sum())
    .multiply(100)
)

#%%

# unifying labels for convenience
data_electronidex_sales.columns = ["category", "volume_perc", "price_perc"]
data_blackwell_sales.columns = ["category", "price_perc", "volume_perc", "profit_perc"]

#%%

# combine dataframes for plotting
data_electronidex_sales["Company"] = "Electronindex"
data_blackwell_sales["Company"] = "Blackwell"

data_sales = pd.concat([data_electronidex_sales, data_blackwell_sales], sort=False)

# Dropping the category unknown as it is just confusing
data_sales.query("category != 'Unknown'", inplace=True)

#%%

figures_path = os.path.join("reports", "figures")

# plotting only categories where it forms at least 2 % of one companies sales

blackwell_categories_price = data_sales[
    (data_sales.Company == "Blackwell") & (data_sales.price_perc >= 1)
].category
electronindex_categories_price = data_sales[
    (data_sales.Company == "Electronindex") & (data_sales.price_perc >= 1)
].category

price_cats_to_plot = (
    blackwell_categories_price.append(electronindex_categories_price)
    .drop_duplicates()
    .tolist()
)

blackwell_categories_vol = data_sales[
    (data_sales.Company == "Blackwell") & (data_sales.volume_perc >= 1)
].category
electronindex_categories_vol = data_sales[
    (data_sales.Company == "Electronindex") & (data_sales.volume_perc >= 1)
].category

volume_cats_to_plot = (
    blackwell_categories_vol.append(electronindex_categories_vol)
    .drop_duplicates()
    .tolist()
)

#%%

ax = sns.barplot(
    "category",
    "price_perc",
    hue="Company",
    data=data_sales[data_sales.category.isin(price_cats_to_plot)],
)
ax.set_xlabel("Product Category")
ax.set(ylabel="% of Sales")
plt.title("Product Categories as % of Sales by Price")
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()
plt.savefig(
    os.path.join(figures_path, "sales_distribution_of_product_categories_by_price.png")
)

ax = sns.barplot(
    "category",
    "volume_perc",
    hue="Company",
    data=data_sales[data_sales.category.isin(volume_cats_to_plot)],
)
ax.set_xlabel("Product Category")
ax.set(ylabel="% of Sales")
plt.title("Product Categories as % of Sales by Volume")
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()
plt.savefig(
    os.path.join(figures_path, "sales_distribution_of_product_categories_by_volume.png")
)

price_cats_to_plot_no_accessories = price_cats_to_plot.copy()
price_cats_to_plot_no_accessories.remove("Accessories")
volume_cats_to_plot_no_accessories = volume_cats_to_plot.copy()
volume_cats_to_plot_no_accessories.remove("Accessories")

ax = sns.barplot(
    "category",
    "price_perc",
    hue="Company",
    data=data_sales[data_sales.category.isin(price_cats_to_plot_no_accessories)],
)
ax.set_xlabel("Product Category")
ax.set(ylabel="% of Sales")
plt.title("Product Categories as % of Sales by Price")
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()
plt.savefig(
    os.path.join(
        figures_path,
        "sales_distribution_of_product_categories_by_price_no_accessories.png",
    )
)

ax = sns.barplot(
    "category",
    "volume_perc",
    hue="Company",
    data=data_sales[data_sales.category.isin(volume_cats_to_plot_no_accessories)],
)
ax.set_xlabel("Product Category")
ax.set(ylabel="% of Sales")
plt.title("Product Categories as % of Sales by Volume")
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()
plt.savefig(
    os.path.join(
        figures_path,
        "sales_distribution_of_product_categories_by_volume_no_accessories.png",
    )
)

# Plotting product category share of Blackwell Profits
ax = sns.barplot(
    "category",
    "profit_perc",
    data=data_sales.query("profit_perc > 0.1").sort_values(
        by=["profit_perc"], ascending=False
    ),
)
ax.set_xlabel("Product Category")
ax.set(ylabel="% of Profits")
plt.title("Product Categories Share of Blackwell Profits")
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()
plt.savefig(
    os.path.join(figures_path, "blackwell_profits_share_by_product_category.png")
)

# Plotting Blackwell individual product profitability distribution by category
ax = sns.swarmplot(
    data_blackwell.ProductType, data_blackwell.Profit_per_unit, palette="Paired"
)
ax.set_xlabel("Product Category")
ax.set(ylabel="Profit per Product Sold")
# ax.set_ylim([0, 2500])
plt.title("Product Profit Distribution by Product Category")
plt.xticks(rotation=90)
plt.legend().remove()
plt.show()
plt.savefig(
    os.path.join(
        figures_path, "blackwell_product_profitability_distribution_by_category.png"
    )
)


#%%

