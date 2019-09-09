#%% [markdown]

# # Data Quality Exploration

# ## Conclusions

# 1. Prices and item quantity sometimes clearly wrong
# 2. Only 0.1 % of the items don't have a matching order
# 3. 7 % of the orders don't have any matching items, but these orders are all from orders with states 'Place order' and 'Shopping Basket'. So no problems with completed orders
# 4. Only around 20 % of the completed orders have matching price between the order price and sum of the prices of the individual items. The percent of the orders with matching prices also shows clear timetrends. The difference almost always small and could be caused by shipping costs and discounts, that we do not see in the data.

#%%

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML

# Setting styles
sns.set(style="whitegrid", color_codes=True)
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline

#%%

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


#%%

# Sample data
data_trans.head()
data_orders.head()
data_items.head()

#%%

# rows plot
rows_list = [
    ("Transactions", len(data_trans)),
    ("Orders", len(data_orders)),
    ("Items", len(data_items))
]
labels_list = ['Dataset', 'Rows']

row_counts_df = pd.DataFrame.from_records(rows_list, columns=labels_list)
sns.barplot('Dataset', 'Rows', data=row_counts_df)
plt.title("Number of Rows in Datasets")
plt.show()

#%%

# Number of missing values
data_trans.isnull().sum().plot(kind='bar')
plt.title("Transactions Missing by Columns")
plt.show()
data_orders.isnull().sum().plot(kind='bar')
plt.title("Orders Missing by Columns")
plt.show()
data_items.isnull().sum().plot(kind='bar')
plt.title("Items Missing by Columns")
plt.show()

#%%

# checking out more closely the 5 orders with missing values

data_orders[data_orders.total_paid.isnull()]

# All are orders that are pending. This looks okay. We are probably going
# to drop these rows anyway

#%% fixing data types and adding more columns

# Adding total price to items that takes into account the number of items
data_items["total_items_price"] = data_items["unit_price"] * data_items["product_quantity"]

# correcting the datetime columns
data_orders.created_date = pd.to_datetime(data_orders.created_date)
data_items.date = pd.to_datetime(data_items.date)

# adding normalized versions for easier handling
data_orders["created_date_norm"] = data_orders["created_date"].dt.normalize()
data_items["date_norm"] = data_items["date"].dt.normalize()

# Lets combine the orders and items with an inner join
data_orders_items = data_orders.join(
    data_items.set_index("id_order"), how="outer", on="id_order", lsuffix="_item"
)

data_orders_items_total_price = (
    data_orders_items[["id_order", "total_items_price"]]
    .groupby(["id_order"], as_index=False)
    .sum()
)

data_orders_items_total_price.columns = ["id_order", "total_price_items"]

data_orders_items = data_orders_items.join(
    data_orders_items_total_price.set_index("id_order"), how="outer", on="id_order"
)


#%% checking that timestamps are within our designated period
# 2017-01-01 00:07:19 - 2018-03-14 13:58:36

min(data_orders.created_date)
max(data_orders.created_date)
min(data_items.date)
max(data_items.date)

# everything is within the designated period

#%%

# Checking duplicates for orders and items. Order ID should be unique
# in orders dataset and ID in items dataset.

sum(data_orders.id_order.value_counts() > 1)
sum(data_items.id.value_counts() > 1)

# no duplicates

#%%

# Checking if the paid sum in orders has weird values

sns.boxplot(data_orders.total_paid.dropna())
plt.show()

# There seem to be lots of values that are zero or close to it.
# There are also lots of values that have extremely high values

# The distrbution looks very much like normal distribution now.
# It seems that it is okay to think about total paid as following
# exponential distribution with a normally distributed error.

#%%

# Lets inspect which kind of orders are those that have 0 paid

sns.countplot(x="state", data=data_orders[data_orders.total_paid == 0])
plt.xticks(rotation=90)
plt.show()

# All these rows are in shopping basket stage or place order state.
# This is good as we will probably drop these categories from the
# analysis

#%%

# Lets still check if there are any values that are below 0
data_orders[data_orders.total_paid < 0].shape[0]

# No

#%%

# Checking the distribution of numerical values in items

items_check_list = ["product_quantity", "unit_price"]

for feature in items_check_list:
    sns.boxplot(x=data_items[feature])
    plt.show()
    print(feature + " min: " + str(min(data_items[feature])))
    print(feature + " max: " + str(max(data_items[feature])))

# Same thing as with amount paid in orders: median 0 and extreme outliers.
# There are also some values that are below 0 in unit_price. Total price comes is
# calculated with unit price so it does not give additional value.

#%% Lets check more closely the rows that are below zero

data_items[data_items.unit_price < 0]

# Only one row. Probably a mistake, but could also be a refund that is marked in weird way

#%%

# Lets see how well orders and items combine

data_orders_items.isnull().sum()
data_orders_items.isnull().mean().multiply(100).round(1)
data_orders_items.id_order.nunique()

# 234 (0.1 %) of the items dont' have a matching orderd
# and 22213 (7.0 %) of the orders don't have a matching items

#%% Lets check that there are still no duplicate items even after the join
# no duplicates

sum(data_orders_items.id.value_counts() > 1)

#%%

# Lets look more closely at the orders that don't have any items
data_orders_items[data_orders_items.id.isnull()].state.value_counts().plot(kind="bar")

# All are in states Place order and Shopping basket.
# These seems reasonable and should not cause a problem

#%% Lets see if the price from orders and from items matches trough time

data_orders_ts = (
    data_orders[["created_date_norm", "total_paid"]]
    .groupby(["created_date_norm"], as_index=False)
    .agg(["count", "sum"])
)

data_items_ts = (
    data_items[["date_norm", "total_items_price"]]
    .groupby(["date_norm"], as_index=False)
    .agg(["count", "sum"])
)

fig, ax = plt.subplots()
sns.lineplot(data_orders_ts.index, data_orders_ts.total_paid["count"], color="r", ax=ax)
sns.lineplot(data_items_ts.index, data_items_ts.total_items_price["count"], ax=ax, color="b")
ax.legend(("orders", "items"), loc="upper left")
plt.xlabel("Date")
plt.ylabel("Count of Items")
plt.show()

fig, ax = plt.subplots()
sns.lineplot(data_orders_ts.index, data_orders_ts.total_paid["sum"], color="r", ax=ax)
sns.lineplot(data_items_ts.index, data_items_ts.total_items_price["sum"], ax=ax, color="b")
ax.legend(("orders", "items"), loc="upper left")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# There are less orders than items, so okay on that front

# The price for orders and items lines up quite well, but not completely.
# This could be of shipping costs and discounts?

#%% Checking timeline of the amount of items per order for completed orders

data_orders_items_ts = (
    data_orders_items[data_orders_items.state == "Completed"][
        ["created_date_norm", "id_order", "product_quantity"]
    ]
    .groupby(["created_date_norm", "id_order"])
    .agg(["sum", "count"])
)

fig, ax = plt.subplots()
sns.lineplot(
    data_orders_items_ts.index.get_level_values("created_date_norm"),
    data_orders_items_ts.product_quantity["sum"],
    ax=ax,
    label="Number of Items",
)
sns.lineplot(
    data_orders_items_ts.index.get_level_values("created_date_norm"),
    data_orders_items_ts.product_quantity["count"],
    ax=ax,
    label="Unique Items",
)
ax.legend()
plt.xlabel("Date")
plt.ylabel("Items")
plt.show()

# The average number of items and unique items is below two, and the variation
# is not huge. Probably the number of orders that have a two or more items
# is actually quite small in comparison

#%% checking how many items have more than two items in completed orders

data_orders_items_only_two_ts = data_orders_items_ts[
    data_orders_items_ts.product_quantity["count"] >= 2
]

data_orders_items_only_two_ts.shape[0]

# 10453 orders. This matches the rows we have in trans.csv.
# Everything looks to be in order

#%%

data_orders_items_unique_order = data_orders_items.drop_duplicates("id_order")
data_orders_items_unique_order["price_matches"] = np.where(
    data_orders_items_unique_order.total_paid
    == data_orders_items_unique_order.total_items_price,
    True,
    False,
)

#%% Checking the percent of orders that have the matching price

# looking only at orders from the order dataset. We can use the total_paid
# from the order dataset to achieve this

data_orders_items_unique_order_price_match_ts = (
    data_orders_items_unique_order.query("price_matches == True")
    .dropna(subset=["total_paid"])[["created_date_norm", "id_order", "state"]]
    .groupby(["created_date_norm", "state"])
    .count()
)

data_orders_items_unique_order_all_ts = (
    data_orders_items_unique_order.dropna(subset=["total_paid"])[
        ["created_date_norm", "id_order", "state"]
    ]
    .groupby(["created_date_norm", "state"])
    .count()
)

data_orders_items_unique_order_price_match_ts.columns = ["matching_orders"]
data_orders_items_unique_order_all_ts.columns = ["all_orders"]

data_orders_items_unique_order_price_match_ts = data_orders_items_unique_order_price_match_ts.join(
    data_orders_items_unique_order_all_ts, how="left"
)

data_orders_items_unique_order_price_match_ts[
    "percent_matching"
] = data_orders_items_unique_order_price_match_ts.apply(
    lambda row: round(row.matching_orders * 100 / row.all_orders), axis=1
)

data_orders_items_unique_order_price_match_ts.reset_index(inplace=True)

data_orders_items_unique_order_price_match_ts.head()

#%%


plot = sns.relplot(
    "created_date_norm",
    "percent_matching",
    hue="state",
    col="state",
    col_wrap=2,
    data=data_orders_items_unique_order_price_match_ts,
    kind="line",
)
plt.xlabel("Date")
plt.show()

# Only around 20 % of the orders have matching prices and these
# big temporal changes dropping down to almost 0 by 2018

#%% Checking the distribution of the difference between prices

data_price_diff = data_orders_items_unique_order.dropna(
    subset=["total_paid", "total_items_price"]
).query("price_matches == False")[["state", "total_paid", "total_items_price"]]

data_price_diff["price_diff"] = data_price_diff.apply(
    lambda row: row.total_paid - row.total_items_price, axis=1
)

#%%

# same but with filtering differences above and below 30

plot = sns.catplot(
    y="price_diff",
    hue="state",
    col="state",
    col_wrap=2,
    kind="violin",
    data=data_price_diff[data_price_diff.price_diff.abs() <= 50],
)
plt.show()

# Order prices are bit higher than the price from individual items
# It think that this is because of the shipping fees and the adjustments
# down are discounts. There are rare extreme differences and these are
# probably true errors


#%%
