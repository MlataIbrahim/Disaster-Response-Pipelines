# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# load messages dataset
messages = pd.read_csv('messages.csv')
# load categories dataset
categories = pd.read_csv('categories.csv')
# merge datasets
df = pd.merge(messages,categories,on='id')
# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(pat=";", expand=True)
# select the first row of the categories dataframe
row = categories.loc[:0,:]
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
print(category_colnames)
# rename the columns of `categories`
categories.columns = category_colnames
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
# drop the original categories column from `df`
df.drop("categories", axis=1,inplace=True)
# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis=1)
# drop duplicates
df.drop_duplicates(inplace=True)
engine = create_engine('sqlite:///DisatserProject.db')
df.to_sql('Disatser', engine, index=False)
