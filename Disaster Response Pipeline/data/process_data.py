import sys
import pandas as pd
import numpy as np
import plotly.express as px


from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merging the messages with the categories
    df = messages.merge(categories, on=('id'))
    return df


def clean_data(df):    
    
    # create a dataframe of the 36 individual category columns
    categories_df = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row=np.array(categories_df.head(1))
    # extract a list of new column names for categories-by eliminating the digits form the name
    categories_df_cols = [x[:-2] for x in row[0]]
    print(categories_df_cols)
    # rename the columns of `categories`
    categories_df.columns = categories_df_cols
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for c in categories_df:
        # set each value to be the last character of the string
        categories_df[c] = categories_df[c].str[-1]
        # convert column from string to numeric
        # taken from https://www.linkedin.com/pulse/change-data-type-columns-pandas-mohit-sharma/
        categories_df[c] = pd.to_numeric(categories_df[c]) 


    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    # below taken from https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    frames = [df, categories_df]
    df = pd.concat(frames, axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DIS_RESP', engine, if_exists='replace', index=False)
    return   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()