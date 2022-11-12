import sys
import pandas as pd
import numpy as np
import plotly.express as px


from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    
    '''
    load_data
    Function for loading data from our raw files containing the messages and the categories and merging it into a simple dataframe
    
    Input:
    messages_filepath: filepath to messages.csv file
    categories_filepath: filepath to categories.csv file
    
    Output:
    df: merged dataframe containing both messages and their categories.
    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merging the messages with the categories
    df = messages.merge(categories, on=('id'))
    return df


def clean_data(df):    
    
    '''
    clean_data
    Function for cleaning the data by processing the merged data frame into a tabular form with clean category names as columns, and numeric binary values for each category and message. 
    
    Input:
    df: loaded and merged data frame, unprocessed.
    
    Output:
    df: cleaned and processed data frame.
    
    '''
    
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
    
    # finding out any non-binary values in the unique values of the data frame
    unique_vals=np.unique(categories_df.values)
    non_binary_values=[]
    # for each uique value in the data set, we check if it binary
    for i in range(len(unique_vals)):
        if ((unique_vals[i]!=0) & (unique_vals[i]!=1)).any():
            # every non-binary value is stored in a separate list
            non_binary_values.append(unique_vals[i])
    
    # replacing all non-binary values (if existing) in the data set with 1
    if len(non_binary_values)>0:
        for c in categories_df:
            categories_df[c] = categories_df[c].replace(non_binary_values, 1)


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
    
    '''
    save_data
    Function for saving the processed and clean dataframe into a SQLITE Database, as a table.
    
    Input:
    df: processed and cleaned dataframe.
    database_filename: name given to the SQLITE database.
    
    Output:
    Database Table: Table created in the SQLITE Database, containing the processed dataframe.
    
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DIS_RESP', engine, if_exists='replace', index=False)
    return   


def main():
    
    '''
    main function: starting point for the execution of the process by triggering the loading, cleaning, and saving the data
    Input: none
    Output: Cleaned data saved to database
    
    '''
 
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