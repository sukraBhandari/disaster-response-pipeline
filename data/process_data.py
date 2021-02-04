import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads two csv files and merges into one dataframe
    
    Parameters
    ----------
    messages_filepath : path for messages csv file
    categories_filepath : path for categories csv file
    
    Returns
    -------
    df : merged dataframe
    '''
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories)
    return df

def clean_data(df):
    '''
    Performs data cleansing
    
    Parameters
    ----------
    df : dataframe for data cleansing
    
    Returns
    -------
    df : cleaned and process data 
    '''
    # split categories and expand columns on each split value
    categories = df.categories.str.split(';', expand=True)    
    # extract a list of new column names for categories
    category_colnames = categories.iloc[0].str[:-2]    
    # rename the columns names
    categories.columns = category_colnames    
    # Iterate through the category columns in df
    # keep only the last character of each string (the 1 or 0).
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    # Drop the categories column from the df dataframe
    df.drop(['categories'], axis=1, inplace=True)    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)    
    # drop duplicates
    df.drop_duplicates(inplace=True)    
    return df

def save_data(df, database_filename):
    '''
    Writes dataframe to sqlite database
    
    Parameters
    ----------
    df : dataframe
    database_filename : database filename
    
    Returns
    -------
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("disaster_messages_clean", engine, if_exists='replace', index=False)


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