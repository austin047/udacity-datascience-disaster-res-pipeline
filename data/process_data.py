import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Description: Load data from all dataset 'messages' and 'categories' in seperate data frames, merge and return  merged dataframe
    
    Arguments:
        messages_filepath: The filepath to the message dataset
        categories_filepath: THe filepath to the categories dataset
        
    Return:
        df: DataFrame from merging messages and categories datasets 
        
    '''
    
    # Load datasets 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two data sets on 'id' column 
    df = messages.merge(right=categories, on='id', )
    
    return df

def clean_data(df):
    '''
    Description: Clean data from df Dataframe and return new clean Dataframe 
    
    Arguments:
        df: DataFrame
    
    Return:
        df: Cleaned Data Frame
    '''
    
    # Split categories into separate category columns.
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0:1]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = pd.Series(row.values[0]).apply(lambda str: str[:len(str) - 2])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column] = categories[column].apply(lambda str: str[len(str) - 1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(labels=['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # return cleaned dataframe
    return df

def save_data(df, database_filename):
    '''
    Description: Save Dataframe df into specified database
    
    Arguments:
        df: Dataframe
        database_filename:  Data base file name 
        
    Return:
        None
    '''
    
    # Initialize sqlite engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # Save dataframe to sql
    df.to_sql('message_and_category', engine, index=False)


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