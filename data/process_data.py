'''
    File name: process_data.py
    Author: Andres Hoyos
    Date created: 09/Feb/2021
    Date last modified: 09/Feb/2021
    Python Version: 3.6
'''

#import libraries
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    load_data merges both datasets and join themon index "id"
    
    Input:
    messages_filepath: Path of database containing text features 
    categories_filepath: Path of database containing categories                                features 
       
    output: 
    df: pandas DataFrame
    
    '''
    #load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on="id")
    return df
    

def clean_data(df):
    
    '''
    clean the data and generates binary response for clasifiers
    
    Input:
    df: Joined dataset with the messages
       
    output: 
    df: pandas DataFrame ready for modelling
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # get columns names
    row = categories.head(1)
    category_colnames = row.apply(lambda x:x.str.slice(stop=-2)).values.tolist()[0]
    categories.columns = category_colnames
    
    # Separate columns and encode binary response
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # Clean binary response
        categories.loc[categories['related'] == 2, 'related'] = 1
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df =  pd.concat([df, categories], axis = 1,names=category_colnames)
    
    #remove duplicates
    print('duplicate rows:{}'.format(sum(df.duplicated())))
    df=df.drop_duplicates()
    print('duplicate rows after removal:{}'.format(sum(df.duplicated())))
    
    #Remove colums
    df = df.drop(["child_alone"], axis=1)
    
    return df

def save_data(df, database_filename):
    
    '''
    save de data in a SQLite DataBase
    
    Input:
    df: clean dataframe
    database_filename: Database name
       
    output: 
    None
    ''' 
    
    engine = create_engine("sqlite:///" + database_filename)
    table_name = "T_Categorized_Messages1"
    df.to_sql(table_name, engine, index=False, if_exists="replace")

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