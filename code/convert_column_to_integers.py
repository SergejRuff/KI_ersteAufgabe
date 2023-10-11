"""
function which turns character types
into integer

"""


def convert_column_to_integers(df, column_name):
    """
    convert unique characters in a dataframe col to integer-type
    important for decisiontree.
    :param df: dataframe with coloumn containing characters
    :param column_name: coloumn with characters
    :return: dataframe with characters in coloumn being changed to integer.
    """
    unique_values = df[column_name].unique()  # extract unique characters
    # creat a dictionary with unique colvalue as key and its index as value
    value_to_integer_mapping = {value: i for i, value in enumerate(unique_values)}
    # replace each coloumn value with number(index) if character matches key.
    df[column_name] = df[column_name].map(value_to_integer_mapping)


def convert_string_columns_to_integers(df):
    """
    convert whole dataframe, not only single coloumns
    :param df:  dataframe
    :return: dataframe with characters(object-type) changes to integer.
    """
    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            convert_column_to_integers(df, column_name)
