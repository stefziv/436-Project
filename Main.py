import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

def encoder(x):
    if x == 'Excellent' or x == '>100' or x == 'Ex':
        x = 5
    elif x == 'Good' or x == '99-90' or x == 'Gd':
        x = 4
    elif x == 'Average' or x == '89-80' or x == 'TA':
        x = 3
    elif x == 'Fair' or x == '79-70' or x == 'Fa':
        x = 2
    elif x == 'Poor' or x == '<70' or x == 'Po':
        x = 1
    else:
        x = 0

    return x

def encode_input(dict):
    for key in dict.keys():
        if key in ['Exterior Quality', 'Kitchen Quality', 'Basement Height']:
            dict[key] = [encoder(dict[key])]
        else:
            dict[key] = [dict[key]]
    
    return [dict]


def encode_object_columns(dataframe):
    temp = dataframe.copy()
    for column in temp.columns:
        if temp[column].dtype == 'object':
            temp[column] = temp.apply(lambda x: encoder(x[column]), axis=1)
            

        if np.issubdtype(temp[column].dtype, np.floating) or np.issubdtype(temp[column].dtype, np.integer):
            if temp[column].isnull().any():
                temp[column].fillna(value=0, inplace=True)  # Encode NaN values as 0
        
    temp = temp.apply(lambda x: (x - x.min())/(x.max() - x.min()))
    return temp

def get_data():
    column_mapping = {
        'OverallQual': 'Overall Quality',
        'ExterQual': 'Exterior Quality',
        'BsmtQual': 'Basement Height',
        'KitchenQual': 'Kitchen Quality',
        'GrLivArea': 'Above Ground Living Area',
        'GarageArea': 'Garage Area',
        'TotalBsmtSF': 'Basement Area',
        'FullBath': 'Full Bathrooms',
        'TotRmsAbvGrd': '# of Rooms',
        'GarageCars': 'Size of Garage',
        'YearBuilt': 'Construction Date',
        'YearRemodAdd': 'Remodel Date',
        'YrSold' : 'Year Sold',
        'SalePrice' : 'Sale Price'
    }

    columns = ['Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area','Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage','Construction Date', 'Remodel Date', 'Year Sold', 'Sale Price']
    raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')
    raw_data_train = raw_data_train.rename(columns=column_mapping)

    train_data = raw_data_train[columns]
    datatypes = train_data.dtypes

    if len(st.session_state.df_sold) > 0:
        train_data = pd.concat([train_data, st.session_state.df_sold], ignore_index=True)
        train_data = train_data.astype(datatypes)

    train_data = encode_object_columns(train_data)

    st.session_state.data = train_data
    


def make_predictions():
    st.session_state.predictions = st.session_state.lm.predict(encode_object_columns(st.session_state.df))

def train_model():
    final_features = ['Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area','Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage','Construction Date', 'Remodel Date']

    data = st.session_state.data

    if "start_end" in st.session_state:
        data = data[(data['Year Sold'] >= st.session_state.start_end['start']) & (data['Year Sold'] <= st.session_state.start_end['end'])]

    X = data[final_features]
    y = data.values[:, -1]
    st.session_state.lm_params = X,y
    lm = LinearRegression()
    lm.fit(X,y)
    st.session_state.lm = lm

def delete_prediction(index):
    st.session_state.predictions = np.delete(st.session_state.predictions , index)

def delete_house(index):
    st.session_state.df = st.session_state.df.drop(index=index,axis=0)
    st.session_state.df = st.session_state.df.reset_index(drop=True)
    delete_prediction(index)

def delete_sold_house(index):
    st.session_state.df_sold = st.session_state.df_sold.drop(index=index,axis=0)
    st.session_state.df_sold = st.session_state.df_sold.reset_index(drop=True)
    get_data()
    train_model()
    make_predictions()


def sold_house(index, sale_price):
    today = datetime.date.today()
    year = today.year
    house = st.session_state.df.loc[index]
    house['Sale Price'] = sale_price
    house['Year Sold'] = year
    st.session_state.df = st.session_state.df.drop(index=index,axis=0)
    st.session_state.df = st.session_state.df.reset_index(drop=True)
    rw = st.session_state.df_sold.shape[0]
    st.session_state.df_sold.loc[rw] = house
    delete_prediction(index)
    get_data()
    train_model()
    make_predictions()


def main():

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=[
          'Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area',
          'Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage',
          'Construction Date', 'Remodel Date'
        ])

    if "df_sold" not in st.session_state:
        st.session_state.df_sold = pd.DataFrame(columns=[
          'Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area',
          'Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage',
          'Construction Date', 'Remodel Date', 'Year Sold', 'Sale Price'
        ])

    if "data" not in st.session_state:
        get_data()

    if "lm" not in st.session_state:
        train_model()

    st.title("House Management")

    st.header("Filter Model Data")
    min_year = st.session_state.data['Year Sold'].min()
    max_year = st.session_state.data['Year Sold'].max()
    st.text('Enter a start and end year if you would like to train the model on a subset of')
    if "start_end" not in st.session_state:
        st.text('listings sold within the provide timeframe. Current Range: ' + str(min_year) + '-' + str(max_year))
    else:
        st.text('listings sold within the provide timeframe. Current Range: ' + str(st.session_state.start_end['start']) + '-' + str(st.session_state.start_end['end']))

    col1, col2 = st.columns(2)

    with col1:
        start_year = st.number_input("Start Year", min_value=min_year, max_value=datetime.date.today().year, step=1)

    with col2:
        end_year = st.number_input("End Year", value=datetime.date.today().year, min_value=start_year, max_value=datetime.date.today().year, step=1)

    if st.button("Filter and Train Model"):
        st.session_state.start_end = {}
        st.session_state.start_end['start'] = start_year
        st.session_state.start_end['end'] = end_year
        train_model()
        make_predictions()
        st.experimental_rerun()

    if "start_end" in st.session_state:
        if st.button("Reset Date Filters"):
            del st.session_state.start_end
            train_model()
            make_predictions()
            st.experimental_rerun()

    st.divider()
    
    st.header("Current Listings")

    if len(st.session_state.df) > 0:
        st.dataframe(st.session_state.df)

        st.subheader("Suggested List Price")

        i = 0
        for x in st.session_state.predictions:
            st.text('House ' + str(i) + ': ' + str(round(x,0)))
            i = i + 1
        
        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Sold a House?")
            sale_index = st.number_input("Enter the index of the house", min_value=0, max_value=len(st.session_state.df)-1, step=1)
            sale_price = st.number_input("Enter the sale price of the house", min_value=0, step=1)

            if st.button("Confirm"):
                sold_house(sale_index, sale_price)
                st.success("House sold successfully!")
                st.experimental_rerun()

        with col4:
            st.subheader("Delete a House")
            delete_index = st.number_input("Enter the index of the house to delete", min_value=0, max_value=len(st.session_state.df)-1, step=1)

            if st.button("Delete House"):
                delete_house(int(delete_index))
                st.success("House deleted successfully!")
                st.experimental_rerun()
    else:
        st.caption("No Current Listings. Please use the 'Add House' tab to add a listing.")

    st.divider()

    st.header("Houses Sold")

    if len(st.session_state.df_sold) > 0:
        st.dataframe(st.session_state.df_sold)

        st.subheader("Delete a House")
        delete_sold_index = st.number_input("Enter the index of the sold house to delete", min_value=0, max_value=len(st.session_state.df_sold)-1, step=1)

        if st.button("Delete Sold House"):
            delete_sold_house(int(delete_sold_index))
            st.success("House deleted successfully!")
            st.experimental_rerun()
    else:
        st.caption("No Houses Sold. Please use the 'Add House' tab to add a listing then enter the sale price.")




if __name__ == '__main__':
    main()
