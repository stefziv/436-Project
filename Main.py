import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

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

    return temp

def make_predictions():
    st.session_state.predictions = st.session_state.lm.predict(encode_object_columns(st.session_state.df))

def train_model():
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
        'YearRemodAdd': 'Remodel Date'
    }

    final_features = ['Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area','Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage','Construction Date', 'Remodel Date']

    raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')
    raw_data_train = raw_data_train.rename(columns=column_mapping)
    X = encode_object_columns(raw_data_train[final_features])
    y = raw_data_train.values[:, -1]
    lm = LinearRegression()
    lm.fit(X,y)
    return lm

def delete_house(index, df):
    df = df.drop(index=index,axis=0)
    df = df.reset_index(drop=True)

    return df

    #remove prediction from predictions

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
          'Construction Date', 'Remodel Date', 'Sale Price'
        ])

    if "lm" not in st.session_state:
        st.session_state.lm = train_model()

    st.title("House Management")

    st.header("Current Listings")

    if len(st.session_state.df) > 0:
        st.dataframe(st.session_state.df)

        st.subheader("Suggested List Price")

        i = 0
        for x in st.session_state.predictions:
            st.text('House ' + str(i) + ': ' + str(round(x,0)))
            i = i + 1

        st.subheader("Delete a House")
        delete_index = st.number_input("Enter the index of the house to delete", min_value=0, max_value=len(st.session_state.df)-1, step=1)

        if st.button("Delete House"):
            st.session_state.df = delete_house(int(delete_index), st.session_state.df)
            st.success("House deleted successfully!")
            st.experimental_rerun()
    else:
        st.caption("No Current Listings. Please use the 'Add House' tab to add a listing.")

    st.header("Houses Sold")

    if len(st.session_state.df_sold) > 0:
        st.dataframe(st.session_state.df)

        st.subheader("Suggested List Price")

        i = 0
        for x in st.session_state.predictions:
            st.text('House ' + str(i) + ': ' + str(round(x,0)))
            i = i + 1

        st.subheader("Delete a House")
        delete_index = st.number_input("Enter the index of the house to delete", min_value=0, max_value=len(st.session_state.df)-1, step=1)

        if st.button("Delete House"):
            house_df = delete_house(int(delete_index))
            st.success("House deleted successfully!")
            st.experimental_rerun()
    else:
        st.caption("No Houses Sold. Please use the 'Add House' tab to add a listing then enter the sale price.")




if __name__ == '__main__':
    main()