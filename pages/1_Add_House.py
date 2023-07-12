import streamlit as st
import pandas as pd
import numpy as np

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

def main():

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=[
          'Overall Quality', 'Exterior Quality', 'Kitchen Quality', 'Basement Height', 'Above Ground Living Area',
          'Basement Area', 'Garage Area', '# of Rooms', 'Full Bathrooms', 'Size of Garage',
          'Construction Date', 'Remodel Date'
        ])

    st.title("House Management")

    st.header("Add House")
    with st.form(key="add form", clear_on_submit= True):
        ExterQual = st.radio('Exterior Quality', ['Excellent','Good','Average','Fair','Poor'], horizontal=True)
        KitchenQual = st.radio('Kitchen Quality', ['Excellent','Good','Average','Fair','Poor'], horizontal=True)
        BsmtQual = st.radio('Basement Height (Inches)', ['>100','99-90','89-80','79-70','<70','No Basement'], horizontal=True)
        OverallQual = st.slider('Overall Quality', min_value=1, max_value=10,step=1)
        GrLivArea = st.number_input("Above Ground Living Area (Square Feet)", step=1)
        TotalBsmtSF = st.number_input("Basement Area (Square Feet)", step=1)
        GarageArea = st.number_input("Garage Area (Square Feet)", step=1)
        TotRmsAbvGrd = st.number_input("# of Rooms Above Ground (Not Including Bathrooms)", step=1)
        FullBath = st.number_input("Full Bathrooms Above Ground", step=1)
        GarageCars = st.number_input("Size of Garage in Car Capacity", step=1)
        YearBuilt = st.number_input("Original Construction Date", step=1)
        YearRemodAdd = st.number_input("Remodel Date (Construction Date if no Remodeling)", step=1)

        new_house = {
            'Overall Quality': OverallQual,
            'Exterior Quality': ExterQual,
            'Kitchen Quality': KitchenQual,
            'Basement Height': BsmtQual,
            'Above Ground Living Area': GrLivArea,
            'Basement Area': TotalBsmtSF,
            'Garage Area': GarageArea,
            '# of Rooms': TotRmsAbvGrd,
            'Full Bathrooms': FullBath,
            'Size of Garage': GarageCars,
            'Construction Date': YearBuilt,
            'Remodel Date': YearRemodAdd,
        }

        if st.form_submit_button("Add"):
            rw = st.session_state.df.shape[0]
            st.session_state.df.loc[rw] = new_house
            st.success("House added successfully!")
            make_predictions()


if __name__ == '__main__':
    main()