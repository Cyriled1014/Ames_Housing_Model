import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

model = joblib.load('xgb_model.joblib')
pipeline = joblib.load('pipeline.joblib')

st.title('Ames Housing Price Prediction 🏡')
st.markdown("***")

feature_names = ['YearBuilt', 'Bldg_Age', 'OverallQual', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'GrLivArea', 'GarageArea', 'TotalSF', 'PoolArea',
       'TotalFullBaths', 'Neighborhood', 'BldgType', 'HouseStyle',
       'Foundation', 'Electrical', 'Heating', 'GarageType']

# Define feature names and their types (numeric or categorical)
feature_types = {
    'YearBuilt': 'numeric',
    'Bldg_Age': 'numeric',
    'OverallQual': 'numeric',
    'TotalBsmtSF': 'numeric',
    '1stFlrSF': 'numeric',
    '2ndFlrSF': 'numeric',
    'GrLivArea': 'numeric',
    'GarageArea': 'numeric',
    'TotalSF': 'numeric',
    'PoolArea': 'numeric',
    'TotalFullBaths': 'numeric',
    'Neighborhood': 'categorical',
    'BldgType': 'categorical',
    'HouseStyle': 'categorical',
    'Foundation': 'categorical',
    'Electrical': 'categorical',
    'Heating': 'categorical',
    'GarageType': 'categorical'
}

# Define options for categorical features
categorical_options = {
    'Neighborhood': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
       'Blueste'],

    'BldgType': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],

    'HouseStyle': ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf',
       '2.5Fin'],

    'Foundation': ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'],

    'Electrical': ['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix', 'nan'],

    'Heating': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],

    'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'nan', 'Basment', '2Types']
}

# Create two columns layout
col1, col2 = st.columns(2)

# Create a dictionary to store user inputs
user_inputs = {}

# Loop through the feature names and create appropriate input fields
for feature, feature_type in feature_types.items():
    if feature_type == 'numeric':
        with col1:
            user_input = st.number_input(f'{feature}', min_value=0, max_value=10000000, value=0)
    elif feature_type == 'categorical':
        with col2:
            options = categorical_options.get(feature, [])
            user_input = st.selectbox(f'{feature}', options)
    user_inputs[feature] = user_input

# Convert the user_inputs dictionary into a list of lists (2D array-like format)
user_input_data = pd.DataFrame([user_inputs], columns=feature_names)

# Transform user input data using the preprocessor
transformed_data = pipeline.transform(user_input_data)

# Make predictions using the model
prediction = model.predict(transformed_data)

prediction_2 = np.exp(prediction)


# Center the prediction at the top of the page
# st.markdown(f'<div style="text-align: center; font-size: 24px;">Predicted Price: ${prediction_2[0]:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align: center; font-size: 24px;">Predicted Price: <b>${prediction_2[0]:.2f}</b></div>', unsafe_allow_html=True)


st.markdown("***")

def convert_features(data):
    data["MSSubClass"] = data["MSSubClass"].astype(str)
    data["YearBuilt_obj"] = data["YearBuilt"].astype(str)
    data["YearRemodAdd_obj"] = data["YearRemodAdd"].astype(str)
    data["MoSold_obj"] = data["MoSold"].astype(str)
    data["YrSold_obj"] = data["YrSold"].astype(str)
    data["Bldg_Age"] =  data["YrSold"] - data["YearBuilt"]
    return data

def load_data():
    df = pd.read_csv("train.csv")
    df = convert_features(df)
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float32"]]
    cat_cols = [col for col in df.columns if df[col].dtype in ["object"]]
    num_df = df[num_cols]
    cat_cols.append("SalePrice")
    cat_df = df[cat_cols]
    
    return df, num_df, cat_df  # Return both the original DataFrame and num_df

def show_explore_page():

    df, num_df, cat_df = load_data()

    st.subheader("Exploratory Data Analysis of Ames Housing Dataset 🏡")




    st.markdown('**Correlation Heatmap of All Features**')
    #Create the heatmap
    corr = plt.figure(figsize=(12, 8))
    sns.heatmap(num_df.corr())
    st.pyplot(corr)



    skewness = num_df['SalePrice'].skew()
    saleprice = px.histogram(num_df, x = "SalePrice",
                   title=(f'Sale Price Distribution (Skewness: {skewness:.2})'),
                   opacity=0.8,)
    # Set x and y axis labels
    saleprice.update_xaxes(title_text='Sale Price')
    saleprice.update_yaxes(title_text='Frequency')
    st.plotly_chart(saleprice)



    price_age = px.scatter(num_df, x= num_df['Bldg_Age'], y='SalePrice', title='Property Age vs Sale Price')
    price_age.update_xaxes(title_text='Building Age')
    price_age.update_yaxes(title_text='Sale Price')
    # Display the plot in Streamlit
    st.plotly_chart(price_age)

    
    price_grliveare = px.scatter(num_df, x= num_df['GrLivArea'], y='SalePrice', title='GrLivArea vs Sale Price')
    price_grliveare.update_xaxes(title_text='Ground Living Area')
    price_grliveare.update_yaxes(title_text='Sale Price')
    st.plotly_chart(price_grliveare)
    

    year = cat_df["YearBuilt_obj"].value_counts()
    year_avgprice = cat_df.groupby("YearBuilt_obj")["SalePrice"].mean()
    avg_saleprice_year = px.bar(cat_df, x=year.index, y=year_avgprice, title="Average Sale Price by Year Built")
    avg_saleprice_year.update_yaxes(title_text='Sale Price')
    avg_saleprice_year.update_xaxes(title_text='Year Built')
    st.plotly_chart(avg_saleprice_year)


    neigborhood= cat_df["Neighborhood"].value_counts()
    neigborhood_avgprice = cat_df.groupby("Neighborhood")["SalePrice"].mean()
    neigborhood_avgprice_fig = px.bar(cat_df, x=neigborhood.index, y=neigborhood_avgprice , title='Average Sale Price by Neigborhood')
    neigborhood_avgprice_fig.update_yaxes(title_text='Sale Price')
    neigborhood_avgprice_fig.update_xaxes(title_text='Neighborhood')
    st.plotly_chart(neigborhood_avgprice_fig)

    neigborhood_avgprice_box = px.box(df, x="Neighborhood", y="SalePrice", title = "Box Plot of SalePrice by Neighborhood")
    neigborhood_avgprice_box.update_yaxes(title_text='Sale Price')
    neigborhood_avgprice_box.update_xaxes(title_text='Neighborhood')
    st.plotly_chart(neigborhood_avgprice_box)


    lotarea_neighborhood = df["Neighborhood"].value_counts()
    avg_lotarea_neighborhood  = df.groupby("Neighborhood")["LotArea"].mean()
    avg_lotarea_neighborhood_fig = px.bar(cat_df, x=lotarea_neighborhood.index, y=avg_lotarea_neighborhood, title='Average Lot Area by Neighborhood')
    avg_lotarea_neighborhood_fig.update_yaxes(title_text='Lot Area')
    avg_lotarea_neighborhood_fig.update_xaxes(title_text='Neighborhood')
    st.plotly_chart(avg_lotarea_neighborhood_fig)

    avg_lotarea_neighborhood_box = px.box(df, x="Neighborhood", y="LotArea", title = "Box Plot of Lot Area by Neighborhood")
    avg_lotarea_neighborhood_box.update_yaxes(title_text='Lot Area')
    avg_lotarea_neighborhood_box.update_xaxes(title_text='Neighborhood')
    st.plotly_chart(avg_lotarea_neighborhood_box)


    Foundation = cat_df["Foundation"].value_counts()
    Foundation_avgprice = cat_df.groupby("Foundation")["SalePrice"].mean()
    Foundation_avgprice_fig = px.bar(cat_df, x=Foundation.index, y=Foundation_avgprice, title='Average Sale Price by Foundation')
    Foundation_avgprice_fig.update_yaxes(title_text='Sale Price')
    Foundation_avgprice_fig.update_xaxes(title_text='Foundation')
    st.plotly_chart(Foundation_avgprice_fig)


    electrical = cat_df["Electrical"].value_counts()
    electrical_avgprice = cat_df.groupby("Electrical")["SalePrice"].mean()
    electrical_avgprice_fig = px.bar(cat_df, x=electrical.index, y=electrical_avgprice, title='Average Sale Price by Electrical Type')
    electrical_avgprice_fig.update_yaxes(title_text='Sale Price')
    electrical_avgprice_fig.update_xaxes(title_text='Electrical Type')
    st.plotly_chart(electrical_avgprice_fig)

    heating = cat_df["Heating"].value_counts()
    heating_avgprice = cat_df.groupby("Heating")["SalePrice"].mean()
    heating_avgprice_fig = px.bar(cat_df, x=heating.index, y=heating_avgprice, title='Average Sale Price by Heating Type')
    heating_avgprice_fig.update_yaxes(title_text='Sale Price')
    heating_avgprice_fig.update_xaxes(title_text='Heating Type')
    st.plotly_chart(heating_avgprice_fig)

    garage= cat_df["GarageType"].value_counts()
    garage_avgprice = cat_df.groupby("GarageType")["SalePrice"].mean()
    garage_avgprice_fig = px.bar(cat_df, x=garage.index, y=garage_avgprice, title='Average Sale Price by Garage Type')
    garage_avgprice_fig.update_yaxes(title_text='Sale Price')
    garage_avgprice_fig.update_xaxes(title_text='Garage Type')
    st.plotly_chart(garage_avgprice_fig)
    

# Run the explore page
show_explore_page()
