# Ames_Housing_Model

Title: Ames Housing Prediction Streamlit App

Project Description:
This project involves creating a Streamlit app for predicting Ames housing prices. The dataset contains relevant information on residential properties sold in Ames, Iowa, between 2006 and 2010. It was extracted from the Ames Assessor’s Office and compiled by De Cock (1).

Objectives:
To conduct an exploratory data analysis on the dataset.
To create a model with a minimal set of features
To develop a Streamlit web application that integrates the best-trained model and provides a visually compelling presentation of the data analysis.

Data: 
The data was obtained from the Kaggle Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview. Additional information about the features can be found here. The zip file includes two datasets: the training dataset and the test dataset, both in CSV format. The dataset comprises 79 features, encompassing details such as LotArea, Neighborhood, BldgType, and other relevant features associated with a residence. The predicted variable is SalePrice.

However, the model used only 18 features: YearBuilt, Bldg_Age, OverallQual, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageArea, TotalSF, PoolArea, TotalFullBaths, Neighborhood, BldgType, HouseStyle, Foundation, Electrical, Heating, and GarageType. Of these, 3 were engineered specifically: TotalSF, TotalFullBaths, and Bldg_Age.

It was noted that the SalePrice has a skewness of 1.9; thus, it was normalized for training the model. In the preprocessing of the data, for numerical features, a simple imputer with the mean as the strategy and a standard scaler was used. On the other hand, for categorical features, a simple imputer with a constant as the strategy and One-Hot Encoder were used.
