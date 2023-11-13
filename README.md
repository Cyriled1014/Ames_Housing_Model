# Title: Ames Housing Prediction Streamlit App

## Project Description:
This project involves creating a Streamlit app for predicting Ames housing prices. The dataset contains relevant information on residential properties sold in Ames, Iowa, between 2006 and 2010. It was extracted from the Ames Assessor’s Office and compiled by De Cock (1).

## Objectives:
* To conduct an exploratory data analysis on the dataset.
* To create a model with a minimal set of features (less than 20).
* To develop a Streamlit web application that integrates the best-trained model and provides a visually compelling presentation of the data analysis.

## Data: 
The data was obtained from the Kaggle Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview. Additional information about the features can be found here. The zip file includes two datasets: the training dataset and the test dataset, both in CSV format. The dataset comprises 79 features, encompassing details such as LotArea, Neighborhood, BldgType, and other relevant features associated with a residence. The predicted variable is SalePrice.

However, the model used only 18 features: YearBuilt, Bldg_Age, OverallQual, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageArea, TotalSF, PoolArea, TotalFullBaths, Neighborhood, BldgType, HouseStyle, Foundation, Electrical, Heating, and GarageType. Of these, 3 were engineered specifically: TotalSF, TotalFullBaths, and Bldg_Age.

It was noted that the SalePrice has a skewness of 1.9; thus, it was normalized for training the model. In the preprocessing of the data, for numerical features, a simple imputer with the mean as the strategy and a standard scaler was used. On the other hand, for categorical features, a simple imputer with a constant as the strategy and One-Hot Encoder were used.

## Model:

To train the model, we initially implemented a base default linear regression model. Subsequently, in the remaining stages of the training development, we employed the XGBRegressor algorithm. The base XGBRegressor was utilized with default parameters, and to optimize the model, a Hyperparameter Tuning process was conducted. The best parameters for the XGBRegressor were determined to be a learning rate of 0.1, a maximum depth of 3, and the number of estimators set to 200.

## Results:

The metric employed in this model was the Root Mean Squared Error (RMSE). The base linear model yielded an RMSE of 910836531.641, while the base XGBRegressor exhibited an RMSE of 0.153. The optimized XGBRegressor model achieved an improved RMSE of 0.152. Consequently, this optimized model was used in the final predictions using the test dataset. The predictions were submitted to Kaggle, resulting in a score of 0.153.

## Application:

For the application, the optimal model was saved using joblib. Subsequently, a Streamlit web application was created. Users can input values and select feature categories, and the application will display the predicted price of the residential property. The web application also presents relevant data visualizations explaining the choice of features.
## Streamlit Web App: https://ameshousingmodel-akwb7ogsrujq9zanisrzus.streamlit.app/
 
