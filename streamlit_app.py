'''
Make sure to install streamlit with `conda install -c conda-forge streamlit`.

Run `streamlit hello` to get started!

Streamlit is *V* cool, and it's only going to get cooler (as of February 2021):

https://discuss.streamlit.io/t/override-default-color-palette/9088/2

To run this app, run `streamlit run streamlit_app.py` from inside this directory
'''


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# # PART 1

# st.write('''
# # Welcome To Streamlit!

# In this streamlit app we will cover:

# - Markdown
# - Importing Data
# - Displaying DataFrames
# - Graphing
# - Interactivity with Buttons
# - Mapping
# - Making Predictions with User Input
# ''')

# # PART 2

# st.write(
# '''
# You can use markdown syntax to style your text

# Headers:
# # Main Title 
# ## Sub Title 
# ### header 

# **bold text**

# *italics*

# Ordered List

# 1. Apples 
# 2. Oranges 
# 3. Bananas 

# [This is a link](https://docs.streamlit.io/en/stable/getting_started.html)


# '''
# )

# # PART 3

# st.write(
# '''
# ## Seattle Home Prices
# We can import data into our streamlit app using pandas read_csv then display the resulting dataframe with st.dataframe()

# ''')

# data = pd.read_csv('SeattleHomePrices.csv')
# data = data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
# st.dataframe(data)

# # PART 4

# st.write(
# '''
# ### Graphing and Buttons
# Lets graph some of our data with matplotlib. We can also add buttons to add interactivity to our app.
# '''
# )

# fig, ax = plt.subplots()

# ax.hist(data['PRICE'])
# ax.set_title('Distribution of House Prices in $100,000s')

# show_graph = st.checkbox('Show Graph', value=True)

# if show_graph:
#     st.pyplot(fig)

# # PART 5
    
# st.write(
# '''
# ### Mapping and Filtering Our Data
# We can also use streamlits built in mapping functionality.
# We can use a slider to filter for houses within a particular price range as well.
# '''
# )

# price_input = st.slider('House Price Filter', int(data['PRICE'].min()), int(data['PRICE'].max()), 100000 )

# price_filter = data['PRICE'] < price_input
# st.map(data.loc[price_filter, ['lat', 'lon']])

# # PART 6

# st.write(
# '''
# ## Train a linear Regression Model
# Create a model to predict house price from sqft and number of beds
# '''
# ) 

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# clean_data = data.dropna(subset=['PRICE', 'SQUARE FEET', 'BEDS'])

# X = clean_data[['SQUARE FEET', 'BEDS']]
# y = clean_data['PRICE']

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# # Notice the R^2 value is changing. This file is run on every update!
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# lr = LinearRegression()

# lr.fit(X_train, y_train)
# st.write(f'R2: {lr.score(X_test, y_test):.2f}')

# # PART 7

# st.write(
# '''
# ## Make predictions with the trained model from user input
# '''
# )

# sqrft = st.number_input('Square Footage of House', value=2000)
# beds = st.number_input('Number of Bedrooms', value=3)

# input_data = pd.DataFrame({'sqrft': [sqrft], 'beds': [beds]})
# pred = lr.predict(input_data)[0]
# st.write(
# f'Predicted Sale Price of House: ${pred:.2f}'
# )

