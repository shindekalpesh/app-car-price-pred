#!/usr/bin/env python
# coding: utf-8

# https://archive.ics.uci.edu/ml/datasets/automobile

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

import streamlit as st

import warnings
warnings.filterwarnings('ignore')

########################################################

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
        "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename,names=headers)

df = df.replace("?",np.nan)

for i in df.columns:
    print(f"In the column {i}, the following are the unique values: \n{df[i].unique()}\n")

avg_normloss = df['normalized-losses'].astype('float').mean()

df['normalized-losses'] = df['normalized-losses'].replace(np.nan,avg_normloss)

miss_col = ["bore",'stroke',"horsepower",'peak-rpm']

for i in miss_col:
    df[i] = df[i].replace(np.nan,df[i].astype('float').mean())

df['num-of-doors'].replace(np.nan,'four',inplace=True)

df.dropna(subset=['price'],inplace=True)

df.reset_index(drop=True,inplace=True)

convert_col = ['bore','stroke','horsepower','normalized-losses','peak-rpm','price']

df[convert_col] = df[convert_col].astype('float')

df['compression-ratio'] = df['compression-ratio'].astype(int)

df['price'] = df['price'].astype(int)

df['horsepower'] = df['horsepower'].astype(int)

df['normalized-losses'] = df['normalized-losses'].astype(int)

df['num-of-cylinders'] = df['num-of-cylinders'].replace({
    'two':'2',
    'three':'3',
    'four':'4',
    'five':'5',
    'six':'6',
    'eight':'8',
    'twelve':'12',
})

df['num-of-doors'] = df['num-of-doors'].replace({'two':2,'four':4})

df['fuel-type'] = df['fuel-type'].replace({'diesel':0,'gas':1})

df['aspiration'] = df['aspiration'].replace({'std':0,'turbo':1})

df['body-style'] = df['body-style'].replace({'convertible':0,'hardtop':1,'hatchback':2,'sedan':3,'wagon':4})

df['drive-wheels'] = df['drive-wheels'].replace({'4wd':0,'fwd':1,'rwd':2})

df['engine-location'] = df['engine-location'].replace({'front':0,'rear':1})

df['engine-type'] = df['engine-type'].replace({'ohcf':0,'ohcv':1,'ohc':2,'l':3,'dohc':4,'rotor':5})

df['fuel-system'] = df['fuel-system'].replace({'mpfi':0,'2bbl':1,'idi':2,'1bbl':3,'spdi':4,'4bbl':5,'mfi':6,'spfi':7})

df['num-of-cylinders'] = df['num-of-cylinders'].astype(int)

df['symboling'] = df['symboling'].astype(float)

cat=[]
num=[]

for i in df.columns:
    if df[i].dtypes == object:
        cat.append(i)
    else:
        num.append(i)
        
print(f"The Categorical Features are as follows: {cat}\n")
print(f"The Numerical Features are as follows: {num}\n")

df = df[['make', 'symboling', 'normalized-losses', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
    'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
    'highway-mpg', 'price']]

df = df.drop(columns=['make', 'normalized-losses'])

X=df.iloc[:,:-1]
y=df.iloc[:,-1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

rfr = RandomForestRegressor()

model = rfr.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(f"The accuracy achieved is {round(r2_score(y_test,y_pred)*100,2)} %.")

result = model.predict([[1,1,0,2,2,2,0,94,171,65.50,52,2823,1,6,152,0,2,3,9,154,5000,19,26]])

print(result)

########################################################



nav = st.sidebar.radio("Navigation Panel",["Home","Prediction"])

if nav == "Home":

    st.title("Welcome to New Car Price Prediction App!")

    st.write("This Web Application made by **Kalpesh Shinde** with **Streamlit**.")

    st.write("This is very initial stage of this application. New update rolling out soon.")

elif nav == "Prediction":
    st.write("**Welcome to the Prediction Page!**")

    symboling = st.select_slider("Safety: ",[-2,-1,0,1,2,3])

    #normalized_losses = st.selectbox("Normalized Loss: ",df['normalized-losses'])#.unique().sort())

    fuel_type = st.radio("Type of fuel: 'Diesel': 0,'Gas': 1",[0,1])

    aspiration = st.radio(label="Turbocharged: 'No': 0, 'Yes': 1",options=[0,1])

    num_of_doors = st.radio("Number of doors: ",[2,4])

    body_style = st.selectbox("Body Style: 0: 'Convertible', 1: 'Hardtop', 2: 'Hatchback', 3: 'Sedan', 4: 'Wagon'",[0,1,2,3,4])

    drive_wheels = st.selectbox("Drive Wheels 0: '4 Wheel Drive', 1: 'Front Wheel Drive',  2: 'Rear Wheel Drive'",[0,1,2])

    engine_location = st.selectbox("Location of engine: 'Front': 0,'Rear': 1",[0,1])

    wheel_base = st.slider("Wheelbase: ",min_value=80.0,max_value=131.0,step=0.02)

    length = st.slider("Length: ",min_value=130.0,max_value=221.0,step=0.02)

    width = st.slider("Width: ",min_value=50.0,max_value=80.0,step=0.02)

    height = st.slider("Height: ",min_value=40.0,max_value=71.0,step=0.02)

    curb_weight = st.slider("Curb Weight: ",min_value=1400.0,max_value=4101.0,step=0.02)

    engine_type = st.selectbox("Type of engine: 'ohcf': 0, 'ohcv': 1, 'ohc': 2, 'l': 3, 'dohc': 4, 'rotor': 5",[0,1,2,3,4,5])

    num_of_cylinders = st.selectbox("Number of cylinders: ",[2,3,4,6,5,8,12])

    engine_size = st.selectbox("Size of engine: ",[i for i in range(50,331)])

    fuel_system = st.selectbox("Fuel system: 'mpfi': 0,'2bbl': 1,'idi': 2,'1bbl': 3,'spdi': 4,'4bbl': 5,'mfi': 6,'spfi': 7",[0,1,2,3,4,5,6,7])

    bore = st.slider("Bore: ",min_value=2.0,max_value=4.5,step=0.02)

    stroke = st.slider("Stroke: ",min_value=2.0,max_value=4.5,step=0.02)

    compression_ratio = st.slider("Compression ratio: ",min_value=7,max_value=23,step=1)

    horsepower = st.slider("Horsepower: ",min_value=40,max_value=270,step=1)

    peak_rpm = st.slider("Peak RPM: ",min_value=4150.0,max_value=6600.0,step=0.1)

    city_mpg = st.slider("City Mileage: ",min_value=10,max_value=50,step=1)

    highway_mpg = st.slider("High Mileage: ",min_value=10,max_value=55,step=1)

    if st.button("Submit"):

        result = model.predict([[symboling,fuel_type,aspiration,num_of_doors,body_style,drive_wheels,
                                engine_location,wheel_base,length,width,height,curb_weight,engine_type,num_of_cylinders,
                                engine_size,fuel_system,bore,stroke,compression_ratio,horsepower,peak_rpm,city_mpg,highway_mpg]])

        st.success(f"The Predicted Price of the car is $ {result}. ")

        st.success(f"The achieved accuracy of the model is {round(r2_score(y_test,y_pred)*100,2)} %.")


########################################################


