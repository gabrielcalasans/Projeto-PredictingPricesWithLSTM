import streamlit as st
import pandas as pd
import streamlit_nested_layout
import datetime
import classes as lstm
import time


st.set_page_config(layout = 'wide')

def msgLoading():
    msg = st.toast('**Loading :clock1:**')
    for i in range(1, 4):
        time.sleep(0.5)
        msg.toast('**Loading :clock4:**')
        time.sleep(0.5)
        msg.toast('**Loading :clock8:**')
        time.sleep(0.5)
        msg.toast('**Loading :clock10:**')


with st.sidebar:
    with st.form("LSTM Prediction"):
        ticker = st.text_input("Ticker :money_with_wings:")
        colunas = st.columns(2)
        with colunas[0]:
            start = st.date_input("Start Date :calendar:", max_value=datetime.date.today())
        
        with colunas[1]:
            end = st.date_input("End Date :calendar:", max_value=datetime.date.today())
        
        with st.expander("**Advanced Options** :gear:"):            
            trainPoint = st.slider("Split Point :heavy_division_sign:", value = 0.65, min_value = 0.01, max_value = 0.99, step = 0.01)
            units = st.number_input("Units :brain:", min_value=1, value = 150, step=1)
            epochs = st.number_input("Epochs :clock2:", min_value=1, value = 20, step=1)
            valSplit = st.number_input("Validation Split :white_check_mark:", min_value=0.01, value = 0.2, step=0.01)         
            batchSize = st.number_input("Batch Size :scales:", min_value=1, value = 32, step=1)        
        
        simular = st.form_submit_button("**Predict** :part_alternation_mark:")

if simular:
    st.header("**Predictions with LSTM (Long Short Term Memory)**")    
    try:
        try:            
            stock = lstm.StockData(ticker, startDate = start, endDate = end, trainPoint = trainPoint)
            msgLoading()
            predicao = lstm.PredictLSTM(stock, epochs = epochs, units = units, batchSize = batchSize, valSplit = valSplit)
            st.plotly_chart(predicao.interactivePlot())
            st.write("The prices are scaleds")
            with st.expander("Data"):
                st.dataframe(predicao.predicted)
        except:
            st.error("**Error on simulating, pleache check the inputs**")
            print(stock.baseStockInfo)
    except:
        print('Error')


else:
    st.warning("**Set inputs to continue...**")