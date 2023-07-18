import pandas as pd
import datetime
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
from functools import wraps
from streamlit_extras.no_default_selectbox import selectbox
import pyautogui


def read_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = func(*args, **kwargs)
        df = pd.read_csv(s)
        df["Date"]=pd.to_datetime(df["Date"],dayfirst=True) #, format="%d-%m-%Y")
        df["Date"]=df["Date"].dt.date
        df['ExitRate'].replace('nil', 0, inplace=True)
        df['ExitRate'] = df['ExitRate'].astype(float)   
        # Time into categoty
        df['EntryAt'] = pd.to_datetime(df['EntryAt'], format="%H.%M.%S")
        df['Time'] = df['EntryAt'].apply(lambda time: "09.00-11.00" if 9  <= time.hour <= 10
                                          else "11.00-01.00" if 11 <= time.hour <= 12 
                                          else "01.00-03.00" if 13 <= time.hour <= 14  else "03.00-03.30")
        def calculate_gross_pl(row):
            if row['Signal'] == 'BUY':
                if row['Closure'] == 'SL':
                    return (row['SL'] - row['Rate']) * row['QTY']
                elif row['Closure'] == 'TGT':
                    return (row['TGT'] - row['Rate']) * row['QTY']
                elif row['Closure'] == 'EXIT':
                    return (row['ExitRate'] - row['Rate']) * row['QTY']
            elif row['Signal'] == 'SELL':
                if row['Closure'] == 'SL':
                    return (row['Rate'] - row['SL']) * row['QTY'] 
                elif row['Closure'] == 'TGT':
                    return (row['Rate'] - row['TGT']) * row['QTY']
                elif row['Closure'] == 'EXIT':
                    return (row['Rate'] - row['ExitRate']) * row['QTY']
            return 0
        df['Gross_PL'] = df.apply(calculate_gross_pl, axis=1)
        
        df['Result'] = np.where(df['Gross_PL'] >= 0, "Profit", "Loss")
        

        # creating columns for profit Amount & loss Amount
        df['P_Amount']=np.where(df['Gross_PL']>=0,df['Gross_PL'],0)
        df['L_Amount']=np.where(df['Gross_PL']<0,df['Gross_PL'],0)


        df['Closure1']=np.where((df['Closure']=='TGT'),'TGT',
                   np.where((df['Closure']=='SL') & (df['Gross_PL'] >= 0),'TSL',
                   np.where((df['Closure']=='SL') & (df['Gross_PL'] < 0),'SL',         
                   np.where((df['Closure']=='EXIT') & (df['Gross_PL'] >= 0),'EXIT PROFIT',
                   'EXIT LOSS'))))

        
        return df
    return wrapper

def add_sidebar(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		#Algo Number:
		options, label, = func(*args, **kwargs)
		df_algo = st.sidebar.selectbox(options= options,
                               label=label)
		return df_algo
	return wrapper

def add_sidebar_1(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		#Algo Number:
		options, label = func(*args, **kwargs)
		df_algo = selectbox(options= options,
                               label=label)
		return df_algo
	return wrapper



def add_sidebar1(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		#Algo Number:
		options, label = func(*args, **kwargs)
		df_algo = st.multiselect(options= options,
                               label=label)
		return df_algo
	return wrapper



def add_sidebar_date(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		#Algo Number:
		label, year, month, day = func(*args, **kwargs)
		df_algo = st.sidebar.date_input(label, datetime.date(year,month,day))
		return df_algo
	return wrapper


def raise_error(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		msg = func(*args, **kwargs)
		st.error(msg)
		return msg
	return wrapper


def write_dataframe(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		df = func(*args, **kwargs)
		st.write(df)
		return df
	return wrapper

@add_sidebar
def algo_sidebar(data, col, label):
	return data[col].unique(), label
#df[df["Zone"].isin(df_Zone)]["District"].unique()

@add_sidebar_1
def algo_sidebar_1(data, col, label):
	return data[col].unique(), label


@add_sidebar1
def algo_sidebar1(data, col, label):
	return data[col].unique(), label

@add_sidebar_date
def date_sidebar(label, year, month, day):
	return label, year, month, day

@read_data
def get_data():
	return "Algo.csv"

@raise_error
def raise_date_error():
	return "Error: End date should be after the start date."

@raise_error
def raise_date_error1():
	return "Data not available on the selected date."

@write_dataframe
def write_data(data):
	return data

def start_app():
    df = get_data()
    algo_name = algo_sidebar(df, "AlgoName", "Select AlgoName")
    algo_filter = df['AlgoName'] == algo_name
    df = df[algo_filter].copy()
    start_date = date_sidebar("Start date", 2023, 1, 1)
    end_date = date_sidebar("End date", 2023, 7, 6)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    with st.sidebar:
        segment = algo_sidebar_1(df, "Segment", "Select segment")
    st.markdown("<h2 style='text-align: center; color: Blue;'>MAV_Algo Performance</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("")
        home= st.button("### **Home**")
        
    with col2:
        symbol_filter=df[df["Segment"] == segment]["Symbol"].unique()
        symbol_filter.sort()
        symbol = selectbox ("Select Symbol", symbol_filter) 
    with col3:
        top = selectbox('Top Players', [3, 5, 10])
    
   
    def autopct_format(values): 
        def my_format(pct):   
            total = sum(values) 
            val = int(round(pct*total/100.0)) 
            return '{:.1f}%\n({v:d})'.format(pct, v=val) 
        return my_format
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    def add_plot1():
        plt.subplot(1,3,1)
        counts=df['Result'].value_counts()
        colors={'Loss': 'red',
                'Profit': 'green'}
        plt.pie(counts.values, labels=counts.index, autopct=autopct_format(counts),textprops={'fontsize': 16},wedgeprops= {'linewidth': 2, 'edgecolor': 'white'},colors=[colors[key] for key in counts.index])  
        
    def add_plot2():    
        plt.subplot(1,3,2)
        values=['TGT', 'TSL', 'EXIT PROFIT']
        filter=df['Closure1'].isin(values)
        data=df[filter]
        counts=data['Closure1'].value_counts()
        colors={'EXIT PROFIT': 'green',
                'TGT': 'green',
                'TSL': 'green'}
        plt.pie(counts.values, labels=counts.index, autopct=autopct_format(counts),textprops={'fontsize': 16},wedgeprops= {'linewidth': 2, 'edgecolor': 'white'},colors=[colors[key] for key in counts.index])  
        
    def add_plot3():     
        plt.subplot(1,3,3)
        values = ['SL', 'EXIT LOSS']
        filter=df['Closure1'].isin(values)
        data=df[filter]
        counts=data['Closure1'].value_counts()
        colors={'EXIT LOSS': 'red',
                'SL': 'red'}  
        plt.pie(counts.values, labels=counts.index, autopct=autopct_format(counts),textprops={'fontsize': 16},wedgeprops= {'linewidth': 2, 'edgecolor': 'white'},colors=[colors[key] for key in counts.index])  
        
    def plot_pl():  
        PL=df.groupby("Date")[['P_Amount','L_Amount','Gross_PL']].sum().reset_index()
        fig = px.line(PL, x='Date', y=[ 'P_Amount','L_Amount','Gross_PL'], color_discrete_map={"Gross_PL": "blue", "P_Amount": "green","L_Amount": "red"})
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig,use_container_width=True)        
       
        
    def button1():    
        csv = convert_df(t)
        st.download_button(
        label="Download Data",
        data=csv,
        file_name=f"PL_Count of{symbol}({segment}).csv",
        mime='text/csv',
        )
    def button2():
        csv = convert_df(t)
        st.download_button(
        label="Download Data",
        data=csv,
        file_name=f"PL_Amount of{symbol}({segment}).csv",
        mime='text/csv',
        )        
            
    def dataframe():
        write_data(t.round(2))             

    if home:
        image = Image.open('download.png')
        st.image(image, caption='Algo Analysis',width=700) 
        pyautogui.hotkey("ctrl","F5")
        
        
    elif end_date < start_date:
        raise_date_error()
        
    elif df.shape[0]==0:
        raise_date_error1()
        
        
    elif algo_name  and segment and symbol:
        st.subheader(f"Profit & Loss Count of {symbol} ({segment})")
        filter = (df['AlgoName'] == algo_name) & (df['Segment'] == segment) & (df['Symbol'] == symbol)
        df = df[filter]
        # -----------plotting pie plots----------
        fig = plt.figure(figsize=(15, 12), dpi=1600)
        add_plot1()
        add_plot2()
        add_plot3()
        st.pyplot(fig)
        t=df.groupby(['Date','Segment','Symbol','Result','Closure1'])['Gross_PL'].agg(Count='count',Total='sum')
        #dataframe()
        button1()        
        #-------------line plot---------------
        st.subheader(f"Profit & Loss Amount of {symbol} ({segment})")
        plot_pl()
        t=df.groupby(['Date','Segment','Symbol'])[['P_Amount','L_Amount','Gross_PL']].agg({'P_Amount': ['sum'], 'L_Amount': ['sum'], 'Gross_PL':['count','sum']})
        #dataframe()
        button2()
        
    elif algo_name  and segment and top:
        filter = (df['AlgoName'] == algo_name) & (df['Segment'] == segment)
        df = df[filter]
        pl=df.groupby(['Symbol'])['Gross_PL'].sum().sort_values(ascending=False).reset_index()
        #st.dataframe(pl) 
              
        if pl.shape[0]<=3:
            st.info(f"{segment}  has only {pl.shape[0]} stocks . Can't see Top {top} Gainers and Losers") 
            t=pl.head(top)
            pl["Color"] = np.where(pl["Gross_PL"]>=0, "green", "red")
            fig = px.bar(t, y="Gross_PL",x="Symbol")            
            fig.update_traces(marker_color = pl["Color"])
            st.plotly_chart(fig)
            #dataframe()
            button2()
        else:
            col1,col2= st.columns(2)
            with col1:
                st.subheader(f"Top {top} Gainers of {segment}")
                t=pl.head(top)
                fig =px.bar(t, y="Gross_PL",x="Symbol")
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                fig.update_traces(marker_color='green')
                st.plotly_chart(fig,use_container_width=True)
                #dataframe()
                button2()
            with col2:
                st.subheader(f"Top {top} Losers of {segment}")
                t=pl.tail(top)
                fig =px.bar(t, y='Gross_PL',x="Symbol")
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                fig.update_traces(marker_color='red')
                st.plotly_chart(fig,use_container_width=True)
                #dataframe() 
                button2()

    elif algo_name  and segment:
        st.subheader(f"Profit & Loss Count and Amount of {segment}")
        filter = (df['AlgoName'] == algo_name) & (df['Segment'] == segment)
        df = df[filter]
        t=df.groupby(['Date','Symbol','Closure1'])[['P_Amount','L_Amount','Gross_PL']].agg({'P_Amount': ['sum'], 'L_Amount': ['sum'], 'Gross_PL':['count','sum']})
        #dataframe()
        st.dataframe(t.round(2),width=1000)
        button2()

    else:
        image = Image.open('download.png')
        st.image(image, caption='Algo Analysis',width=600) 
        
        
        
   
    
    
            
if __name__ == "__main__":
    start_app()