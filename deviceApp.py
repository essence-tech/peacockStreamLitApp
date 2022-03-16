import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.offsetbox import AnchoredText
from datetime import date, datetime
import time

import sklearn
from sklearn.preprocessing import *
from sklearn.neighbors import *
from sklearn.model_selection import *
from sklearn.inspection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.gaussian_process import *
from sklearn.gaussian_process.kernels import *
from sklearn.svm import *
from sklearn.linear_model import *

import joblib

import optuna

from sklearn import set_config

set_config(display="diagram")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

##---------------------------------------------------------------------------------------------------------------------
# User Defined Functions
# Define the function to return the SMAPE value
def symmetric_mean_absolute_percentage_error(actual, predicted) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return np.round(
        np.mean(
            np.abs(predicted - actual) /
            ((np.abs(predicted) + np.abs(actual)) / 2)
        ) * 100, 2
    )

def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )
    return selection
##---------------------------------------------------------------------------------------------------------------------

st.title('Monthly Active Accounts (MAA) Analysis')
st.header('add something here')
st.write("")

DATA_URL1 = (r'https://raw.githubusercontent.com/essence-tech/PeacockVision/main/20220310_Dev/20220124_accountFlowNetAddCheck.csv')
DATA_URL2 = (r'https://raw.githubusercontent.com/essence-tech/PeacockVision/main/20220310_Dev/20220310_ResponseVariableFormated.csv')

@st.cache
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

# Load the data into the dataframe
acctDF = load_data(DATA_URL1)
datDF = load_data(DATA_URL2)

##---------------------------------------------------------------------------------------------------------------------

# Pre-process AccountFlow data
acctDF = acctDF[(acctDF['Date']>='2021-01-01') & (acctDF['Date']<='2021-12-31')].reset_index(drop=True)
acctDF['Date'] = pd.to_datetime(acctDF['Date'])
acctDF = acctDF.sort_values(by=['Date'])
acctDF.index = acctDF.pop('Date')

# Create dataframe for each account entitlement
freeDF = acctDF[(acctDF['Account_Entitlement']=='Free') & (acctDF['paying_account_flag']=="NonPaying")]
freeDF = freeDF.resample("D").sum()

premiumPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium') & (acctDF['paying_account_flag']=="Paying")]
premiumPayDF = premiumPayDF.resample("D").sum()

premiumNonPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium') & (acctDF['paying_account_flag']=="NonPaying")]
premiumNonPayDF = premiumNonPayDF.resample("D").sum()

premiumPlusPayDF = acctDF[(acctDF['Account_Entitlement']=='Premium+') & (acctDF['paying_account_flag']=="Paying")]
premiumPlusPayDF = premiumPlusPayDF.resample("D").sum()

##---------------------------------------------------------------------------------------------------------------------

# Pre-process response variable data
datDF = datDF[(datDF['Date']>='2021-01-01') & (datDF['Date']<='2021-12-31')].reset_index(drop=True)
datDF['Date'] = pd.to_datetime(datDF['Date'])
datDF = datDF.sort_values(by=['Date'])
datDF.index = datDF.pop('Date')

##---------------------------------------------------------------------------------------------------------------------


acctLST = ['Free Non-Pay']
#acctLST = ['Free Non-Pay', 'Premium Pay', 'Premium Non-Pay', 'Premium Plus Pay']

option1 = st.sidebar.selectbox(
    'What account entitlement would you like to use?',
    acctLST)

datasetDF = datDF.copy()

if option1 == 'Free Non-Pay':
    datasetDF['Inflow'] = freeDF['Inflows']
    datasetDF['Outflow'] = freeDF['Outflows']

elif option1 == 'Premium Pay':
    datasetDF['Inflow'] = premiumPayDF['Inflows']
    datasetDF['Outflow'] = premiumPayDF['Outflows']

elif option1 == 'Premium Non-Pay':
    datasetDF['Inflow'] = premiumNonPayDF['Inflows']
    datasetDF['Outflow'] = premiumNonPayDF['Outflows']
else:
    datasetDF['Inflow'] = premiumPlusPayDF['Inflows']
    datasetDF['Outflow'] = premiumPlusPayDF['Outflows']

datasetCOLS = list(datasetDF.columns)

responseLST = ['Spend']

option2 = st.sidebar.selectbox(
    'What response variable would you like to use?',
    responseLST)

# Filter out columns

if option2 == 'Spend':
    spendCOLS = []
    for i in range(len(datasetCOLS)):
        if datasetCOLS[i].split("_")[0] == 'Impressions':
            pass
        else:
            spendCOLS.append(datasetCOLS[i])
else:
    spendCOLS = []
    for i in range(len(datasetCOLS)):
        if datasetCOLS[i].split("_")[0] == 'Spend':
            pass
        else:
            spendCOLS.append(datasetCOLS[i])

dataset = datasetDF[spendCOLS].copy()
dataset = dataset.reset_index(drop=False)
dataset = dataset[(dataset['Date']>='2021-01-01') & (dataset['Date']<='2021-09-30')]

dataset['Month'] = [dataset['Date'][i].month for i in range(len(dataset))]
dataset['DOW'] = [dataset['Date'][i].dayofweek for i in range(len(dataset))]
dataset['DOY'] = [dataset['Date'][i].dayofyear for i in range(len(dataset))]
dataset['Quarter'] = [dataset['Date'][i].quarter for i in range(len(dataset))]
dataset['fracDOY'] = [dataset['Date'][i].dayofyear/365 for i in range(len(dataset))]

date_time = dataset.pop('Date')

##---------------------------------------------------------------------------------------------------------------------

# Split Dataframe
training_set = dataset.copy()
training_set = training_set + 1

target_Inflow = training_set.pop('Inflow')

target_Outflow = training_set.pop('Outflow')
target_Outflow = target_Outflow*-1

training_set = training_set.values
target_Inflow = target_Inflow.values
target_Outflow = target_Outflow.values

##---------------------------------------------------------------------------------------------------------------------

if option2 == 'Spend':
    training_set_columns = ['Spend_Audio_Podcast', 'Spend_Audio_Streaming',
                            'Spend_Audio_Terrestrial Radio', 'Spend_Device_Amazon',
                            'Spend_Device_LG', 'Spend_Device_Playstation', 'Spend_Device_Roku',
                            'Spend_Device_Samsung', 'Spend_Device_Vizio', 'Spend_Device_Xbox',
                            'Spend_Digital_SiteServed_SiteDirect', 'Spend_OutOfHome_None',
                            'Spend_SIF_Addressable TV', 'Spend_SIF_Bigfoot/Spin', 'Spend_SIF_DDL',
                            'Spend_SIF_IPG', 'Spend_SIF_PDTV', 'Spend_Social_Facebook',
                            'Spend_Video_Addressable', 'Spend_Video_Broadcast', 'Spend_Video_Cable',
                            'Spend_Video_Cinema', 'Spend_Video_FEP', 'Month', 'DOW',
                            'DOY', 'Quarter', 'fracDOY']
else:
    training_set_columns = ['Impressions_Audio_Podcast', 'Impressions_Audio_Streaming',
                            'Impressions_Audio_Terrestrial Radio', 'Impressions_Device_Amazon',
                            'Impressions_Device_LG', 'Impressions_Device_Playstation', 'Impressions_Device_Roku',
                            'Impressions_Device_Samsung', 'Impressions_Device_Vizio', 'Impressions_Device_Xbox',
                            'Impressions_Digital_SiteServed_SiteDirect', 'Impressions_OutOfHome_None',
                            'Impressions_SIF_Addressable TV', 'Impressions_SIF_Bigfoot/Spin', 'Impressions_SIF_DDL',
                            'Impressions_SIF_IPG', 'Impressions_SIF_PDTV', 'Impressions_Social_Facebook',
                            'Impressions_Video_Addressable', 'Impressions_Video_Broadcast', 'Impressions_Video_Cable',
                            'Impressions_Video_Cinema', 'Impressions_Video_FEP', 'Month', 'DOW',
                            'DOY', 'Quarter', 'fracDOY']


##---------------------------------------------------------------------------------------------------------------------

X_scaler = PowerTransformer(method='box-cox')

Y_Inflow_scaler = PowerTransformer(method='box-cox')

Y_Outflow_scaler = PowerTransformer(method='box-cox')

##---------------------------------------------------------------------------------------------------------------------

X_data = X_scaler.fit_transform(training_set)

Y_Inflow_data = Y_Inflow_scaler.fit_transform(target_Inflow.reshape(-1, 1))

Y_Outflow_data = Y_Outflow_scaler.fit_transform(target_Outflow.reshape(-1, 1))

##---------------------------------------------------------------------------------------------------------------------

# Load Model Files
if option1 == 'Free Non-Pay':
    if option2 == 'Spend':
        automlInflow = joblib.load('venv/models/gprFreeInflow20220313.joblib')
        automlOutflow = joblib.load('venv/models/gprFreeOutflow20220313.joblib')

elif option1 == 'Premium Pay':
    pass

elif option1 == 'Premium Non-Pay':
    pass
else:
    pass

##---------------------------------------------------------------------------------------------------------------------

# Model predictions
y_inflow_predictions = automlInflow.predict(X_data)
y_outflow_predictions = automlOutflow.predict(X_data)

y_inflow_predictions_inv = Y_Inflow_scaler.inverse_transform(y_inflow_predictions.reshape(-1, 1))
y_otflow_predictions_inv = Y_Outflow_scaler.inverse_transform(y_outflow_predictions.reshape(-1, 1))

y_inflow_inv = Y_Inflow_scaler.inverse_transform(Y_Inflow_data.reshape(-1, 1))
y_outflow_inv = Y_Outflow_scaler.inverse_transform(Y_Outflow_data.reshape(-1, 1))

resDataDF = pd.DataFrame(y_inflow_inv, columns=['y_inflow_inv'])
resDataDF['y_inflow_predictions_inv'] = y_inflow_predictions_inv

resDataDF['y_outflow_inv'] = y_outflow_inv
resDataDF['y_otflow_predictions_inv'] = y_otflow_predictions_inv

resDataDF['NetAdds_True'] = resDataDF['y_inflow_inv'] - resDataDF['y_outflow_inv']
resDataDF['NetAdds_Pred'] = resDataDF['y_inflow_predictions_inv'] - resDataDF['y_otflow_predictions_inv']

resDataDF['sAPE_IN'] = np.abs(resDataDF['y_inflow_predictions_inv']-resDataDF['y_inflow_inv'])/((np.abs(resDataDF['y_inflow_predictions_inv']) + np.abs(resDataDF['y_inflow_inv'])/2))*100
resDataDF['sAPE_OT'] = np.abs(resDataDF['y_otflow_predictions_inv']-resDataDF['y_outflow_inv'])/((np.abs(resDataDF['y_otflow_predictions_inv']) + np.abs(resDataDF['y_outflow_inv'])/2))*100
resDataDF['sAPE_NA'] = np.abs(resDataDF['NetAdds_Pred']-resDataDF['NetAdds_True'])/((np.abs(resDataDF['NetAdds_Pred']) + np.abs(resDataDF['NetAdds_True'])/2))*100

##---------------------------------------------------------------------------------------------------------------------

def plot_predictions(df, var0, var1, var2, var3, var4):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    if (var0=='Inflow') | (var0=='NetAdds'):
        plt.plot(df[var1], 'o-', label='True')
        plt.plot(df[var2], '-', label='Pred')
    else:
        plt.plot(df[var1] * -1, 'o-', label='True')
        plt.plot(df[var2] * -1, '-', label='Pred')
    plt.xlabel('Index', fontsize=16)
    plt.ylabel('{}'.format(var0), fontsize=16)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.title('{} Tier sMAPE: {:.2f}%'.format(var4 ,np.mean(df[var3])), fontsize=20)
    plt.grid(True)
    st.pyplot(fig)

if option2 == 'Spend':
    with st.expander("Plot background data?", expanded=False):
        # Plot data
        st.subheader('Inflows')
        var0 = 'Inflow'
        var1 = 'y_inflow_inv'
        var2 = 'y_inflow_predictions_inv'
        var3 = 'sAPE_IN'
        var4 = option1
        plot_predictions(resDataDF, var0, var1, var2, var3, var4)

        st.subheader('Outflows')
        var0 = 'Outflow'
        var1 = 'y_outflow_inv'
        var2 = 'y_otflow_predictions_inv'
        var3 = 'sAPE_OT'
        var4 = option1
        plot_predictions(resDataDF, var0, var1, var2, var3, var4)

        st.subheader('NetAdds')
        var0 = 'NetAdds'
        var1 = 'NetAdds_True'
        var2 = 'NetAdds_Pred'
        var3 = 'sAPE_NA'
        var4 = option1
        plot_predictions(resDataDF, var0, var1, var2, var3, var4)
        st.write("")
st.write("")
##---------------------------------------------------------------------------------------------------------------------

def define_budget(trial, inputVal, inputDat, budget, inputNumCol):
    inputValArr = np.zeros((1,len(inputVal)))
    # Month
    inputValArr[:,23] = pd.to_datetime(inputDat).month
    # DOW
    inputValArr[:,24] = pd.to_datetime(inputDat).dayofweek
    # DOY
    inputValArr[:,25] = pd.to_datetime(inputDat).dayofyear
    # Quarter
    inputValArr[:,26] = pd.to_datetime(inputDat).quarter
    # fracDOY
    inputValArr[:,27] = pd.to_datetime(inputDat).dayofyear/365

    sugVarArr = np.zeros((1,23))
    for i in range(0,23):
        sugVarArr[:,i] = trial.suggest_categorical(inputNumCol[i], [True, False])
    for i in range(0,23):
        if sugVarArr[:,i] == True:
            inputValArr[:,i] = trial.suggest_float('x_'+str(i), inputVal['MinVals'][i], inputVal['MaxVals'][i])
        else:
            inputValArr[:,i] = 1.0

    budgetTMP = np.sum(inputValArr[:,0:23])

    return (budgetTMP-budget)**2, inputValArr

def eval_model(inputValArr):
    X_arr = X_scaler.transform(inputValArr)
    I_arr = Y_Inflow_scaler.inverse_transform(automlInflow.predict(X_arr).reshape(-1, 1))
    O_arr = Y_Outflow_scaler.inverse_transform(automlOutflow.predict(X_arr).reshape(-1, 1))
    N_arr = I_arr[0][0] - O_arr[0][0]
    return N_arr

def objective(trial):
    error, inputValArr = define_budget(trial, inputVal, inputDat, budget, inputNumCol)
    N_arr = eval_model(inputValArr)
    return error, N_arr

# Launch training & forecast
if option2 == 'Spend':
    if st.checkbox("Launch basic spend optimizer", value=False):
        form = st.form(key="annotation")
        with form:
            cols = st.columns((1, 1))
            inputDat = cols[0].date_input(
                        "Select start date",
                        date(2021, 1, 1),
                        min_value=datetime.strptime("2021-01-01", "%Y-%m-%d"),
                        max_value=datetime.strptime("2023-01-01", "%Y-%m-%d"),
                    )
            time_frame = cols[1].selectbox(
                        "Select prediction interval", ("daily")
                    )
            budget = st.number_input('Insert a number for the budget', value=1000000)
            submitted = st.form_submit_button(label="Submit")

        if submitted:
            medianValsLST = []
            minValsLST = []
            maxalsLST = []
            for i in range(0, 28):
                medianValsLST.append(np.round_(np.mean(training_set[:, i]), 0))
                minValsLST.append(np.round_(np.min(training_set[:, i]), 0))
                maxalsLST.append(np.round_(np.max(training_set[:, i]), 0))

            inputVal = pd.DataFrame({'Parameters': training_set_columns,
                                     'MeanVals': medianValsLST,
                                     'MinVals': minValsLST,
                                     'MaxVals': maxalsLST})

            inputDateCol = ['Month', 'DOW', 'DOY', 'Quarter', 'fracDOY']

            if option2 == 'Spend':
                inputNumCol = ['Spend_Audio_Podcast', 'Spend_Audio_Streaming',
                               'Spend_Audio_Terrestrial Radio', 'Spend_Device_Amazon',
                               'Spend_Device_LG', 'Spend_Device_Playstation', 'Spend_Device_Roku',
                               'Spend_Device_Samsung', 'Spend_Device_Vizio', 'Spend_Device_Xbox',
                               'Spend_Digital_SiteServed_SiteDirect', 'Spend_OutOfHome_None',
                               'Spend_SIF_Addressable TV', 'Spend_SIF_Bigfoot/Spin', 'Spend_SIF_DDL',
                               'Spend_SIF_IPG', 'Spend_SIF_PDTV', 'Spend_Social_Facebook',
                               'Spend_Video_Addressable', 'Spend_Video_Broadcast', 'Spend_Video_Cable',
                               'Spend_Video_Cinema', 'Spend_Video_FEP']
            else:
                inputNumCol = ['Impressions_Audio_Podcast', 'Impressions_Audio_Streaming',
                               'Impressions_Audio_Terrestrial Radio', 'Impressions_Device_Amazon',
                               'Impressions_Device_LG', 'Impressions_Device_Playstation', 'Impressions_Device_Roku',
                               'Impressions_Device_Samsung', 'Impressions_Device_Vizio', 'Impressions_Device_Xbox',
                               'Impressions_Digital_SiteServed_SiteDirect', 'Impressions_OutOfHome_None',
                               'Impressions_SIF_Addressable TV', 'Impressions_SIF_Bigfoot/Spin', 'Impressions_SIF_DDL',
                               'Impressions_SIF_IPG', 'Impressions_SIF_PDTV', 'Impressions_Social_Facebook',
                               'Impressions_Video_Addressable', 'Impressions_Video_Broadcast', 'Impressions_Video_Cable',
                               'Impressions_Video_Cinema', 'Impressions_Video_FEP']

            optuna.logging.set_verbosity(verbosity=0)
            study = optuna.create_study(directions=["minimize", "maximize"])
            study.optimize(objective, n_trials=1000, show_progress_bar=False)
            st.success("Success! Your campaign was optimized.")

            ##--------------------------------------------------------------------------------------------------------------

            t = study.trials_dataframe()
            t = t.sort_values(by=['values_0', 'values_1'], ascending=[True, False]).reset_index(drop=True)
            t = t[t['values_1']>0].reset_index(drop=True)

            disDF = pd.DataFrame({'Point': ["Point-1", "Point-2", "Point-3", "Point-4", "Point-5"],
                                  'MAA': np.round_(t['values_1'][:5], 0)})

            ##--------------------------------------------------------------------------------------------------------------

            annotations = ["Pt1", "Pt2", "Pt3", "Pt4", "Pt5"]
            st.write("")
            st.subheader("Pareto-optimal")
            st.write("Pareto-optimal solutions can be joined by line or surface. The five labeled points form a boundary called Pareto-optimal front")
            with st.container():
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)
                plt.plot(t['values_0'], np.abs(t['values_1']), 'o', label="feasible objective space")
                plt.plot(t['values_0'][:5], t['values_1'][:5], 'o', label='Pareto-optimal front')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Objective 1 (Minimized)', fontsize=8)
                plt.ylabel('Objective 2 (Maxminize)', fontsize=8)
                plt.tick_params(axis='x', which='major', labelsize=8)
                plt.tick_params(axis='y', which='major', labelsize=8)
                plt.legend(fontsize=8)
                plt.grid(True)
                st.pyplot(fig)

            ##--------------------------------------------------------------------------------------------------------------

            tVals = t.copy()
            tVals = tVals[tVals['state'] == 'COMPLETE']
            tVals = tVals.drop(columns=['values_0', 'values_1',
                                        'datetime_start', 'datetime_complete', 'duration',
                                        'system_attrs_nsga2:generation', 'system_attrs_nsga2:parents', 'state'])
            tVals = tVals.reset_index(drop=False)
            tValsIndex = tVals.pop('index')
            tVals = tVals.fillna(0)

            tValsCOLSFLOAT = []
            tValsCOLSBOOL = []
            for i in range(0, len(inputVal) - len(inputDateCol)):
                tValsCOLSBOOL.append("params_" + inputVal['Parameters'][i])
                tValsCOLSFLOAT.append("params_x_" + str(i))

            for i in range(len(tValsCOLSFLOAT)):
                converted_value = []
                for j in range(len(tVals)):
                    converted_value.append((tVals[tValsCOLSFLOAT[i]][j].tolist()) * (tVals[tValsCOLSBOOL[i]][j]))
                tVals['m_' + tValsCOLSBOOL[i]] = converted_value

            tValBudget = tVals.copy()
            tValBudget = tValBudget.drop(columns=tValsCOLSBOOL)
            tValBudget = tValBudget.drop(columns=tValsCOLSFLOAT)

            tValBudgetCOLS = list(tValBudget.columns)

            estBudgetLST = []
            for i in range(len(tValBudget)):
                sumROW = 0.0
                for j in range(1, len(tValBudgetCOLS)):
                    sumROW += tValBudget[tValBudgetCOLS[j]][i]
                estBudgetLST.append(sumROW)

            tValBudget['estBudget'] = estBudgetLST
            tValBudget['estBudgetError'] = np.abs(tValBudget['estBudget'] - budget) / budget * 100

            tValBudget = tValBudget.sort_values(by=['estBudgetError'])

            tValBudgetEst1 = tValBudget[tValBudget['estBudgetError'] <= 1].reset_index(drop=True)

            ##--------------------------------------------------------------------------------------------------------------

            st.write("Top five optimizations")
            with st.container():
                col1, col2, col3  = st.columns(3)
                col1.text("Trial")
                col1.text(t['number'][0])
                col1.text(t['number'][1])
                col1.text(t['number'][2])
                col1.text(t['number'][3])
                col1.text(t['number'][4])
                col2.text("MAA")
                col2.text('{:,.0f}'.format(t['values_1'][0]))
                col2.text('{:,.0f}'.format(t['values_1'][1]))
                col2.text('{:,.0f}'.format(t['values_1'][2]))
                col2.text('{:,.0f}'.format(t['values_1'][3]))
                col2.text('{:,.0f}'.format(t['values_1'][4]))
                col3.text("Optimized Budget")
                col3.text('{:,.0f}'.format(tValBudgetEst1['estBudget'][0]))
                col3.text('{:,.0f}'.format(tValBudgetEst1['estBudget'][1]))
                col3.text('{:,.0f}'.format(tValBudgetEst1['estBudget'][2]))
                col3.text('{:,.0f}'.format(tValBudgetEst1['estBudget'][3]))
                col3.text('{:,.0f}'.format(tValBudgetEst1['estBudget'][4]))

            st.write("")

            ##--------------------------------------------------------------------------------------------------------------

            channelLST = []
            subChannelLST = []
            for i in range(1, len(tValBudgetCOLS)):
                if option2 == 'Spend':
                    tmp0 = tValBudgetCOLS[i][15:].split("_")
                    channelLST.append(tmp0[0])
                    if len(tmp0) == 2:
                        subChannelLST.append(tmp0[1])
                    else:
                        subChannelLST.append(tmp0[1] + "_" + tmp0[2])
                else:
                    tmp0 = tValBudgetCOLS[i][21:].split("_")
                    channelLST.append(tmp0[0])
                    if len(tmp0) == 2:
                        subChannelLST.append(tmp0[1])
                    else:
                        subChannelLST.append(tmp0[1] + "_" + tmp0[2])

            tmpDF = pd.DataFrame()
            tmpDF['Channel'] = channelLST
            tmpDF['Sub_Channel'] = subChannelLST
            for i in range(0, 5):
                estBudgetLST = []
                for j in range(1, len(tValBudgetCOLS)):
                    estBudgetLST.append(np.round_(tValBudget[tValBudgetCOLS[j]][i], 0))
                tmpDF[str(tValBudget['number'][i])] = estBudgetLST

            aggrid_interactive_table(tmpDF)


