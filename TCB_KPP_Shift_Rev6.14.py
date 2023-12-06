#Last modified: W48.3 (Rev6.14)
#Owner: Aqilah / Ean Zou / Nen Huang
import platform
print("python", platform.python_version())

import warnings
warnings.simplefilter(action='ignore', category=Warning)
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import matplotlib.dates as mdates
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.colors as mcolors

# either read by prodgroup3 or directly from raw file and pivot
this_prod = "<<<prodgroup3>>>"
#this_prod = 'MTLH682'
#this_prod = 'SPRXCS'
#this_prod = 'CLXXCP'
#this_prod = 'RHX8161'
#this_prod = 'RHR8161'
#this_prod = 'RPR8161'

# Read the data from the CSV files
main_df = pd.read_csv(f'{this_prod}_kpp.csv')
main_df.columns = main_df.columns.str.upper()
# remove rows without any of below value
main_df = main_df.dropna(subset=['VISUAL_ID', 'COMPLEVEL', 'TEST_OR_BOND_HEAD_ID', 'BONDSTAGE'])

main_control_data = pd.read_csv('kpp_control_file_rev.csv')
main_control_data.columns = main_control_data.columns.str.upper()
main_control_data = main_control_data.drop_duplicates()

# Convert the 'PROCESSTIMESTAMP' column to datetime type for proper plotting
main_df['PROCESSTIMESTAMP'] = pd.to_datetime(main_df['PROCESSTIMESTAMP'],  errors='coerce')

# check if already triggered before
try:
    old_triggers_df = pd.read_csv(f'{this_prod}_arima_anomalies_old.csv').reset_index()
    check_old = True
except:
    check_old = False
    old_triggers_df = pd.DataFrame()
    print("No old triggers")
        
def analyze_kpp_with_arima(data, kpp, ucl, lcl, sequence_length=10):   
    model = ARIMA(data[kpp], order=(1, 0, 1))
    fit_model = model.fit()
    data['predicted'] = fit_model.predict(typ='levels')
    data['residuals'] = data[kpp] - data['predicted']

    mean_residual = data['residuals'].mean()
    std_residual = data['residuals'].std()

    if np.isnan(lcl):
        upper_threshold = mean_residual + (3 * std_residual)
        lower_threshold = np.nan
        data['anomaly'] = (data[kpp] > (data[kpp].mean() + upper_threshold))
    elif np.isnan(ucl):
        upper_threshold = np.nan
        lower_threshold = mean_residual - (3 * std_residual)
        data['anomaly'] = (data[kpp] < (data[kpp].mean() + lower_threshold))
    else:
        upper_threshold = mean_residual + (3 * std_residual)
        lower_threshold = mean_residual - (3 * std_residual)
        data['anomaly'] = (data[kpp] > (data[kpp].mean() + upper_threshold)) | (data[kpp] < (data[kpp].mean() + lower_threshold))

    # Use rolling windows to determine if the sequence of anomalies is long enough
    data['anomaly_sequence'] = data['anomaly'].rolling(window=sequence_length, min_periods=6).sum()

    # Define the 'anomaly' column based on the 'anomaly_sequence' while considering the threshold
    data['anomaly'] = data['anomaly_sequence'] >= sequence_length

    return upper_threshold, lower_threshold, data

def visualize_kpp_arima(data, prodgroup3, link, bonder, kpp, stage, complevel, subs, upper_threshold, lower_threshold, ucl, lcl):
    # Generate the color mapping for the current filtered_data
    colors = [color for color in mcolors.TABLEAU_COLORS if color != 'tab:red']
    num_lots = len(data['LOT'].unique())
    if num_lots > len(colors):
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 1, num_lots)]
    lot_color_mapping = dict(zip(data['LOT'].unique(), colors))
    
    plt.figure(figsize=(8, 3))
    for lot, color in lot_color_mapping.items():
        subset = data[data['LOT'] == lot]
        plt.scatter(subset['PROCESSTIMESTAMP'], subset[kpp], color=color, s=15, label=lot)
 
    # Combine all types of anomalies into a single category "Anomaly"
    #anomalies = data[(data[kpp] > (data[kpp].mean() + upper_threshold)) |
    #                 (data[kpp] < (data[kpp].mean() + lower_threshold)) |
     #                (data['anomaly'])]
    anomalies = data[data['anomaly']]
    plt.scatter(anomalies['PROCESSTIMESTAMP'], anomalies[kpp], color='red', s=40, marker='x', label='Anomaly')
    
    title = f"{prodgroup3}_{link}_{bonder}_{stage}_{complevel}_{subs}_{kpp}"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d:%m:%Y %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.axhline(y=data[kpp].mean() + upper_threshold, color='g', linestyle='--', label='ARIMA Upper Threshold')
    plt.axhline(y=data[kpp].mean() + lower_threshold, color='r', linestyle='--', label='ARIMA Lower Threshold')
    plt.axhline(y=ucl, color='blue', linestyle='--', label='UCL')
    plt.axhline(y=lcl, color='orange', linestyle='--', label='LCL')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(kpp)
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    pdf_pages_arima.savefig(plt.gcf(), bbox_inches="tight")
    plt.savefig(rf'.\{prodgroup3}\{title}.png', bbox_inches="tight")
    #plt.show()
    plt.close()
    
    # Return the anomalies DataFrame
    return anomalies

def append_trigger_data_arima(df, prodgroup3, link, bonder, stage, complevel, subs, kpp, upper_threshold, lower_threshold, ucl, 
                        lcl, anomaly_data, columns, count, old_triggers_df, check_old):
    
    
    title = f"{prodgroup3}_{link}_{bonder}_{stage}_{complevel}_{subs}_{kpp}"
    anomaly_data = anomaly_data[anomaly_data['anomaly'] == 1]
    
    temp_df = anomaly_data[['LOT']]
    temp_df = temp_df.drop_duplicates()
    temp_df = temp_df.assign(PRODGROUP3=prodgroup3, LINK_ID=link, TEST_OR_BOND_HEAD_ID=bonder, BONDSTAGE=stage, KPP=kpp, VENDOR_ID=subs,
                                       ARIMA_UCL=upper_threshold, ARIMA_LCL=lower_threshold, UCL=ucl, LCL=lcl, COUNT=count,
                                       TITLE=title, COMPLEVEL=complevel)
    
    #temp_df = pd.DataFrame(temp_data)
    temp_df = temp_df.drop_duplicates()
    
    # sort sequence
    temp_df = temp_df.filter(items = columns)
    
    # check old triggers before appending
    if check_old: 
        check_columns = ['TITLE', 'LOT']
        merged_data = pd.merge(temp_df[check_columns], old_triggers_df[check_columns], how='inner')
        triggered = len(merged_data) > 0
        if triggered:
            return df, count, triggered
        else:
            df = df.append(temp_df, ignore_index=True)
            count = count + 1
            return df, count, triggered
    else:
        df = df.append(temp_df, ignore_index=True)
        count = count + 1
        triggered = False
        return df, count, triggered

def custom_agg(x):
    x = x.sort_values('KPP') 
    
    return pd.Series({
        'LINK_ID': ', '.join(dict.fromkeys(x['LINK_ID'])),
        'TEST_OR_BOND_HEAD_ID': ', '.join(dict.fromkeys(x['TEST_OR_BOND_HEAD_ID'])),
        'BONDSTAGE': ', '.join(dict.fromkeys(x['BONDSTAGE'])),
        'KPP': ', '.join(dict.fromkeys(x['KPP'])),
        'COMPLEVEL': ', '.join(dict.fromkeys(x['COMPLEVEL'])),
        'COUNT': x['COUNT'].max(),
        'VENDOR_ID':x['VENDOR_ID'].max()
    })

def quick_summary(df):
    summary_columns = ['PRODGROUP3', 'LINK_ID', 'TEST_OR_BOND_HEAD_ID', 'BONDSTAGE', 'KPP', 'COMPLEVEL', 'VENDOR_ID', 'COUNT']
    # filter and drop dup
    sum_df = df.filter(items = summary_columns)
    sum_df = sum_df.drop_duplicates()
    sum_df = sum_df.sort_values(by=['KPP']).reset_index(drop=True)

    sum_df = sum_df.groupby('PRODGROUP3').apply(custom_agg).reset_index()
    
    return sum_df

prodgroup3_list = main_df['PRODGROUP3'].unique()
#############################################################################################
for prodgroup3 in prodgroup3_list:
    # Initialize a PDF to save all plots
    pdf_pages_arima = PdfPages(f"{prodgroup3}_ARIMA_anomalies.pdf")
    
    # flag the pdf as empty
    empty_arima = 1
    
    # create empty df for data appending
    columns = ['PRODGROUP3', 'LINK_ID', 'TEST_OR_BOND_HEAD_ID', 'BONDSTAGE', 'KPP', 'COMPLEVEL', 'VENDOR_ID',
               'ARIMA_UCL', 'ARIMA_LCL', 'UCL', 'LCL', 'TITLE', 'COUNT', 'LOT']
    
    count = 1
    triggered_combo = pd.DataFrame(columns = columns) 
    
    # filter current prodgroup3
    prod_df = main_df[main_df['PRODGROUP3'] == prodgroup3].copy()
    control_data = main_control_data[main_control_data['PRODGROUP3'] == prodgroup3].copy()
    
    # Replace all -9999 values with NaN
    control_data = control_data.replace(-9999, np.nan)

    # get the associated KPPs 
    kpps_to_analyze  = control_data[control_data['MONITOR'] == 1]['KPP'].unique()
    #kpps_to_analyze = filtered_data['KPP'].unique()
    
    # get substrate vendor list
    vendor_list =  main_df['VENDOR_ID'].unique() 
    
    #############################################################################################
    for subs in vendor_list:
        subs_df = prod_df[prod_df['VENDOR_ID'] == subs].copy()
        complevel_list = subs_df['COMPLEVEL'].unique()
        
        #############################################################################################
        for complevel in complevel_list:
            df = subs_df[subs_df['COMPLEVEL'] == complevel].copy()
            
            #############################################################################################
            for kpp in kpps_to_analyze:

                # skip processing if control file kpp not found
                if kpp not in df.columns:
                    continue

                # Fetch UCL and LCL for the current KPP and prodgroup3 from the control file data
                control_ucl = f'HILIMIT_{complevel}'
                control_lcl = f'LOLIMIT_{complevel}'

                ucl_lcl_data = control_data[(control_data['KPP'] == kpp)][[control_ucl, control_lcl]]
                if ucl_lcl_data.empty:
                    print(f"No UCL/LCL data found for KPP: {kpp} and prodgroup3: {prodgroup3}")
                    continue

                ucl = ucl_lcl_data[control_ucl].values[0]
                lcl = ucl_lcl_data[control_lcl].values[0]

                # Fetch user defined target (directly skip processing if within target)
                hi_lo_data = control_data[(control_data['KPP'] == kpp)][['HITARGET', 'LOTARGET']]
                if not hi_lo_data.empty:
                    max_val = df[kpp].max()
                    min_val = df[kpp].min()
                    hi = hi_lo_data['HITARGET'].values[0]
                    lo = hi_lo_data['LOTARGET'].values[0]

                    if not np.isnan(lo) and not np.isnan(hi):
                        # if have both hi and lo target
                        if max_val < hi and min_val > lo:
                            print(f"{prodgroup3} skip {kpp} as {max_val} < {hi} and {min_val} > {lo}")
                            continue
                    elif not np.isnan(lo) and np.isnan(hi):
                        # if only have low target
                        if min_val > lo:
                            print(f"{prodgroup3} skip {kpp} as {min_val} > {lo}")
                            continue
                    elif not np.isnan(hi) and np.isnan(lo):
                        # if only have high target
                        if max_val < hi:
                            print(f"{prodgroup3} skip {kpp} as {max_val} < {hi}")
                            continue

                combinations = list(set(tuple(zip(df['LINK_ID'], df['TEST_OR_BOND_HEAD_ID'], df['BONDSTAGE']))))
                
                #############################################################################################
                for combo in combinations:
                    # only filter by bonder, filter bondstage as well
                    if len(combo) == 3:
                        link, bonder, stage = combo
                        filtered_data = df[(df['LINK_ID'] == link) & (df['TEST_OR_BOND_HEAD_ID'] == bonder) & (df['BONDSTAGE'] == stage)].copy()

                    filtered_data = filtered_data.sort_values(by=['PROCESSTIMESTAMP']).reset_index(drop=True)

                    if filtered_data.empty:
                        continue

                    # Check if the KPP exists in the filtered data
                    if kpp not in filtered_data.columns:
                        continue  # Skip the current iteration and move to the next combination

                    # skip processing if whole column having same value
                    if len(filtered_data[kpp].unique()) <= 1:
                        #print('skip', kpp)
                        continue  # Skip the current iteration

                    #################################### ARIMA ALGO ####################################
                    #initialize rolling window value
                    sequence_length = 10 
                    upper_threshold, lower_threshold, filtered_data = analyze_kpp_with_arima(filtered_data, kpp, ucl, lcl,sequence_length=sequence_length)

                    # output summary table with triggered lot and vid (with respective KPP)
                    if filtered_data['anomaly'].sum() > 0:
                        # Convert boolean to integer
                        triggered_combo, count, triggered = append_trigger_data_arima(triggered_combo, prodgroup3, link, bonder, stage, complevel, subs, kpp, 
                                                                     upper_threshold, lower_threshold, ucl, lcl, filtered_data, columns, count, old_triggers_df,check_old)
                        if triggered:
                            print(prodgroup3, complevel, kpp, combo, complevel, subs, 'triggered last session, removing')
                        else:
                            print(prodgroup3, complevel, kpp, combo, complevel, subs, 'Arima have abnormalites!!!')
                            visualize_kpp_arima(filtered_data, prodgroup3, link, bonder, kpp, stage, complevel, subs, upper_threshold, lower_threshold, ucl, lcl)
                            empty_arima = 0
                    #else:
                        #print(prodgroup3, complevel, kpp, combo, 'clean')
                
    # Construct the filename
    file_name = f"{prodgroup3}_arima_anomalies.csv"
        
    # Save the dataframe
    triggered_combo.to_csv(file_name, index=False)
    
    # get summary count
    if len(triggered_combo) > 0:
        sum_df = quick_summary(triggered_combo)
        sum_df.to_csv(f"{prodgroup3}_last_xx_hours.csv", index=False)
    
    # Save all plots to the PDF and close it
    pdf_pages_arima.close()

    # Delete the pdf if it's empty
    if empty_arima:
        os.remove(f"{prodgroup3}_ARIMA_anomalies.pdf")

print('-------------COMPLETED-------------')