import os
import pandas as pd

# Define folder paths
folder_path = r'C:\Users\USER\source\repos\walerazaq\spatiotemporal-energy-forecasting\data\hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01'
output_path = r'C:\Users\USER\source\repos\walerazaq\spatiotemporal-energy-forecasting\cleaned_data'

# work hours
start_hour = 6
end_hour = 19

# List all CSV files in the folder, excluding the specific file
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != 'MainTerminal_PhaseCount_3_geq_2017-10-01_lt_2018-01-01.csv']

# Initialize lists to hold the dataframes
dataframes_p_kw = []
dataframes_s_kva = []
dataframes_std_p_kw = []
dataframes_std_s_kva = []

# Loop through each file
for file in csv_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, file)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract the required columns: SensorDateTime, P_kW, and S_kVA
    df = df[['SensorDateTime', 'P_kW', 'S_kVA']]
    
    # Remove the timezone part by splitting the string
    df['SensorDateTime'] = df['SensorDateTime'].str.split('+').str[0]
    
    # Convert SensorDateTime to datetime
    df['SensorDateTime'] = pd.to_datetime(df['SensorDateTime'], format='mixed')

    # Filter out weekends
    df = df[df['SensorDateTime'].dt.weekday < 5]

    # Filter out off-work hours (keep rows where hour is within working hours)
    df = df[(df['SensorDateTime'].dt.hour >= start_hour) & (df['SensorDateTime'].dt.hour < end_hour)]   
    
    # Round SensorDateTime to the nearest 10 minutes
    df['SensorDateTime'] = df['SensorDateTime'].dt.round('10T')
    
    # Group by SensorDateTime and aggregate (sum and standard deviation)
    df_agg = df.groupby('SensorDateTime').agg(
        P_kW_sum=('P_kW', 'sum'),
        S_kVA_sum=('S_kVA', 'sum'),
        std_P_kW=('P_kW', 'std'),
        std_S_kVA=('S_kVA', 'std')
    ).reset_index()

    # Replace NaN values in std deviation with 0
    df_agg['std_P_kW'] = df_agg['std_P_kW'].fillna(0)
    df_agg['std_S_kVA'] = df_agg['std_S_kVA'].fillna(0)
    
    # Get the machine name from the filename (prefix before '_')
    prefix = file.split('_')[0]
    
    # Rename the columns based on the machine
    df_agg.rename(columns={
        'P_kW_sum': f'{prefix}_P_kW',
        'S_kVA_sum': f'{prefix}_S_kVA',
        'std_P_kW': f'{prefix}_std_P_kW',
        'std_S_kVA': f'{prefix}_std_S_kVA'
    }, inplace=True)
    
    # Append each variable's dataframe to the appropriate list
    dataframes_p_kw.append(df_agg[['SensorDateTime', f'{prefix}_P_kW']])
    dataframes_s_kva.append(df_agg[['SensorDateTime', f'{prefix}_S_kVA']])
    dataframes_std_p_kw.append(df_agg[['SensorDateTime', f'{prefix}_std_P_kW']])
    dataframes_std_s_kva.append(df_agg[['SensorDateTime', f'{prefix}_std_S_kVA']])

# Merge all dataframes for P_kW, S_kVA, std_P_kW, and std_S_kVA on SensorDateTime
final_p_kw_df = dataframes_p_kw[0]
for df in dataframes_p_kw[1:]:
    final_p_kw_df = pd.merge(final_p_kw_df, df, on='SensorDateTime', how='outer')

final_s_kva_df = dataframes_s_kva[0]
for df in dataframes_s_kva[1:]:
    final_s_kva_df = pd.merge(final_s_kva_df, df, on='SensorDateTime', how='outer')

final_std_p_kw_df = dataframes_std_p_kw[0]
for df in dataframes_std_p_kw[1:]:
    final_std_p_kw_df = pd.merge(final_std_p_kw_df, df, on='SensorDateTime', how='outer')

final_std_s_kva_df = dataframes_std_s_kva[0]
for df in dataframes_std_s_kva[1:]:
    final_std_s_kva_df = pd.merge(final_std_s_kva_df, df, on='SensorDateTime', how='outer')

# Replace negative readings with 0 in all dataframes (if necessary)
final_p_kw_df[final_p_kw_df.iloc[:, 1:] < 0] = 0
final_s_kva_df[final_s_kva_df.iloc[:, 1:] < 0] = 0
final_std_p_kw_df[final_std_p_kw_df.iloc[:, 1:] < 0] = 0
final_std_s_kva_df[final_std_s_kva_df.iloc[:, 1:] < 0] = 0

# Create a dataframe for the sum of P_kW across all machines
final_p_kw_sum_df = final_p_kw_df.copy()
final_p_kw_sum_df['sum_P_kW'] = final_p_kw_sum_df.iloc[:, 1:].sum(axis=1)

# Save each dataframe to CSV
final_p_kw_df.to_csv(os.path.join(output_path, 'P_kW.csv'), index=False)
final_s_kva_df.to_csv(os.path.join(output_path, 'S_kVA.csv'), index=False)
final_std_p_kw_df.to_csv(os.path.join(output_path, 'std_P_kW.csv'), index=False)
final_std_s_kva_df.to_csv(os.path.join(output_path, 'std_S_kVA.csv'), index=False)
final_p_kw_sum_df[['SensorDateTime', 'sum_P_kW']].to_csv(os.path.join(output_path, 'total_power.csv'), index=False)

print("Done!")
