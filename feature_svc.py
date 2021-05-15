import pandas as pd
import logging

from pandas import DataFrame, Series

logging.basicConfig(filename='svc.log', format='%(asctime)s[%(name)s] - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('Feature Engineering')


def read_data(filename: str):
    processed_data = pd.read_csv(filename)
    for column in processed_data.columns:
        try:
            processed_data[column] = processed_data[column].str.replace(',', '.')
        except AttributeError:
            logger.info('str not appliable for column with key: ' + column)
    processed_data = processed_data.round(
        {'ACC_N': 3, 'ACC_N_ROT_filtered': 3, 'Acc_x_Fil': 3, 'Acc_y_Fil': 3, 'Acc_z_Fil': 3,
         'Gyro_x_Fil': 3, 'Gyro_y_Fil': 3, 'Gyro_z_Fil': 3,
         'DJump_SIG_I_x LapEnd': 3, 'DJump_SIG_I_y LapEnd': 3, 'DJump_SIG_I_z LapEnd': 3,
         'DJump_Abs_I_x LapEnd': 3, 'DJump_Abs_I_y LapEnd': 3, 'DJump_Abs_I_z LapEnd': 3,
         'TimeInJump': 3, 'Time': 3})
    return processed_data


def time_frames(data: DataFrame, portion: int):
    columns = ['SprungID', 'Sprungtyp']
    p = 0
    while p < portion-1:
        p += 1
        columns.append('Acc_x_Fil_' + str(p))
        columns.append('Acc_y_Fil_' + str(p))
        columns.append('Acc_z_Fil_' + str(p))
        columns.append('Gyro_x_Fil_' + str(p))
        columns.append('Gyro_y_Fil_' + str(p))
        columns.append('Gyro_z_Fil_' + str(p))
    # columns = columns + ['DJump_SIG_I_x LapEnd', 'DJump_SIG_I_y LapEnd', 'DJump_SIG_I_z LapEnd',
    #                      'DJump_Abs_I_x LapEnd', 'DJump_Abs_I_y LapEnd', 'DJump_Abs_I_z LapEnd']
    data_time_splits = pd.DataFrame(columns=columns)

    for id in data['SprungID'].unique():
        subframe = data[data['SprungID'] == id]
        subframe = subframe.reset_index(drop=True)
        jump_type = subframe['Sprungtyp'].unique()[0]
        # lap_ends = subframe.loc[0:0, 'DJump_SIG_I_x LapEnd': 'DJump_Abs_I_z LapEnd']
        i = 0
        values = [id, jump_type]
        while i < portion-1:
            i += 1
            slice: Series = subframe.loc[int(len(subframe.index) * i / portion), 'Acc_x_Fil':'Gyro_z_Fil']
            values = values + slice.to_list()
        data_time_splits.loc[len(data_time_splits)] = values
    return data_time_splits


def quater_average(data: DataFrame, portion: int):
    columns = ['SprungID', 'Sprungtyp']
    p = 0
    while p < portion:
        p += 1
        columns.append('Acc_x_Fil_avg_' + str(p))
        columns.append('Acc_y_Fil_avg_' + str(p))
        columns.append('Acc_z_Fil_avg_' + str(p))
        columns.append('Gyro_x_Fil_avg_' + str(p))
        columns.append('Gyro_y_Fil_avg_' + str(p))
        columns.append('Gyro_z_Fil_avg_' + str(p))
    data_time_splits = pd.DataFrame(columns=columns)

    for id in data['SprungID'].unique():
        subframe = data[data['SprungID'] == id]
        subframe = subframe.reset_index(drop=True)
        jump_type = subframe['Sprungtyp'].unique()[0]
        i = 0
        values = [id, jump_type]
        while i < portion:
            i += 1
            slice = subframe.loc[int(len(subframe.index) * (i-1) / portion):int(len(subframe.index) * i / portion), 'Acc_x_Fil':'Gyro_z_Fil'].mean().round(3)
            values = values + slice.to_list()
        data_time_splits.loc[len(data_time_splits)] = values
    return data_time_splits


if __name__ == '__main__':
    # data = read_data('Sprungdaten_processed/data_point_jumps/data_point_jumps_train.csv')
    # portioned = time_frames(data, 101)
    # portioned.to_csv('Sprungdaten_processed/jumps_time_splits/jumps_time_splits_train_101.csv', index=False)
    # data = read_data('Sprungdaten_processed/data_point_jumps/data_point_jumps_test.csv')
    # portioned = time_frames(data, 101)
    # portioned.to_csv('Sprungdaten_processed/jumps_time_splits/jumps_time_splits_test_101.csv', index=False)
    data = read_data('Sprungdaten_processed/data_point_jumps/data_point_jumps_train.csv')
    quatered = quater_average(data, 2)
    quatered.to_csv('Sprungdaten_processed/jumps_time_splits/half_avg_train.csv', index=False)