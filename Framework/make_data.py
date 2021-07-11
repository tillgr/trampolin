import Framework.process_data as process_data


def create_all_data():

    process_data.make_all_data()
    process_data.make_data_only_jumps()
    for p in ['with_preprocessed', 'without_preprocessed']:
        for m in [None, 'mean', 'mean_std']:
            process_data.make_percentage_data(p, ['0.25', '0.20', '0.10', '0.05', '0.02', '0.01'], m)
    for p in ['with_preprocessed', 'without_preprocessed']:
        process_data.make_avg_std_data(p)
    for p in ['with_preprocessed', 'without_preprocessed']:
        for n in ['percentage_mean_std', 'percentage_mean', 'percentage']:
            process_data.make_AJ_jumps(p, n, ['25', '20', '10', '5', '2', '1'])


def create_used_data():

    process_data.make_all_data()
    process_data.make_data_only_jumps()

    for p in ['with_preprocessed', 'without_preprocessed']:
        for m in [None, 'mean', 'mean_std']:
            process_data.make_percentage_data(p, ['0.25', '0.20', '0.10', '0.05'], m)


if __name__ == '__main__':

    print("You can now choose if you want to create all data or only the data files which are actually used")
    print("All files will be saved in Sprungdaten_processed/")
    print("Keep in mind this will take some time")
    print("Do you want to create all data? [y/n]")
    x = input()

    if x == 'y' or x == 'Y':
        create_all_data()

    elif x == 'n' or x == 'N':
        create_used_data()

    else:
        raise AttributeError("You did not specify whether to create all data or only the used ones")
