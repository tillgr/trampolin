# Trampolin - Sportdatenanalyse
A short Documentation of the Framework

1. [Dataintegration](#dataintegration)
2. [Models](#models)
3. [Predictions](#predictions)
4. [Plots](#plots)

## Dataintegration
We will first cover how we create all the different data and how we can detect jump starts.
If you just want to know how to run the script which creates the data skip to [How to create data](#how-to-create-data).

### Process data
Files we got:
- Rohdaten  ->  many datapoints, 0.002s steps, non-preprocessed data
- Sprungzuordnung ->  Classes of jumps, preprocessed data

#### All data
To create all data we need to map the Sprungzuordnung to the Rohdaten. We do that by joining the two together on the time.

The files have to be in a folder and have to have the same name for Rohdaten and Sprungzuordnung.<br>
E.g. a folder named 2019.08.27 containing:
- Rohdaten_BK_3_Salto_CBA.19_0827-BK1-01.csv
- Sprungzuordnung_BK_3_Salto_CBA.19_0827-BK1-01.XLSX

These folders need to be in the folder `Sprungdaten Innotramp/`.

When we run the `make_all_data()` function from `process_data.py` we first get the data divided by folder, so that we can save some RAM. Afterwards we combine them to one big file. The parts will be saved to `Sprungdaten_processed/all_data/`

Finally we get a big `all_data.csv` file in `Sprungdaten_processed/`

This file now contains all the data, except for the pauses as that would make the file way to large

#### Data only jumps
For the data with only the jumps we remove the errors, the Einturnen and correct some spelling mistakes. Additionally we sample the Strecksprünge so that they are not overrepresented.

You can set the number of Strecksprünge to be contained in the data by altering the `num_strecksprünge` parameter in `make_data_only_jumps` from `process_data.py`

This is the first step where we differentiate between `with_preprocessed data` and `without_preprocessed data`, as we will get two files from this function.

#### Difference between with_preprocessed and without_preprocessed
Our data is differentiated into two big categories. As you probably guessed those categories are with_preprocessed and without_preprocessed.
The difference between them being, that the data in with_preprocessed contains all column named `DJump...`. This is the preprocessed data which originates from the Sprungzuordnung. All other columns stay the same.

We can further differentiate between the DJump columns by dividing them into 4 categories:
1. first 9 columns of preprocessed data
2. DJump_SIG_I_S...
3. DJUMP_ABS_I_S...
4. DJUMP_I_ABS_S...

This is important for the models, as some models like some type of preprocessed data more than others. To reflect that in their parameters, we give them a list of which type of data to use.<br>
E.g.: `pp_list = [3, 4]`

From this point on we will take it as granted that all data we create, will be created with- and without preprocessed data.

#### Avg std data
We will now create 3 datasets:
- avg_data
- std_data
- avg_std_data

avg_data contains the averaged data for each feature(column) for each jump.

std_data contains the standard deviation for each feature for each jump. For with_preprocessed data this would be useless as all DJump columns per jump would be 0.
This is the only time a with_preproccessed and without_preprocessed variant will be the same.

avg_std_data merges avg_data and std_data together. This is done in a way, that the original order of features will be held in place. Meaning that the avg and std part of a feature are besides each other.

#### Percentage data

#### AJ data

#### Difference between vectorized data and non-vectorized data

### Split of train and test data

### Jump detection

### How to create data

## Models

## Predictions

## Plots
