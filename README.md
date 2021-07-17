# Trampolin - Sportdatenanalyse
A short Documentation of the Framework

1. [Dataintegration](#dataintegration)
    1. [Process data](#process-data)
    2. [Split of train and test data](#split-of-train-and-test-data)
    3. [Jump detection](#jump-detection)
    4. [How to create data](#how-to-create-data)
2. [Models](#models)
    1. [How to train the models for predictions](#how-to-train-the-models-for-predictions)
3. [Predictions](#predictions)
    1. [Steps for predictions](#steps-for-predictions)
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
Percentage data is our most used dataset as it leads to the best results.

We define a percentage_step and each jump is then segmented into 100/percenage_steps parts.
We use a couple of different steps:
- 1%
- 2%
- 5%
- 10%
- 20%
- 25%

Example: For 5% steps we will have 20 data points for each jump.

Furthermore we have 3 different methods, which do different things with the data for each step:
- None
- Mean
- Mean & Std

When using the None method, we just take the datapoints at the indexes corresponding to the percentage_steps.<br>
When using the mean method, we calculate the average of each feature between two steps.<br>
When using the mean_std method, we calculate the average and the standard deviation of each feature between two steps.

#### AJ data
This is self created data, which calculates an average jump for each jump class.<br>
This is done by collecting all jumps of a class and then averaging each feature.

#### Difference between vectorized data and non-vectorized data
Non vectorized data looks the way we discussed until this point. Meaning each jump has multiple rows. The thing is, we can only use this version for the CNN. For all other models the data for each jump need to be in one row. So we need to vectorize it.<br>
We do this by concatenating all rows and then naming them according to the corresponding percentage_step.
This means that we have multiple columns which correspond to the same feature, just for different datapoints.

### Split of train and test data
We split the data into train/test using a 80/20 split.
Because we use the stratify function from sklearn to keep the distribution, we need atleast 3 examples of each jump, so we remove all jumps which occur less than 3 times. This number can also be adjusted in the future.

### Jump detection
Later, when we use the models to make predictions for the athletes, we need to detect when a new jump starts.

This is being done by observing the Acc_N_Fil feature and looking for local maximums of it.

### How to create data
If you want the easiest way to create the data, just run `make_data.py`.<br>
This will prompt you to choose whether you want to create all the data again or if you want to create only the data we actually use. The latter is ofcourse faster as less datasets need to be created.

Keep in mind this will take some time regardless of what you choose.

Also make sure that the raw data is located in individual folders in `Sprungdaten Innotramp/` and both the Rohdaten and the Sprungzuordnung have corresponding names.

## Models
We tested 7 different machine learning algorithms to determine the best.
- Convolutional Neural Network  (CNN)
- Deep Feed Forward Neural Network  (DFF)
- k-nearest Neighbours  (KNN)
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier  (GBC)
- Stochastic Gradient Descent   (SGD)
- Gaussian Naive Bayes  (GNB)

We will not explain here in detail what each of them does.

Important to know is, that we achieved the best results for with_preprocessed data with the CNN, and the best result for without_preprocessed data with the DFF.

### How to train the models for predictions
The easiest way to train the models is to run the `make_models.py` script. This will prompt you to enter, whether you want to train the DFF (used for predicting data without preprocessed), or CNN (used for preprocessed data). You can also train both.<br>
Please make sure that all the needed data was created before. For that see [How to create data](#how-to-create-data).<br>
Each of the options will trigger the training of the models which can take some time. It uses the best parameter we could determine using our trainings data.

If you want to change the parameters, this can be done via calling the functions in `neural_networks.py`.

## Predictions
To make predictions using our models you can call `make_prediction.py`.<br>
The prediction will be made on data, that is in `Prediction data/`.

When you made sure that the data is there, you can then run the `make_predictions.py` script. You then need to enter, whether or not the data contains preprocessed data, as we use different models for predicting those. The process should then start and should'nt take too long.

### Steps for predictions
We can't just predict on the raw data that will be delivered. We will now show which steps we take in order to make predictions.

1. Detect start of each jump
2. Give unique ID to each jump
3. Apply percentage preprocessing to the data
4. Predict on the data using the correct model

## Plots
