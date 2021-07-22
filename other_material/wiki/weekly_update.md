# 20.07.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lukas|- fixed jump start detection<br>- got threshold for jump start detection<br>- finished prediction function<br>- wrote Documentation for Framework<br>- helped with Presentation<br>- ran framework test|||
|Lisa|- helped with fixing start detection<br>- made powerpoint<br>- saved (plot) data<br>- function to save plot data and shap values<br>- checked plot data of all|||
|Weiyun ||||
|Till |||
|Anna |- add shap plots, csv and txt for plots <br>- comment the code <br>- add sgd slide|||

# 13.07.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lukas|- Framework files<br>- single files to create data, train models, predict|||
|Lisa|- started with Presentation<br>- Framework file for plotting<br>- DFF AJ plots|||
|Weiyun ||||
|Till |||
|Anna |- test jump detection with all_data set <br>- made new shap values with new color map <br>- made new funktions and deleted unused folder||power point sgd slide, comment the code, add .csv and .txt for plots |

# 06.07.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lukas|- jump core analysis plots: smaller y-axis and alpha=0.7<br>- Started cleaning and commenting code for framework like structure|||
|Lisa|- wrote function for all:<br>- plots for global feature importance<br>- plots for feature importance for each jump<br>-> ran the function for DFF for all models|||
|Weiyun |shap plots||redo ship plots and clean up code with comment|
|Till |- new shap plots for kNN, GBC|- TreeExplainer not supported for GBC, using KernelExplainer instead||
|Anna |- new shap values for SGD, jump detection||-shap values for only preprocesed data sets|

# 29.06.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lukas|- jump core analysis<br>- diagram for jump core analysis|||
|Lisa|- new colormap -> function for generating colormap <br> (so that each jump get always the specific color for him)<br>- updated bar plots<br>- new shap plots for each jump | | |
|Weiyun | -plots for shap values <br> -helped Anna with recognising the jumps in a jump series <br> -jump phase analysis|-when just looking at the 20-80% of the jump. The so called main action phase, the prediction accuracy went down.||
|Till |- run GBC randomSearchCV on all data sets --> best params <br>- run GBC with best params on all data sets <br>- fill out metric table||shap values for kNN and GBC
|Anna |jump detection with all_data.csv|-|test jump detection with new data, shap|

# 22.06.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- created all the data again due to typos in class names<br>- created SHAP sampled data<br>- created SHAP average jump data<br>- SHAP for DFF (summary and beeswarm)<br>- SHAP for CNN (image)<br>- Confusion Matrix for DFF and CNN<br>- Colormaps for plots and CM<br>- helped Till with Shap, and GB and optimize kNN Result|--|--|
|Weiyun |- GNB testing results in matrix was somehow wrong, corrected that.<br> - while doing last point, ran some more combinations, found a better result for SVC <br> -confusion matrix for SVC, GNB|||
|Till |- kNN: updated metric table (new params: n_neighbors [3...15]), shap values<br>- GBC: new params: depth [1...5]|- GBC: still running slowly| - waiting for GBC best combination --> metric table + shap |
|Anna |-shap values as diagrams|-|jump detection |


# 15.06.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- updated data sets<br>- deleted old data sets<br>- wrote process_all_data more adaptive on data <br>- random grid search for cnn and dff <br>- tested cnn and dff on new datasets|||
|Weiyun |- write a mini script to iterate through all data sets <br> -updated SVC and GNB results||-shap values without the features that does not have positive impact on accuracy|
|Till |- kNN: tested all data sets with all parameters for best combination, updated metric table|- GBC very slow|- GBC testing <br>- shap values|
|Anna |-test all data sets with all parameters for best combination <br> -add results to metric table|-|-shap values <br> -jump detection|


# 08.06.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- reading papers <br>- adding shap for cnn and dff|- preprocessed data not consistent with old data (missing jumps and other time in jump) --> waiting for new data |<br>- updating data sets and testing for cnn and dff|
|Weiyun |papers, shap for svc(KernelExplainer for multiclass) | summary plots looks weird and it takes forever to run it| - print some meaningful shap values plot<br> |
|Till |trying shap|shap not working, type/construction/masking errors <br> --> fixed: running shap for kNN|updating data table and testing for GNB and kNN|
|Anna |run shap|-|add data to metric table|


# 01.06.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- run cnn and dff without preprocessed data and without ACC_N_ROT_Fil <br>- write best of models|--|- maybe adapt data (without acc_n_rot_filtered) <br> - function for updating datasets <br>- youden or accuracy curve with cutted data<br>- shap|
|Weiyun |- run all data sets with gab <br> -run with and without processed/n rot filtered|--|--|
|Till |run kNN, GB and with all data sets, document accuracy and scores in metrics table|--|--|
|Anna |-run sgd with all datasets, document accuracy |--|--|


# 25.05.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- fix percentage data sets<br>- new tabel for metrics (in excel and wiki) <br>- implemented youden and f1 for naive classifier <br>- filled metric data for cnn and dffnn |-|- try cnn and dffnn without preprocessed and only preprocessed data <br>- make "best of models" tabel|
|Weiyun |- run SVC with all datasets in metrics and filled the scores. <br>-first implementation of Gaussian NB.  | | run all datasets on GNB and fill metrics|
|Till | implemented Gradient Boosting Classifier, implemented scores/metrics, tested different data sets| | test classifier with different parameters, fill out metrics table|
|Anna | -implementierung von dem Algorithmus für die Erkennung vom Sprunganfang. <br>-run with vector_percentage_mean_std_10_train.csv datacet -> accuracy 90% |-| fill in metrics table |

# 18.05.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas|- Deep Feed Forward NN not so great (accuracy around 80%) <br>- implement Youden and F1 score for CNN <br>- TP, Youden etc. for naive classifier  |- Sample size of jumps in trainings/test/validation data different -> different shapes|fixing metrics calculation; sorting accuracy table; try CNN with merge avg_std|
|Weiyun|new dataset: quarter average the measurements of each acc, gyro per jumps. <br> -> no significant improvement on prediction|| - Clean up svc.py. <br> - run classifier with percentage data again(to align the data sets.) <br> -Gaussian Naive Bayes Claasifier|
|Till|tested different datasets, best accuracy until now is 90%||fill in metrics table|
|Anna|run with all datasets -> improvement with preditions -> best accuracy 89%|-|what else can be done to improve accuracy?|

# 11.05.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Lisa & Lukas |- calculating accuracy for naive Classifier <br>- generating new datasets without time dependency for different percents <br>- creating test and trainingsdata <br>- merge std and mean in one file <br>- starting CNN with grid search for parameters |- |Deep Feed Forward NN or so |
|Weiyun|Run with more data sets/ features combination. <br> Best Result till now 93.52% <br> split around 1% the result got worse. between 50-100-> not clear <br> Parameter other than linear kernel has no effects till now.||look into why the other parameters brings till now nothing|
|Till|implemented different variations of paramters, best accuracy about 85%|-|testing on various data sets|
|Anna|accuracy implemented, run with standardize dataset|what dataset to use in order to get more than 75% accuracy?|run with all datasets and choose most accurate result|

# 04.05.2021

|Participant|Progress|Problems?|Next|
|--|--|--|--|
|Weiyun|1.SVC(linear) classify using abs value. Prediction accuracy: best around 80% <br>2. new feature: measurements at x% time of the whole jump duration: Accuracy swings around 85-89%.(finer split does not lead to a much better result) <br>|Not clear about Train, test, validate datasets| 1. Train with other features(which? tbd) <br>2. Train with reasonable train, test, validation set?|
|Till|kNN implemented, needs quick lookover|which dataset is the right one to use?|Course of dimensionality|
| Lisa & Lukas | sorted out spelling errors, Datasets: only jumps with points, average and std data for each jump, min-max normalized data, random classifier | - | Trainings/Test-Split, all jumps same length, evaluation of classifier, delete jumps with too few occurences |
|Anna|SVG implemented without accuracy|what loss function should be used to differ from SVC classifier?|add accuracy and train with dataset|
