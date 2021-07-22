# best of models
*sorting descending*


|Classifier|Result|with preproc. Data |Data Set|Parameters|
|--|--|--|--|--|
|DFF |Acc: 0.9638<br>Youden: 0.9629<br>F1: 0.9638|no |percentage_mean_std_20 |act_func='relu', loss='categorical_crossentropy', optim='Nadam', epochs=50|
|CNN |Acc: 0.9614<br>Youden: 0.9582<br>F1: 0.9636 |yes: (3) only DJump_ABS_I_S|percentage_mean_std_20 |conv=3, kernel=3, pool=2, dense=2, act_func='tanh', loss='kl_divergence', optim='Nadam'|
|CNN |Acc: 0.9589<br>Youden: 0.9532<br>F1: 0.9576| no |percentage_mean_5|conv=3, kernel=3, pool=2, dense=2, act_func='tanh', loss='kl_divergence', optim='Nadam' |
|DFF |Acc: 0.9589<br>Youden: 0.9532<br>F1: 0.9564|yes: (3) only DJump_ABS_I_S |percentage_mean_std_25 |act_func='relu', loss='categorical_crossentropy', optim='Nadam', epochs=60 |
|kNN |Acc: 0.9589<br>Youden: 0.9028<br>F1: 0.8949|no |percentage_mean_std_20 |n_neighbors= 3, weights= 'uniform', dist_metric= 'manhattan' |
|kNN |Acc: 0.9565<br>Youden: 0.8915<br>F1: 0,8791 |yes with (0) |percentage_mean_10 |n_neighbors= 3, weights = 'distance', dist_metric= 'manhattan', data=[0]|
|SVC |Acc: 0.942<br>Youden: 0.9435<br>F1: 0.9216|yes with (1) |percentage_mean_20 |kernel = 'linear'|
|SVC |Acc: 0.9372<br>Youden: 0.8984<br>F1: 0.878|no |percentage_mean_std_25 |kernel = 'linear' |
|GBC |Acc: 0.93<br>Youden: 0.834<br>F1: 0.8329|yes with (0)|percentage_25 |n_estimators = 200, max_depth=7 |
|GBC |Acc: 0.93<br>Youden: 0.8163<br>F1: 0.8125|no |percentage_mean_20 |n_estimators = 90, max_depth=3 |
|SGD |Acc: 0.9155<br>Youden: 0.8689<br>F1: 0.8773|yes:(1,3,4) = without DJump_SIG_I_S |percentage_mean_std_10 |loss='log', penalty='l1', max_iter=10000 |
|GNB |Acc: 0.8889<br>Youden: 0.8044<br>F1:0.7628|yes with (0) |percentage_mean_std_10|- |
|GNB |Acc: 0.8865<br>Youden: 0.8028<br>F1: 0.761|no |percentage_mean_std_10 |- |
|SGD |Acc: 0.8841<br>Youden: 0.8124<br>F1: 0.795|no |percentage_mean_std_10 |loss='perceptron', penalty='l1',max_iter=10000,random_state=10 |




**Variants with preprocessed data**: <br>
(0) all preprocessed data <br>
(1) first 9 columns (without S in name) <br>
(2) starts with DJump_SIG_I_S <br>
(3) starts with DJump_ABS_I_S <br>
(4) starts with DJump_I_ABS_S
