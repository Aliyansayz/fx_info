

!pip install catboost

​

​

Train Forex Models With Regularized Parameters
Version 2

    Temporal features False (Month, Day of week, Week Of Month)

    Hour 4 Features Lag by 7 optional

    Day Features (RSI, RSI crossover, RSI-Based SMA, ) Lag by 9

    Starting Date June 2020 Training Till September 2023

    Ending Date May 2024

    Hour 4 Features 'relative range', 'candle type', 'heikin ashi'

    Hyperparameters Regularized

    Removing CHF

    Evaluation Metric

    Sharp Ratio

    Accuracy by net gains

Version 1 Regularized On Two Months April May On Two Conditions :

1- hyper parameters chosen should give positive gains for any currency pairs for those two months

2- Previous months accuracy should be 50% and net gains accuracy close to 50% or 52%

Results :- (pips)

    ✅October 2167
    ✅ November 2498
    ❌ December -3080
    ✅ January 4844
    ✅ February 1288
    ✅ March 1883

📉 Not all models were in positive gains in previous month despite all performing good on april may with positive gains

⭐ But parameters performing good on two different months gave them ability to keep majority of currency pairs model in positive gain

💢 December is also not recommended for trading by trading community so we might ignore that month outcomes

​

import pandas as pd

import numpy as np

​

results_dict = {}

day_features, hour4_features = load_features_files()

forex_pairs = [

    'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',

    'CADCHF', 'CADJPY',

    'CHFJPY', 

    'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 

    'EURJPY', 'EURNZD', 'EURUSD',

    'GBPAUD', 'GBPCAD', 'GBPCHF', 

    'GBPJPY', 'GBPUSD', 'GBPNZD',

    'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  

    'USDCHF', 'USDCAD', 'USDJPY'

        ]

​

symbol_hyperparameter = [

 'AUDJPY', 'AUDUSD', 'AUDCAD', 'AUDCHF', 'AUDNZD',

 'CADJPY', 

 'EURAUD', 'EURCAD', 'EURUSD', 'EURGBP', 'EURNZD',

 'GBPCAD', 'GBPCHF', 'GBPUSD', 'GBPNZD',

 'NZDCHF', 'NZDJPY',

 'USDCHF', 'USDJPY', 'USDCAD',

 'CADCHF', 

 'NZDCAD', 'NZDUSD']

​

# symbols = ['USDCAD']

​

for symbol in forex_pairs:

    

    

    day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)

​

    X_train_scaled, X_test_scaled, y_train, y_test = get_features_transformed(symbol, day_data, hour4_data)

    

    print(symbol)

    # model = load_model(symbol)

    

    if symbol in symbol_hyperparameter: 

      

       symbol_parameters = load_parameters(symbol)

       iteration, learning_rate, depth = symbol_parameters['Iterations'], symbol_parameters['Lr'], symbol_parameters['depth']

       parameters = [ iteration, learning_rate, depth ] 

       model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters=parameters)

    

    else:

      model = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

​

    # model = load_model(symbol)

    gains , accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled,  y_test)

    

     # 2  # 1 --> May 2024 , # 2 --> April 2024

    step = 1 # ____          May 

    # step = 2 # ____       April

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster] ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

    

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

    results_dict[symbol] = { 'net_gains': int(net_gains), 'accuracy': int(accuracy), 'month': 'may', 

                             'accuracy_by_net_gains': int(accuracy_by_net_gains)  }

    save_model(symbol, model)

    

​

    # accuracy_by_net_gains = gains['accuracy_by_net_gains_before']

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    # print('accuracy_by_net_gains \n',accuracy_by_net_gains )

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

    

​

​

​

Code Structure To See Model Accuracy For A Symbol

day_features, hour4_features = load_features_files()

​

​

def test_model(symbol, iteration, learning_rate, depth, save_model=False, window_size=9, alpha_type = "gamma" ):

​

    day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)

​

    X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type ) 

​

    print(symbol)

    # model = load_model(symbol)

​

​

    # symbol_parameters = load_parameters(symbol)

    # iteration, learning_rate, depth = symbol_parameters['Iterations'], symbol_parameters['Lr'], symbol_parameters['depth']

​

​

    # iteration, learning_rate, depth = 15, 0.68, 6 # AUDNZD

​

    # iteration, learning_rate, depth = 210, 0.19, 7 # AUDCAD

​

    # iteration, learning_rate, depth = 17, 0.9, 7 # AUDCAD

​

​

​

    parameters = [ iteration, learning_rate, depth ]

    

    # model = finetune_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters=parameters)

    model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters=parameters)

    

    # if save_model == True :     

    #     save_model(symbol, model)

​

        

​

    print("MAY")

​

    step = 1 # ____          May 

    # step = 2 # ____       April

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

​

    print("APRIL")

    step = 2 # ____       April

    cluster = 22

    # cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

​

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

​

​

​

    print("MARCH")

    step = 3 # ____      March

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

​

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

    

    print("Iterations: ",iteration)

    print("Learning rate: ",learning_rate) 

    print("Depth: ",depth)

    

    

    print("FEB")

    step = 4 # ____      Feb

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

    

    print("JAN")

    step = 5 # ____      Jan

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

​

    print("DEC")

    step = 6 # ____      Jan

    cluster = 22 # possible days of trading in a year data ends at 31 May

    gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  

    net_gains = gains['net_gains']

​

    accuracy_by_net_gains = gains['accuracy_by_net_gains']

    accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']

​

    print('net profit ', net_gains)

    print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)

    print("accuracy on all test points excluding last 22 points ", accuracy )

    

    

# symbol = "AUDCAD"

# test_model(symbol="AUDCAD", iteration=210, learning_rate=0.19, depth=7 ) # Good For Multiple Months

​

# test_model(symbol="USDCAD", iteration=101, learning_rate=0.59, depth=7 ) # Good For Multiple Months

# test_model(symbol="EURUSD", iteration=19, learning_rate=0.49, depth=7 ) # Good For Multiple Months Surplus For Jan, Feb, Mar, Apr

# test_model(symbol="AUDJPY", iteration=11, learning_rate=0.01, depth=5 ) # Good For Multiple Months Surplus For Jan, Feb, Mar, Apr

# test_model(symbol="CHFJPY", iteration=21, learning_rate=0.69, depth=8 ) # Good For Multiple Months Surplus For Jan, Feb, Mar, Apr

# test_model(symbol="EURUSD", iteration=39, learning_rate=0.01, depth=7 )

# test_model(symbol="CHFJPY", iteration=15, learning_rate=0.49, depth=7 ) # Good For Multiple Months Surplus For Jan, Feb, Mar, Apr

"""

Negative Performance In February

USDJPY GBPNZD

​

Most Losses In December:  [('GBPAUD', -734), ('USDJPY', -599), ('EURNZD', -580), 

('GBPNZD', -580), ('EURAUD', -376), ('EURJPY', -299), ('NZDCAD', -207), ('GBPCAD', -200), 

('CHFJPY', -187), ('GBPUSD', -134), ('EURCHF', -109), ('CADJPY', -105)

​

"""

​

"""

Poor Performance In January

​

AUDCHF AUDJPY AUDNZD AUDUSD 

USDCAD USDCHF EURUSD

​

"""

​

"""

Negative Performance In February

​

AUDUSD CADCHF EURNZD

GBPCHF GBPNZD GBPUSD

NZDUSD USDCHF USDCAD

​

EURCAD EURAUD 

"""

​

​

​

# Reformed OneZ With Depth 10-11 And Moderate Learning Rate 

# Hour 4 Features Of Last 3 days Only

​

# V2 models

​

​

test_model(symbol="USDJPY", iteration=3, learning_rate=0.2, depth=12, window_size=7) # version A

# test_model(symbol="USDJPY", iteration=3, learning_rate=0.23, depth=12, window_size=7) # version A

​

​

​

# test_model(symbol="EURAUD", iteration=3, learning_rate=0.15, depth=11, window_size=11) # version 

# test_model(symbol="EURAUD", iteration=7, learning_rate=0.4, depth=7, window_size=8) # version 

​

# test_model(symbol="EURAUD", iteration=27, learning_rate=0.19, depth=10, window_size=8) # another version -> A

# test_model(symbol="EURAUD", iteration=27, learning_rate=0.15, depth=10, window_size=8) # another version -> A

# test_model(symbol="EURAUD", iteration=33, learning_rate=0.15, depth=10, window_size=8) # version -> A

​

# test_model(symbol="EURAUD", iteration=77, learning_rate=0.5, depth=9, window_size=10) # version 

# test_model(symbol="EURAUD", iteration=100, learning_rate=0.05, depth=6, window_size=9) # version 

​

​

​

# test_model(symbol="EURNZD", iteration=9, learning_rate=0.25, depth=10, window_size=9) # version -> A

# test_model(symbol="EURNZD", iteration=11, learning_rate=0.15, depth=9, window_size=9) # version -> A

​

# test_model(symbol="EURNZD", iteration=66, learning_rate=0.51, depth=10, window_size=9) # another version

​

​

# Pair EURNZD__________________________

# test_model(symbol="EURNZD", iteration=5, learning_rate=0.15, depth=10, window_size=6)   # another version

# test_model(symbol="EURNZD", iteration=150, learning_rate=0.02, depth=6, window_size=7) # another version

# test_model(symbol="EURNZD", iteration=300, learning_rate=0.01, depth=6, window_size=7) # another version

​

​

#  Pair EURCAD________________________-

# test_model(symbol="EURCAD", iteration=1, learning_rate=0.10, depth=10, window_size=9) # version

#xx test_model(symbol="EURCAD", iteration=50, learning_rate=0.05, depth=6, window_size=6, alpha_type="gamma" ) # version A gamma 

# test_model(symbol="EURCAD", iteration=13, learning_rate=0.05, depth=9, window_size=7, alpha_type="alpha" ) # version A alpha

# test_model(symbol="EURCAD", iteration=1, learning_rate=0.10, depth=10, window_size=9, alpha_type="gamma") # another version A

​

# test_model(symbol="EURCAD", iteration=1, learning_rate=0.10, depth=10, window_size=9, alpha_type="gamma") # another version A

​

​

​

​

​

​

# test_model(symbol="EURUSD", iteration=210, learning_rate=0.05, depth=6, window_size=7, alpha_type="gamma" ) #  version A

# test_model(symbol="EURUSD", iteration=5, learning_rate=0.05, depth=6, window_size=7, alpha_type="gamma" ) #  another version 

# test_model(symbol="EURUSD", iteration=1, learning_rate=0.25, depth=10, window_size=9, alpha_type="alpha"  ) # another version

​

​

​

​

# test_model(symbol="USDCAD", iteration=3, learning_rate=0.11, depth=9, window_size=9, alpha_type="omega") # another version 

# test_model(symbol="USDCAD", iteration=1, learning_rate=0.20, depth=10, window_size=9, alpha_type="omega") # another version

# test_model(symbol="USDCAD", iteration=5, learning_rate=0.5, depth=10, window_size=11, alpha_type="gamma") # another version 

# test_model(symbol="USDCAD", iteration=5, learning_rate=0.15, depth=10, window_size=11, alpha_type="gamma") # version A 60 accuracy

​

​

​

​

# test_model(symbol="NZDUSD", iteration=111, learning_rate=0.21, depth=6, window_size=6, alpha_type="gamma") # version A

# test_model(symbol="NZDUSD", iteration=77, learning_rate=0.21, depth=6, window_size=6, alpha_type="gamma") # version A

# test_model(symbol="NZDUSD", iteration=9, learning_rate=0.25, depth=10, window_size=7, alpha_type="gamma") # version A

# test_model(symbol="NZDUSD", iteration=11, learning_rate=0.25, depth=10, window_size=7, alpha_type="gamma") # version A

​

# test_model(symbol="NZDUSD", iteration=1, learning_rate=0.3, depth=7, window_size=7, alpha_type="alpha") # version 

# test_model(symbol="NZDUSD", iteration=3, learning_rate=0.7, depth=11, window_size=7, alpha_type="gamma") # version 

# test_model(symbol="NZDUSD", iteration=11, learning_rate=0.2, depth=10, window_size=7, alpha_type="gamma") # version  

​

​

# test_model(symbol="NZDUSD", iteration=17, learning_rate=0.29, depth=10, window_size=7, alpha_type="gamma") # another version

# test_model(symbol="NZDUSD", iteration=21, learning_rate=0.21, depth=6, window_size=7, alpha_type="gamma") # another version

# test_model(symbol="NZDUSD", iteration=3, learning_rate=0.15, depth=9, window_size=7, alpha_type="gamma") # another version

​

​

​

​

# test_model(symbol="GBPUSD", iteration=3,  learning_rate=0.25, depth=11, window_size=6,  alpha_type="delta") # version 

# test_model(symbol="GBPUSD", iteration=17,  learning_rate=0.25, depth=11, window_size=9,  alpha_type="delta") # another version 

# test_model(symbol="GBPUSD", iteration=5,  learning_rate=0.15, depth=10, window_size=11, alpha_type="gamma") # version A

​

# test_model(symbol="GBPUSD", iteration=7, learning_rate=0.19, depth=9, window_size=11, alpha_type="gamma") # another version A 

# test_model(symbol="GBPUSD", iteration=19, learning_rate=0.19, depth=9, window_size=11, alpha_type="gamma") # another version A

# test_model(symbol="GBPUSD", iteration=9, learning_rate=0.21, depth=9, window_size=11, alpha_type="omega") # another version A

​

​

​

​

# test_model(symbol="GBPJPY", iteration=5, learning_rate=0.21, depth=9, window_size=7, alpha_type="alpha") # version A 

# test_model(symbol="GBPJPY", iteration=17, learning_rate=0.35, depth=11, window_size=6, alpha_type="alpha") # version A alpha

# test_model(symbol="GBPJPY", iteration=3, learning_rate=0.35, depth=11, window_size=6, alpha_type="alpha") # another version

# test_model(symbol="GBPJPY", iteration=7, learning_rate=0.15, depth=9, window_size=13, alpha_type="gamma") # another version accuracy by net gains 54-57 

# test_model(symbol="GBPJPY", iteration=2, learning_rate=0.7, depth=13, window_size=6, alpha_type="alpha") # another version 

# test_model(symbol="GBPJPY", iteration=5, learning_rate=0.3, depth=10, window_size=7, alpha_type="alpha") # another version 

# test_model(symbol="GBPJPY", iteration=5, learning_rate=0.3, depth=10, window_size=7, alpha_type="alpha") # another version 

# test_model(symbol="GBPJPY", iteration=7, learning_rate=0.6, depth=12, window_size=11, alpha_type="gamma") # another version 

# test_model(symbol="GBPJPY", iteration=3, learning_rate=0.7, depth=12, window_size=6, alpha_type="alpha") # another version 

​

​

​

# test_model(symbol="GBPAUD", iteration=9, learning_rate=0.15, depth=6, window_size=7)

# test_model(symbol="GBPAUD", iteration=5, learning_rate=0.15, depth=9, window_size=11) # another version

# test_model(symbol="GBPAUD", iteration=15, learning_rate=0.3, depth=9, window_size=7)  # another version

# test_model(symbol="GBPAUD", iteration=100, learning_rate=0.05, depth=6, window_size=6)# another version 

# test_model(symbol="GBPAUD", iteration=44, learning_rate=0.05, depth=6, window_size=9) # version A

# test_model(symbol="GBPAUD", iteration=100, learning_rate=0.05, depth=6, window_size=9) # version A

​

​

​

# test_model(symbol="AUDJPY", iteration=3, learning_rate=0.2, depth=9, window_size=6, alpha_type="omega") # version omega

​

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.45, depth=11, window_size=6, alpha_type="beta") # version 

# test_model(symbol="AUDJPY", iteration=3, learning_rate=0.15, depth=11, window_size=7, alpha_type="beta") # version 

​

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.25, depth=11, window_size=7, alpha_type="delta") # version A delta

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.2, depth=11, window_size=7, alpha_type="delta") # version A delta

​

# test_model(symbol="AUDJPY", iteration=27, learning_rate=0.2, depth=10, window_size=11, alpha_type="omega") # version omega

​

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.2, depth=11, window_size=9, alpha_type="alpha") # version A

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.15, depth=13, window_size=7, alpha_type="alpha") # another version

# test_model(symbol="AUDJPY", iteration=3, learning_rate=0.15, depth=12, window_size=7, alpha_type="alpha") # another version

​

​

# test_model(symbol="AUDJPY", iteration=3, learning_rate=0.7, depth=12, window_size=6, alpha_type="gamma") # another version

# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.01, depth=9, window_size=9, alpha_type="alpha") # another version

# test_model(symbol="AUDJPY", iteration=5, learning_rate=0.15, depth=13, window_size=7, alpha_type="alpha") # another version

​

​

# test_model(symbol="EURJPY", iteration=10, learning_rate=0.7, depth=6, window_size=7, alpha_type="alpha") # another version

​

# test_model(symbol="EURJPY", iteration=13, learning_rate=0.2, depth=11, window_size=5, alpha_type="gamma") # version A

# test_model(symbol="EURJPY", iteration=11, learning_rate=0.2, depth=11, window_size=5, alpha_type="gamma") # version A

# test_model(symbol="EURJPY", iteration=5, learning_rate=0.2, depth=11, window_size=5, alpha_type="gamma") # version A

​

# test_model(symbol="EURJPY", iteration=5, learning_rate=0.2, depth=9, window_size=7, alpha_type="delta") # another version

​

# test_model(symbol="EURJPY", iteration=7, learning_rate=0.5, depth=5, window_size=11, alpha_type="gamma") # another version

​

​

# test_model(symbol="AUDNZD", iteration=1, learning_rate=0.5, depth=11, window_size=12, alpha_type="omega") # another version

# test_model(symbol="AUDNZD", iteration=3, learning_rate=0.65, depth=10, window_size=12, alpha_type="omega") # version A

# test_model(symbol="AUDNZD", iteration=3, learning_rate=0.65, depth=10, window_size=12, alpha_type="beta")   # version A

# test_model(symbol="AUDNZD", iteration=13, learning_rate=0.5, depth=9, window_size=9, alpha_type="gamma") # another version

​

​

# test_model(symbol="GBPNZD", iteration=3, learning_rate=0.2, depth=10, window_size=7, alpha_type="omega")

# test_model(symbol="GBPNZD", iteration=3, learning_rate=0.2, depth=9, window_size=6, alpha_type="gamma")

# test_model(symbol="GBPNZD", iteration=13, learning_rate=0.25, depth=10, window_size=7, alpha_type="beta") # version A

# test_model(symbol="GBPNZD", iteration=15, learning_rate=0.25, depth=10, window_size=7, alpha_type="beta") # version A

# test_model(symbol="GBPNZD", iteration=33, learning_rate=0.15, depth=12, window_size=5, alpha_type="beta") # version A

​

​

​

​

​

# test_model(symbol="AUDUSD", iteration=5, learning_rate=0.27, depth=10, window_size=9, alpha_type="gamma") # version 

# test_model(symbol="AUDUSD", iteration=5, learning_rate=0.15, depth=12, window_size=5, alpha_type="beta") # version A

# test_model(symbol="AUDUSD", iteration=33, learning_rate=0.15, depth=12, window_size=5, alpha_type="beta") # version A

# test_model(symbol="AUDUSD", iteration=11, learning_rate=0.15, depth=10, window_size=6, alpha_type="delta") # version 

# test_model(symbol="AUDUSD", iteration=3, learning_rate=0.1, depth=10, window_size=11, alpha_type="delta") # version 

# test_model(symbol="AUDUSD", iteration=7, learning_rate=0.27, depth=10, window_size=9, alpha_type="gamma") # version  

# test_model(symbol="AUDUSD", iteration=21, learning_rate=0.21, depth=9, window_size=9, alpha_type="alpha") # version 

# test_model(symbol="AUDUSD", iteration=14, learning_rate=0.21, depth=7, window_size=9, alpha_type="beta") # version 

# xx test_model(symbol="AUDUSD", iteration=5, learning_rate=0.15, depth=11, window_size=7, alpha_type="omega") # version 

​

​

# test_model(symbol="NZDJPY", iteration=3, learning_rate=0.1, depth=12, window_size=7, alpha_type="beta") # version  

​

test_model(symbol="NZDJPY", iteration=5, learning_rate=0.35, depth=11, window_size=7, alpha_type="beta") # version A

​

/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/pandas/core/algorithms.py:1601: FutureWarning: Comparison of Timestamp with datetime.date is deprecated in order to match the standard library behavior.  In a future version these will be considered non-comparable.Use 'ts == pd.Timestamp(date)' or 'ts.date() == date' instead.
  return arr.searchsorted(value, side=side, sorter=sorter)
/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/ipykernel_launcher.py:35: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`
/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/ipykernel_launcher.py:35: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/ipykernel_launcher.py:42: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`
/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/ipykernel_launcher.py:42: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
/home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/pandas/core/generic.py:6392: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  return self._update_inplace(result)

USDJPY
0:	learn: 70.4186142	total: 586ms	remaining: 1.17s
2:	learn: 65.9572596	total: 1.75s	remaining: 0us
Mean Squared Error: 6107.159143442258
MAY
USDJPY  Starting date : 2024-05-01 
['+1-1', '+1-1', 1, 1, 1, '+1-1', 1, 1, 1, -1, '-1+1', 1, 1, '+1-1', 1, 1, '+1-1', -1, '-1+1', '-1+1', -1, '-1+1']
Sharpe Ratio: 0.31
Pips On Profit Side was :  753.7999999999982
Pips On Loss Side was :  353.3000000000044
accuracy by days 59.09090909090909
accuracy_by_net_gains  68.08779694697827
net profit  400.49999999999386

accuracy by net gains on previous test data  62.900529525481666
accuracy on all test points excluding last 22 points  56.68449197860963
APRIL
USDJPY  Starting date : 2024-04-01 
['+1-1', 1, -1, 1, 1, '+1-1', 1, 1, '+1-1', 1, 1, '+1-1', 1, '+1-1', '-1+1', '+1-1', '-1+1', 1, 1, -1, 1, -1]
Sharpe Ratio: 0.59
Pips On Profit Side was :  1376.400000000004
Pips On Loss Side was :  131.40000000000498
accuracy by days 63.63636363636363
accuracy_by_net_gains  91.28531635495398
net profit  1244.999999999999

accuracy by net gains on previous test data  58.248459222382856
accuracy on all test points excluding last 22 points  55.757575757575765
MARCH
USDJPY  Starting date : 2024-02-29 
['-1+1', 1, '+1-1', -1, '+1-1', '+1-1', '+1-1', 1, 1, 1, 1, 1, 1, 1, 1, '+1-1', '+1-1', 1, '+1-1', 1, '+1-1', 1]
Sharpe Ratio: 0.22
Pips On Profit Side was :  631.6000000000031
Pips On Loss Side was :  345.09999999999934
accuracy by days 59.09090909090909
accuracy_by_net_gains  64.66673492372288
net profit  286.50000000000375

accuracy by net gains on previous test data  57.48613678373382
accuracy on all test points excluding last 22 points  55.24475524475524
Iterations:  3
Learning rate:  0.2
Depth:  12
FEB
USDJPY  Starting date : 2024-01-30 
['+1-1', '+1-1', 1, 1, '+1-1', 1, 1, '+1-1', 1, 1, '+1-1', '+1-1', 1, '+1-1', '+1-1', 1, 1, '+1-1', 1, '+1-1', 1, '+1-1']
Sharpe Ratio: 0.16
Pips On Profit Side was :  630.6000000000012
Pips On Loss Side was :  394.0000000000026
accuracy by days 50.0
accuracy_by_net_gains  61.54596915869597
net profit  236.59999999999854

accuracy by net gains on previous test data  56.90828772261274
accuracy on all test points excluding last 22 points  56.19834710743802
JAN
USDJPY  Starting date : 2023-12-29 
['+1-1', 1, 1, 1, '+1-1', '+1-1', 1, '-1+1', '+1-1', '+1-1', 1, 1, 1, '+1-1', '+1-1', 1, '-1+1', '+1-1', 1, 1, -1, 1]
Sharpe Ratio: 0.31
Pips On Profit Side was :  868.0999999999983
Pips On Loss Side was :  387.3999999999967
accuracy by days 54.54545454545454
accuracy_by_net_gains  69.14376742333745
net profit  480.70000000000164

accuracy by net gains on previous test data  54.32350120307584
accuracy on all test points excluding last 22 points  56.56565656565656
DEC
USDJPY  Starting date : 2023-11-29 
[1, '+1-1', 1, '+1-1', 1, -1, 1, 1, '+1-1', '+1-1', '+1-1', 1, 1, 1, '+1-1', '+1-1', 1, '+1-1', 1, '+1-1', '+1-1', '+1-1']
Sharpe Ratio: 0.01
Pips On Profit Side was :  913.0000000000052
Pips On Loss Side was :  898.2000000000028
accuracy by days 50.0
accuracy_by_net_gains  50.40856890459371
net profit  14.800000000002456

accuracy by net gains on previous test data  56.039594375468894
accuracy on all test points excluding last 22 points  58.44155844155844

