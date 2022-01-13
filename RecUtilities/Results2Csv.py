import os
import pandas as pd

def results_to_csv(exp_dict):
    exp_results_pd = pd.DataFrame(pd.Series(exp_dict)).transpose()
    if not os.path.isfile('results/' + exp_dict['model'] + '.csv'):
        exp_results_pd.to_csv('results/' + exp_dict['model'] + '.csv', header=True, index=False,
                              columns=list(exp_dict.keys()))
    else:
        exp_results_pd.to_csv('results/' + exp_dict['model'] + '.csv', mode='a', header=False, index=False,
                              columns=list(exp_dict.keys()))