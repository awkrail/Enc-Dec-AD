from EncDecAD import EncDecAD
import sys
import pandas as pd

#filename = 'model/' + sys.argv[1] 
#print('loading file path:', filename)
model = EncDecAD()
#model.load_model(filename)

# calculate gaussian params
print('calculate gaussian params from test_source...')
mu, sigma = model.calc_gaussian_params()
save_params = [['mu', 'sigma'], [mu, sigma]]
params_df = pd.DataFrame(save_params)
params_df.to_csv('data/params.csv')
