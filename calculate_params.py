from EncDecAD import EncDecAD
import sys
import pickle

filename = 'model/' + sys.argv[1] 
print('loading file path:', filename)
model = EncDecAD()
model.load_model(filename)

# calculate gaussian params
print('calculate gaussian params from test_source...')
mu, sigma = model.calc_gaussian_params()
print("μ :", mu)
print("sigma: ", sigma)
save_params = [['mu', 'sigma'], [mu, sigma]]

# dump
with open('data/params.pickle', 'wb') as f:
    pickle.dump(save_params, f)
