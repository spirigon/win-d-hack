import sys, pickle
sys.path.insert(0, r'F:\Claude\win_d')
study = pickle.load(open(r'F:\Claude\win_d\models\optuna_v27_cf_study.pkl', 'rb'))
t23 = next(t for t in study.trials if t.number == 23)
print('Trial 23 params:')
for k, v in t23.params.items():
    print(f'  {k}: {v!r}')
