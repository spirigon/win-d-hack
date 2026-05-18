import sys, warnings, pickle
sys.path.insert(0, r'F:\Claude\win_d')
warnings.filterwarnings('ignore')

study = pickle.load(open(r'F:\Claude\win_d\models\optuna_v27_cf_study.pkl', 'rb'))
best = study.best_trial
print('Best trial:', best.number)
print('Fold-5:', best.user_attrs['fold5_nmae'])
print('Fold-4:', best.user_attrs['fold4_nmae'])
print('Params:', best.params)

top5 = sorted(study.trials, key=lambda t: t.value)[:5]
print('\nTop-5 trials:')
for t in top5:
    lr = t.params['learning_rate']
    leaves = t.params['num_leaves']
    f5 = t.user_attrs['fold5_nmae']
    print(f'  trial {t.number}: obj={t.value:.4f} f5={f5:.4f} lr={lr:.5f} leaves={leaves}')
