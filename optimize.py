import optuna

from trainer import Trainer

def objective(trial):
    iters = trial.suggest_int('iters', 100, 5000)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    depth = trial.suggest_int('depth', 2, 7)
    # use_ranker = trial.suggest_categorical('use_ranker', [True, False])
    model_params = {
        "iters": iters, "lr": lr, "depth": depth,
    }
    return Trainer(model_params=model_params, use_ranker=True, verbose=False).run()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(study.best_trial, study.best_params)
