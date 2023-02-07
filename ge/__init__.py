import pandas as pd
import numpy as np
import pathlib
import feyn
import sklearn.linear_model
import scipy.stats

from tqdm import trange

_datasets = {
    path.name.split(".")[0]: path for path in pathlib.Path("data/reg_num/").glob("*.csv")
}

_cache = {}
datasets = list(_datasets.keys())


def get_dataset(name: str, n_features: int = None):
    """
    Load a dataset from the tabular-benchmark repository.

    Parameters:
    name (str): The name of the dataset to load.

    Returns:
    pandas.DataFrame: The dataset.
    """
    if name in _cache:
        res = _cache[name]
    else:
        res = pd.read_csv(_datasets[name]).drop(columns=["Unnamed: 0"])
        _cache[name] = res

    if n_features is not None:
        # Find the n_features most correlated variables with the target
        vars = res.corrwith(res.iloc[:,-1]).sort_values(ascending=False)[0:n_features+1].sort_values().index.to_list()
        return res[vars]

    return res

def count_outliers(df, sensitivity = 1.5):

    res = {}

    for varname in df.columns:
        s = df[varname]
        if s.dtype == "float":
            q1, q3 = s.quantile([0.25, 0.75])
            lower_lim = q1 - sensitivity * (q3 - q1)
            upper_lim = q3 + sensitivity * (q3 - q1)
            
            res[varname] = ((s < lower_lim) | (s > upper_lim)).sum()
    return res

    
def remove_outliers(df, columns = None, sensitivity = 1.5):
    df = df.copy()
    if columns is None:
        columns = df.columns

    for varname in df.columns:
        s = df[varname]
        if s.dtype == "float":
            q1, q3 = s.quantile([0.25, 0.75])
            lower_lim = q1 - sensitivity * (q3 - q1)
            upper_lim = q3 + sensitivity * (q3 - q1)
            
            df = df[(s >= lower_lim) & (s <= upper_lim)]
            df[varname] = s

    return df


def bootstrap(data: pd.DataFrame, statistic, num_samples: int):
    """
    Calculate the confidence interval for a given statistic using bootstrapping

    Parameters:
    data (pandas.DataFrame): The data to be used for the calculation.
    statistic (function): The statistic to be calculated on the bootstrapped samples.
    num_samples (int): The number of bootstrapped samples to generate.

    Returns:
    float: the z score of the difference between the true and the estiamted statistic.
    """
    n = len(data)
    bootstrapped_statistics = np.zeros(num_samples)
    for i in range(num_samples):
        bsample = data.sample(n, replace=True)
        bootstrapped_statistics[i] = statistic(bsample)
    return bootstrapped_statistics.std()



def measure_generalization_error(population: pd.DataFrame, iterations:int=50, sample_size:int=100):
    res = []

    target = population.columns[-1]
    for _ in trange(iterations):
        # Take a sample
        sample = population.sample(n=sample_size, replace=True)
        
        # Learn a model on the sample
        model = feyn.reference.SKLearnRegressor(sklearn.linear_model.LinearRegression, sample, target)
        
        # Compute the true R2 of the model
        true_r2 = model.r2_score(population)
        
        # Compute the point estimate of the R2 of the model
        point_r2 = model.r2_score(sample)
        
        loss_mae_std = feyn.losses.absolute_error(sample[target], model.predict(sample)).std()

        res.append({
            "n_features": len(population.columns)-1, 
            "loss_mse_std": feyn.losses.squared_error(sample[target], model.predict(sample)).std(),
            "target_mean": sample[target].mean(),
            "target_std": sample[target].std(),
            "loss_mae_std": loss_mae_std,
            "loss_mae_std_norm": loss_mae_std/sample[target].std(),
            "boot_r2_std": bootstrap(sample, lambda d: model.r2_score(d), 300),
            "target_normality": scipy.stats.shapiro(sample[target])[1],
            "f1_normality": scipy.stats.shapiro(sample[sample.columns[0]])[1],
            "r2": point_r2, 
            "generalization_error": point_r2 - true_r2
        })

    return pd.DataFrame(res)