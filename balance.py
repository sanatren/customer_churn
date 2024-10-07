"""""
#balancing the exited columns 
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state = 42)
X_resampled , Y_resampled = oversampler.fit_resample(X,Y)

print(pd.Series(Y_resampled).value_counts())
"""