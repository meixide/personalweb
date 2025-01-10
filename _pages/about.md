---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I see mathematical statistics as a unifying framework in this time of disruption brought by AI. While causal inference is often described as my research focus, I view it not as confined to a single area but as a thread running through the entire fabric of machine learning and statistics. My passion lies in deriving useful, abstract and elegant results that address data-driven problems arising in clinical medicine and epidemiology, fields undergoing their own quiet revolution.

![My Image](/images/pizarra.png)

```python
class GenerativeInvariance(Estimator):
    def __init__(self, intercept=True):
        self.intercept = intercept

    def fit(self, data, source, target):
    # ...
    
    def predict(self, x_new):
        x_mean = np.mean(x_new, axis=0)
        x_centered = x_new - x_mean

        cov_xnew = np.cov(x_new, rowvar=False)
        cov_inv = np.linalg.pinv(cov_xnew)
     
        epsy = x_centered @ cov_inv @ self.khat[1:]

        y_pred = self.betahat[0] + x_new @ self.betahat[1:] + epsy

        return y_pred

    def __str__(self):
        return self.__class__.__name__
```


