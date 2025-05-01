---
permalink: /
title: "Home"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

- [Teaching](/teaching/)
- [Publications](/publications/)
- [Gallery](/gallery/)

- [Short CV](https://www.dropbox.com/scl/fi/oxbyvrctfj263mg28eqit/main.pdf?rlkey=41k6uh1x101at7iquemnibty6&dl=0)
- [LinkedIn](https://es.linkedin.com/in/carlos-garcía-meixide-b8ba4a1a8)




## About me
I see mathematical statistics as a unifying framework in this time of disruption brought by AI. While causal inference is often described as my research focus, I view it not as confined to a single area but as a thread running through the entire fabric of machine learning and statistics. My passion lies in deriving useful, abstract and elegant results that address data-driven problems arising in clinical medicine and epidemiology, fields undergoing their own quiet revolution. 

## News
- **Apr 30, 2025**: Great energy and insightful feedback at my talk [*Causal inference via proxy interventions*](https://ctml.berkeley.edu/43025-seminar-causal-inference-proxy-interventions) at UC Berkeley’s CTML.
- **Feb 24, 2025**: Touchdown in Berkeley after a long 13-hour flight. Exciting times ahead at [CTML](https://ctml.berkeley.edu)!
- **Feb 11, 2025**: *Causal Survival Embeddings* has been published in *Statistical Methods in Medical Research*! Check it out: [https://doi.org/10.1177/09622802241311455](https://doi.org/10.1177/09622802241311455)
- **Dec 18, 2024**: Honored to receive the Institute of Mathematical Statistics Award in Nice for *Uncertainty quantification for intervals*. 


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


