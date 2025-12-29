---
permalink: /
title: "Home"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

[Short CV](https://www.dropbox.com/scl/fi/oxbyvrctfj263mg28eqit/main.pdf?rlkey=41k6uh1x101at7iquemnibty6&dl=0)

## About me
I see mathematical statistics as a unifying framework in this time of disruption brought by AI. The principles of my work are abstraction and utility, nowhere more evident than in clinical trials, where medicine and data science converge at unprecedented speed. The field is innovating quietly but relentlessly, with cutting-edge designs and endpoints. Causal inference has become the common language connecting data, evidence, and purpose in this delicate conversation; allowing regulators, statisticians, and clinical researchers to communicate with clarity, rigor, and confidence. While causal inference is often described as my research focus, I view it not as confined to a single area but as a thread running through the entire fabric of machine learning and statistics.


## News
- **Dec 2, 2025**: *Generative Invariance* (feat. [David Ríos](https://www.davidriosinsua.es/)) is now published in the [Electronic Journal of Statistics](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-19/issue-2/Domain-adaptation-under-hidden-confounding/10.1214/25-EJS2474.full). 
- **Nov 5, 2025**: Grateful to the Statistics Department at Universidad de Valladolid for having me as a speaker. The talk was great, and the company even better. 
- **Sep 26, 2025**: Took the stage at Espacio Fundación Telefónica for my talk [*The Mathematics of Electronic Music*](https://www.instagram.com/p/DPMIRmQDMLz/?igsh=MWJjZnNyb3lkZnlsdA==).
- **Sep 22, 2025**: [Mark van der Laan](https://vanderlaan-lab.org/about/) discusses our *implied interventions* approach to causal inference with instruments in his [interview](https://www.youtube.com/watch?v=qr5JolEAuJU&t=1498s) with [Aleksander Molak](https://alxndr.io/). 
- **Sep 1, 2025**: Delighted to begin a new lecturing semester in the 3rd year of the Environmental Sciences BSc program at Universidad Autónoma de Madrid. 
- **Apr 30, 2025**: Great energy and insightful feedback at my talk [*Causal inference via proxy interventions*](https://ctml.berkeley.edu/43025-seminar-causal-inference-proxy-interventions) at UC Berkeley’s CTML.
- **Feb 24, 2025**: Touchdown in Berkeley after a long 13-hour flight. Exciting times ahead at [CTML](https://ctml.berkeley.edu)!
- **Feb 11, 2025**: *Causal Survival Embeddings* has been published in *Statistical Methods in Medical Research*! Check it out: [https://doi.org/10.1177/09622802241311455](https://doi.org/10.1177/09622802241311455)
- **Dec 18, 2024**: Honored to receive the Institute of Mathematical Statistics Award in Nice for *Uncertainty quantification for intervals*.


![My Image](/images/pizarra.png)
![My Image](/images/telefonica.png)
![My Image](/images/hal_design.png)

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


