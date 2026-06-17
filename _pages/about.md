---
permalink: /
title: "Home"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

[Short CV](https://www.dropbox.com/scl/fi/rpj3shfcvt3wsfqjbbmdj/main.pdf?rlkey=zlnq6nrlaanoajs23b9bwxn0l&st=3etzvxdt&dl=0)

## About me
I see mathematical statistics as a unifying framework in this time of disruption brought by AI. The principles of my work are abstraction and practicality, nowhere more evident than in clinical trials: the convergence point of medicine and data science at unprecedented speed. The field is innovating quietly but relentlessly, with sophisticated designs and endpoints. Causal inference has become the common language connecting data, evidence, and purpose in this delicate conversation; allowing regulators, statisticians, and clinical researchers to communicate with clarity, rigor, and confidence. While causal inference is often described as my research focus, I view it not as confined to a single area but as a thread running through the entire fabric of machine learning and statistics.


## News
- **Jun 27, 2026** — Just arrived in Nagoya, Japan and excited to attend the [2026 ISBA World Meeting](https://isba2026.github.io)
- **Jun 4, 2026** Happy to present [“Shrinkage through Multiple Identifiability”](https://arxiv.org/abs/2604.18430) at the [II International Workshop on Bayesian Statistics](https://www.cunef.edu/eventos/ii-international-workshop-on-bayesian-statistics/). 
- [“Neural interval-censored survival regression with feature selection”](https://onlinelibrary.wiley.com/doi/10.1002/sam.11704) was named one of Wiley’s most-viewed articles of 2025. 
- **May 13, 2026** [Interview](https://www.elconfidencial.com/tecnologia/ciencia/2026-05-13/pandemia-covid-hantavirus-matematicas-probabilidades_4354139/) in El Confidencial on pandemic risk, emerging infectious diseases and causal inference.  
- **Jan 26, 2026**: Visited IES Nuestra Señora de la Almudena as part of the NightMadrid scientific dissemination program.
- **Jan 14, 2026**: Delighted to join [Celera](https://acelerame.org) as as fellow, an initiative that supports exceptional young leaders.
- **Dec 2, 2025**: *Generative Invariance*, feat. [David Ríos Insua](https://www.davidriosinsua.es/), is now published in the [Electronic Journal of Statistics](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-19/issue-2/Domain-adaptation-under-hidden-confounding/10.1214/25-EJS2474.full). 
- **Nov 5, 2025**: Grateful to the Statistics Department at Universidad de Valladolid for having me as a speaker. The talk was great, and the company even better.
- **Sep 30, 2025**: Honored that [my entrepreneurial project](https://creative.variacle.com/) has been chosen by HealthStart, leading accelerator program in Spain dedicated to advancing innovative healthcare solutions.
- **Sep 26, 2025**: Took the stage at Espacio Fundación Telefónica for my talk [*The Mathematics of Electronic Music*](https://www.instagram.com/p/DPMIRmQDMLz/?igsh=MWJjZnNyb3lkZnlsdA==).
- **Sep 22, 2025**: [Mark van der Laan](https://vanderlaan-lab.org/about/) discusses our *implied interventions* approach to causal inference with instruments in his [interview](https://www.youtube.com/watch?v=qr5JolEAuJU&t=1498s) with [Aleksander Molak](https://alxndr.io/). 
- **Sep 1, 2025**: Delighted to begin a new lecturing semester in the 3rd year of the Environmental Sciences BSc program at Universidad Autónoma de Madrid. 
- **Apr 30, 2025**: Great energy and insightful feedback at my talk [*Causal inference via proxy interventions*](https://ctml.berkeley.edu/43025-seminar-causal-inference-proxy-interventions) at UC Berkeley’s CTML.
- **Feb 24, 2025**: Touchdown in Berkeley after a long 13-hour flight. Exciting times ahead at [CTML](https://ctml.berkeley.edu)!
- **Feb 11, 2025**: *Causal Survival Embeddings* has been published in *Statistical Methods in Medical Research*! Check it out: [https://doi.org/10.1177/09622802241311455](https://doi.org/10.1177/09622802241311455)
- **Dec 18, 2024**: Honored to receive the Institute of Mathematical Statistics Award in Nice for *Uncertainty quantification for intervals*.


![My Image](/images/pizarra.png)
![My Image](/images/hal_design.png)
![My Image](/images/telefonica.png)

```r
remotes::install_github("meixide/hapc")
library(hapc)
rescv <- cv.hapc(X, Y,
                 npcs = n-1,
                 log_lambda_min =-6,
                 log_lambda_max = -1,
                 norm = "1",
                 max_degree=d,
                 predict=Xnew
)
```

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


