---
title: "Causal inference via implied interventions"
collection: publications
category: manuscripts
permalink: /publication/ciii
excerpt: 'with Mark van der Laan'
date: 2024-02-23
venue: 'arXiv'
paperurl: 'https://arxiv.org/abs/2506.21501'
#citation: 'Your Name, You. (2024). &quot;Paper Title Number 3.&quot; <i>GitHub Journal of Bugs</i>. 1(3).'
---

In the context of having an instrumental variable, the standard practice in causal inference begins by targeting an effect of interest and proceeds by formulating assumptions enabling its identification. We turn this around by simply not making assumptions anymore and adhere to the interventions we can identify, rather than starting with a desired causal estimand and imposing untestable conditions. The randomization of an instrument and its exclusion restriction define a class of auxiliary stochastic interventions on the treatment that are implied by stochastic interventions on the instrument. This mapping characterizes the identifiable causal effects of the treatment on the outcome given the observable distribution, leading to an explicit transparent G-computation formula under hidden confounding. Alternatively, searching for an intervention on the instrument whose implied one best approximates a desired target naturally leads to a projection representing the closest identifiable treatment effect. The generality of this projection allows to select different norms and indexing functional sets that give rise to diverse estimation problems, some of which we address using Expectation-Maximization and the Highly Adaptive Lasso. Our approach interprets instrumental-variables–based causal inference as a projection onto the effect space identifiable from the observed data distribution, rather than relying on potentially risky assumptions. 
