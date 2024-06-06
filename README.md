# DISEE

Python 3.8.3 and Pytorch 1.12.1 implementation of the latent Signed relational Latent dIstance Model (SLIM).

## Description


Understanding the structure and dynamics of scientific research, i.e., the science of science
(SciSci), has become an important area of research in order to address imminent questions including how scholars interact to advance science, how disciplines are related and evolve,
and how research impact can be quantified
and predicted. Central to the study of SciSci
has been the analysis of citation networks.
Here, two prominent modeling methodologies have been employed: one is to assess
the citation impact dynamics of papers using
parametric distributions, and the other is to
embed the citation networks in a latent space
optimal for characterizing the static relations
between papers in terms of their citations. Interestingly, citation networks are a prominent
example of single-event dynamic networks,
i.e., networks for which each dyad only has
a single event (i.e., the point in time of citation). We presently propose a novel likeli-
hood function for the characterization of such
single-event networks. Using this likelihood,
we propose the Dynamic Impact Single-Event
Embedding model (DISEE). The DISEE
model characterizes the scientific interactions
in terms of a latent distance model in which
random effects account for citation heterogeneity while the time-varying impact is characterized using existing parametric representations for assessment of dynamic impact. We
highlight the proposed approach on several
real citation networks finding that the DISEE
well reconciles static latent distance network
embedding approaches with classical dynamic
impact assessments.
## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```


## Reference

[Time to Cite: Modeling Citation Networks using the Dynamic Impact Single-Event Embedding Model](https://proceedings.mlr.press/v238/nakis24a.html), Nikolaos Nakis et al., AISTATS 24



