# Informative Semi-factuals for XAI

![alt text](informative_sf.png "Title")



This is the repository for the paper *"Informative Semi-Factuals for XAI: The Elaborated Explanations that People Prefer*.


Semi-factual or *Even if* explanations explains how a predicted outcome *can remain the same* even when certain input-features are altered. For example, in the commonly-used banking app scenario, a semi-factual explanation could inform customers about better options, other alternatives for their successful application, by saying "Even if you asked for double the loan amount, you would still be accepted". Most semi-factuals XAI algorithms focus on finding maximal value-changes to a single key-feature that do not alter the outcome (unlike counterfactual explanations that often find minimal value-changes to several features that alter the outcome). However, no current semi-factual method explains *why* these extreme value-changes do not alter outcomes; for example, a more informative semi-factual could tell the customer that it is their good credit score that allows them to borrow double their requested loan. This work advances a new algorithm -- the *Informative Semi-factuals* (ISF) method -- that generates more elaborated explanations supplementing semi-factuals with information about additional *hidden features* that influence an automated decision.

--------------

## Experiments

To reproduce these results run:

```
conda create --name semifactual
conda activate semifactual
conda install -c anaconda pandas
conda install -c anaconda seaborn
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge jsonschema
conda install -c conda-forge imbalanced-learn
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge tensorboard
conda install -c conda-forge cvxpy
```