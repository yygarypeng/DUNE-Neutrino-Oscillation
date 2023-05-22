# Neutrino-Oscillation-DUNE
## 2023 NTHU PHYS591000 AI lecture (final project)
## Neutrino Oscillation

### Description

Task: Regression of parameters [ $\Theta_{23}$, $\delta_{cp}$ ]
You will use neural network to build a regressor for this Regression Task.

From neutrino physics, we know that PMNS matrix connects flavor eigenstates to mass eigenstates.
The future long baseline neutrino experiment- Deep Underground Neutrino Experiment (DUNE) has chance to pin down the $sin\theta_{23}$-octant degeneracy and discover the potential CP violation in the leptonic sector. This experiment will search for the CP violation in the leptonic sector and conduct precision measurements using appearance and disappearance channels by muon neutrino and anti muon neutrino.

Instead using traditional chi-square fit to estimate $\Theta_{23}$, $\delta_{cp}$, you need to build a neural network model to estimate these two parameters from four appearance and disappearance spectrum.

We will provide smooth ideal spectrum based on DUNE to be training dataset.
The test dataset is the pseudo data which is considered statistical uncertainty (poisson fluctuation) for each bin to be experimental-like data.
We hide truth parameters of test dataset to evaluate your model performance.
For further data and submission information, please go to Data page.

ref1: [Neutrino Oscillation on WIki](https://en.wikipedia.org/wiki/Neutrino_oscillation)   
ref2: [Neutrino Group on SLAC](https://sites.slac.stanford.edu/neutrino/research/neutrino-oscillations)   
ref3: [PHYSICS OF NEUTRINO OSCILLATION](https://arxiv.org/pdf/1511.06752.pdf).   


### Evaluation

`Root mean square error` will be metric for evaluation of your model.

`Root mean square error`:

$\sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{d_i -f_i}{\sigma_i}\Big)^2}}$

You need to prepare a csv file for submission.
It should contain two columns: id and prediction.
Total length will be 2000.
For top 1000 rows, you should fill predicted cp phase in degree.
For last 1000 rows, you should fill predicted theta23 in degree.

### Dataset Description
neutrino_training_data
> You will use this data to train your model.

This file is in npz format.
There are 1,000,000 ideal energy spectrum.
It contains:

* ve: stands for electron neutrino energy spectrum [Unit: Events/per Energy bin]
* vebar: stands for anti electron neutrino energy spectrum [Unit: Events/per Energy bin]
* vu: stands for muon neutrino energy spectrum [Unit: Events/per Energy bin]
* vubar: stands for anti muon neutrino energy spectrum [Unit: Events/per Energy bin]
* theta23: oscillation mixing angle: $\Theta_{23}$ [Unit: degree]
delta: $\delta_{cp}$: cp phase [Unit: degree]
* ldm: mass ordering: $\Delta m_{31}^2$ [Unit: GeV]

The energy starts from 0.625 GeV to 8.000 GeV with energy bin size 0.125 GeV and 9, 10, 12, 14 and 16 GeV.
Total energy bin is 65 bins.
All ideal simulation are based on the Deep Underground Neutrino Experiment (DUNE).

### neutrino_test_data
You will use this data to test your model for submission.

This is the test file for your model.
This file is in `npz` format.
There are 1,000 pseudo experiment energy spectrum.
It contains:

* ve: stands for electron neutrino energy spectrum [Unit: Events/per Energy bin]
* vebar: stands for anti electron neutrino energy spectrum [Unit: Events/per Energy bin]
* vu: stands for muon neutrino energy spectrum [Unit: Events/per Energy bin]
* vubar: stands for anti muon neutrino energy spectrum [Unit: Events/per Energy bin]

The energy starts from 0.625 GeV to 8.000 GeV with energy bin size 0.125 GeV and 9, 10, 12, 14 and 16 GeV.
Total energy bin is 65 bins.
The pseudo experiment means we only consider statistic fluctuations (poisson fluctuation for each bin) on ideal spectrum.

Submission Template: prediction_template.csv

> This is a template for submission.
> The file should be like this format.

This is a `csv` template for submission.
It contains `id` and `prediction`.
Total length will be 2000.
For top 1000 rows, you will fill predicted cp phase in degree.

&copy; COPYRIGHT:    
The description and evaluation are retrieved from     
Kaggle: https://www.kaggle.com/competitions/phys591000-2023-final-project-i
