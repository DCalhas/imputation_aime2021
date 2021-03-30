# fMRI Multiple Missing Values Imputation Regularized by a Recurrent Denoiser

Github repository for [AIME 2021](https://aime21.aimedicine.info/) article.

Incremental Missing Values Imputation on fMRI volumes, a complete data method. The paper can be found [here](https://arxiv.org/pdf/2009.12602.pdf).

Proposed layer implementation can be found [here](https://github.com/DCalhas/missing_imputation/blob/8b2a2a6f6a4540d2fa104fc94d94450f71cdbfa6/src/utils/layers_utils.py#L312)

To replicate the results please run the [notebook](https://github.com/DCalhas/multiple_imputation_aime2021/blob/master/src/notebooks/mae_mse_rmse_models.ipynb)

## Requirements:
- tensorflow_cpu==2.1.0
- nilearn==0.6.2
- mne==0.20.5
- GPy==1.9.9
- GPyOpt==1.2.6
- matplotlib==3.2.1
- sklearn==0.23.1
- scipy==1.4.1

Please create an anaconda enviromnent and install the packages in the requirements.txt file with:
`$ pip install -r requirements.txt`

## Datasets Used:
- [EEG, fMRI and NODDI dataset](https://osf.io/94c5t/)
- [Auditory and Visual Oddball EEG-fMRI](https://legacy.openfmri.org/dataset/ds000116/)