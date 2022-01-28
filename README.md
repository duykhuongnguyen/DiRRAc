# Distributionally Robust Recourse Action (DiRRAc)

Source-code submission for paper Distributionally Robust Recourse Action (DiRRAc) - ICML 2022

## 1. Install requirements
```
pip install -r requirements.txt
```

## 2. Experiments with synthetic data:

1. Feasible set of DiRRAc: [Figure2](figure2.ipynb)

2. Comparison of DiRRAc and ROAR: [Figure3](figure3.ipynb)

3. Impact of magnitude of distribution shifts to the empirical validity: [Figure4](figure4.ipynb)

4. Different types of data distribution shifts: [Figure5](figure5.ipynb)

5. Model parameters in 2D: [Figure6](figure6.ipynb)

6. Impact of varying parameters of DiRRAc: [Figure7](figure7.ipynb)

7. Cost of Robustness of DiRRAc: [Figure9](figure9.ipynb)

Results of each figure are saved in result/

## 3. Experiments with real-world data:

Generate recourse and evaluate on 3 different real-world datasets:

```
python train_real_data.py --mode linear --num_samples <number of samples to evaluate>
```

Example:
```
python train_real_data.py --mode linear --num_samples 40
```

Experiments with prior on the covariance matrix:

```
python train_real_data.py --mode linear --num_samples <number of samples to evaluate> --sigma_identity True
```

Example:
```
python train_real_data.py --mode linear --num_samples 40 --sigma_identity True
```

Result of csv format is saved in result/real_data/

## 4. Experiments with non-linear model:

Generate recourse for non-linear model and evaluate on 3 different real-world datasets:

```
python train_real_data.py --mode non --num_samples <number of samples to evaluate>
```

Example:
```
python train_real_data.py --mode non --num_samples 20
```
