# Distributionally Robust Recourse Action (DiRRAc)

Source-code submission for paper Distributionally Robust Recourse Action - ICLR 2022

## 1. Install requirements
```
pip install -r requirements.txt
```

## 2. Experiments with synthetic data:

1. [Figure2](figure2.ipynb)

2. [Figure3](figure3.ipynb)

3. [Figure4](figure4.ipynb)

4. [Figure5](figure5.ipynb)

5. [Figure6](figure6.ipynb)

Results of each figure are saved in result/

## 3. Experiments with real-world data:

Generate recourse and evaluate on 3 different real-world datasets:

```
python train_real_data.py --num_samples <number of samples to evaluate> --save_dir <file name>
```

Example:
```
python train_real_data.py --num_samples 40 --save_dir real_data_40
```

Result of csv format is saved in result/real_data/
