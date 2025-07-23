## Dataset Preparation

Please copy the dataset directory to the folder `./Data` in our repository.

### Training & Sampling

For training, you can reproduce the experimental results of all benchmarks by runing

~~~bash
python main.py --name {experiment_name} --config_file {config.yaml} --gpu {gpu_id} --train
~~~

Config files are located in Config directory.

#### Unconstrained
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
```

#### Imputation
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 1 --milestone {checkpoint_number} --mode infill --missing_ratio {missing_ratio}
```

#### Forecasting
```bash
(myenv) $ python main.py --name {dataset_name} --config_file {config.yaml} --gpu 0 --sample 1 --milestone {checkpoint_number} --mode predict --pred_len {pred_len}
```

This paper has been submitted for peer review and is currently under review.
