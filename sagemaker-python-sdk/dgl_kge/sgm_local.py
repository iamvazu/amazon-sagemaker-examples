#!/usr/bin/env python
# coding: utf-8
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig, rule_configs



role = get_execution_role()

hyperparameters = {'random_seed': True, 'num_steps': 50, 'epochs': 5,
                   'data_dir':'/tmp/pytorch-smdebug'}

hyperparameters=dict(
    dataset = 'FB15k',
    model = 'DistMult',
    batch_size = 1024,
    neg_sample_size = 256,
    hidden_dim = 2000,
    gamma = 500.0,
    lr = 0.1,
    max_step = 100000,
    batch_size_eval = 16,
    valid = True,
    test = True,
    log_interval=10,
    neg_adversarial_sampling = True
)
task_tags = [{'Key':'ML Task', 'Value':'DGL'}]

#rules = [
#    Rule.sagemaker(rule_configs.loss_not_decreasing())
#]
# Choose a built-in rule to monitor your training job
rules = [Rule.sagemaker(
    rule_configs.exploding_tensor(),
    # configure your rule if applicable
    rule_parameters={"tensor_regex": ".*"},
    # specify collections to save for processing your rule
    collections_to_save=[
        CollectionConfig(name="weights"),
        CollectionConfig(name="losses"),
    ],
)]

debugger_hook_config = DebuggerHookConfig(s3_output_path="/opt/ml/smdebug/", container_local_output_path="/opt/ml/smdebug/")

estimator = PyTorch(
                  entry_point='train.py',
                  source_dir="./",
                  role=role,
                  train_instance_count=1,
                  train_instance_type='local_gpu',
                  #train_volume_size=400,
                  #train_max_run=3600,
                  hyperparameters=hyperparameters,
                  tags=task_tags,
                  framework_version='1.3.1',
                  py_version='py3',
                  rules = rules,
                  debugger_hook_config = debugger_hook_config
#                  output_path="/opt/ml/smdebug/"
                 )



estimator.fit(wait=True)

