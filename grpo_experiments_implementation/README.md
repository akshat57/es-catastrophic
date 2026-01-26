## Training a model with GRPO

To train a model with GRPO, first download the [VeRl](https://verl.readthedocs.io/en/latest/start/install.html).

Ensure that the training and validation are discoverable by the script, along with the reward function. Run the script with bash as follows:

Ex:
```
bash grpo_llama_experiment.sh
```

If needed, adjust the number of GPUs required in the scripts by editing the following lines:
```
ray start --head --num-gpus=<number of gpus> --port=<port>
```
```
trainer.n_gpus_per_node=<number of gpus> \
```
And also tell the script what GPUs should be used by editing the following line:
```
export CUDA_VISIBLE_DEVICES= <include comma separated values of device names>
```