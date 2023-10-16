# Air Hockey Challenge Framework
## Training Scripts and Testing Scripts
Everything related to the training and testing process of agents is under `step_based_ik/`.

To start a new training `config.yaml`, you have to configure the parameters in `config.yaml`:
- `Group`, `job_type` and `name` are used `wandb` to categorize the run. You have to set a unique combination of 3 for each new run.
- `Training` specifies the environment that you want to train in. The possible values for `env` can be found in `fancy_gym/envs/__init__.py`. If you implement a new environment, you have to register it in this file. Also, you should add an import to `fancy_gym/envs/air_hockey/__init__.py` for your new environment.
- Under `hyperparameters` you can set the hyperparameters of the run. Possible values are documented in Stable Baselines 3.
- `Wandb/mode` decides if your run will be saved remotely. You can disable this by setting it to `disabled` or enable it by setting it to `online`.

To start the training, you can simply run:
```
# To run directly
python run.py
# To run in slurm
python run.py slurm
```

After the training starts, it should log the results in `logs/` directory. Each test epoch, the agent is tested and saved as best model if its reward is higher than the best reward so far. Separately, the agent is saved each checkpoint epoch. Test and checkpoint frequencies can be configured in `ppo_air_hockey_benchmark/train.py` (`checkpoint_callback` and `custom_eval_callback`).

To test an agent, you have to configure it again in `config.yaml`, `local_testing`. Simply give the path to the agent and other relevant parameters. The best saved result should be displayed on your screen after running:
```
python run.py --test
```

# Configuring Environments
The air hockey environments in fancy_gym use the challenge environments as the underlying environment. In order to modify things such as initial puck positions, or initial robot configuration, you have to find the relevant files and lines under `air_hockey/air_hockey_challenge/environments/iiwas` and modify according to your wishes.