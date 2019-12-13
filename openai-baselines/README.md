## Instructions
1. Setup the environment as specified in [README-openai-baselines](https://github.com/prabhat1081/self-supervision-cs221/blob/master/openai-baselines/README-openai-baselines.md).

2. To run the experiments, edit run-experiment.sh to point to the log directory and the command below.
```
python run-experiment.sh [Name of the game, eg, Pong, Breakout] [Self-supervision task, e.g, rotation, sequence]
```
For more details about the commands in run-experiment.sh please refer to [README-openai-baselines](https://github.com/prabhat1081/self-supervision-cs221/blob/master/openai-baselines/README-openai-baselines.md) as we have used the same framework as theirs
and extend the algorithms with self supervision task in [baselines/deepq_self_supervised](https://github.com/prabhat1081/self-supervision-cs221/tree/master/openai-baselines/baselines/deepq_self_supervised).

3. The model and logs are saved in the root log directory with experiment name as the directory name.
