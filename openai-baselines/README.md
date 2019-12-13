## Instructions
1. Setup the environment as specified in README-openai-baselines.

2. To run the experiments, edit run-experiment.sh to point to the log directory and the command below.
```
python run-experiment.sh [Name of the game, eg, Pong, Breakout] [Self-supervision task, e.g, rotation, sequence]
```
For more details about the commands in run-experiment.sh please refer to README-openai-baselines as we have used the same framework as theirs
and extend the algorithms with self supervision task in baselines/deepq-self-supervised.

3. The model and logs are saved in the root log directory with experiment name as the directory name.
