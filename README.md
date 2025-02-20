## How To Train Model
1. Run on terminal  `pip install -r requirements.txt`
2. Set the model that you want to run on the `hyperparameters.yaml`. As well as editing hyperparameters
3. Run the JSON script on `.vscode\launch.json`. Optionally, use the command `python agent.py <model> --train` (e.g. `python agent.py flappybird1 --train`)
4. Optional for checkpoint, put the directory on the checkpoint field on `hyperparameters.yaml`

## How to Test Model
1. Run the command `python agent.py <model> --render` (e.g. `python agent.py flappybird1 --render`)

Note: Testing from checkpoint models are still being developed
