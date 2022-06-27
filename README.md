# RNN and CNN hyperparameter optimization with genetic algorithms using TF and DEAP

> This is part of my Thesis work. Some ideas were based on the book 'Hands-On Genetic Algorithms with Python' by Eyal Wirsansky. You can use it for educational purpose.

## Setting up evironment and programm execution

In order to run this program you need to:

1. Create venv from `environment.yml` file. It's recommended to use `conda`:

```bash
  conda env create -f environment.yml
```

2. Activate this environment:

```bash
  conda activate ./venv
```

3. Define experiment by setting type of testing model (RRN or CNN) and datasets in configurational file `./src/configs/ga-config.json`. When setting datasets check its order in `./src/configs/data-config.json`.

4. Set current directory to `./src`:

```bash
  cd src
```

5. To run the programm use:

```bash
  python main.py
```

6. When execution is done you'll receive graphs with confusion matrix, loss function and classification accuracy calculated for the NN model with optimized hyperparameters.

## Specific runnung modes

If you want to do some specific test, you'll need to comment the whole `---Automated optimization for all datasets` block and you can chose some test from those block by uncommenting them:

- `---Generating baseline CNN and RNN models`. Train and evaluate RNN and CNN models with baseline hyperparameters.
- `---Specific dataset optimization`. Allows you to set manually type of testing NN and dataset.
- `---EDA for timeseries datasets`. Will make a EDA(Exploratory Data Analysis) and print a diagramms for every dataset.
- `---Time series analysis`. Allows you to plot plot specific batch of data from time series dataset.

## NN module description

The NN module located in `./src/models/` directory and has 3 files:

1. File `base_model.py` has `BaseModel` class which allows to create NN model no matter its type.
2. File `cnn_model.py` has `cnnModel` class which is fully inherits the whole functionality from `BaseModel` class and add configuration for hidden convolution laeyers. In addiction this class has methods that allow to convert _genetic chromosome_ representation into the list of the hyperparameters of CNN and also methods to build, test and evaluate CNN model.
3. File `rnn_model.py` has `rnnModel` class which is fully inherits the whole functionality from `BaseModel` class and add configuration for hidden recurrent laeyers. In addiction this class has methods that allow to convert _genetic chromosome_ representation into the list of the hyperparameters of RNN and also methods to build, test and evaluate CNN model.

## Genetic algorithm module

The whole GA module consists of one file `./src/evolutionary_search/genetic_optimization.py` and one class `GeneticSearch` which allows to finetune the hyperparameters of testing NN model for specific dataset.

## Visualizing experiments with TensorBoard

All experiments are logging to the specific directory `./src/experiments/<NN_TYPE>_<DATASET_NAME>`.
In order to use tensorboard inside root project directory, you'll need to run following script:

```bash
  tensorboard --logdir src/experiment
```

Once it's running you'll receive message in your terminal with a localhost URL that allows to open TensorBoard tool.
