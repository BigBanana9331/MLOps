# MLOps project

Project deloys image classifier model on tflite with MLOps pipeline

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install folder.

```bash
pip install pipenv
```

Create a python 3.9 virtual environment using pipenv

```bash
pipenv --python 3.9
```

Install dependencies

```bash
pipenv install
```

Active shell for virtual environment

```bash
pipenv shell
```

## Clone dataset and models (Optional)

Get dataset

```bash
dvc get https://github.com/BigBanana9331/MLOps.git dataset
```

Get models

```bash
dvc get https://github.com/BigBanana9331/MLOps.git models
```

## Usage

Navigate to FMClassifier folder and run command

```bash
python train.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)