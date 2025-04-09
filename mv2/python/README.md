### Virtual environment setup
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```
Or alternatively, [install conda on your machine](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
```bash
conda create -yn executorch-examples-mv2 python=3.10.0 && conda activate executorch-examples-mv2
```

### Install dependencies
```
pip install -r requirements.txt
```

### Export a model
```
python export.py
```

### Run model via pybind
```
python run.py
```