# ExecuTorch Program Data Separation Demo C++.

This directory contains the C++ code to run the examples generated in [program-data-separation](../program-data-separation/README.md).


## Virtual environment setup.
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```
Or alternatively, [install conda on your machine](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
```bash
conda create -yn executorch-ptd python=3.10.0 && conda activate executorch-ptd
```

Install dependencies:
```bash
pip install executorch==0.7.0
```

## Export the model/s.

Change into the program-data-separation directory and create a directory to hold exported artifacts.
```bash
cd ~/executorch-examples/program-data-separation
mkdir models
```

Export models into the `models` directory. The first command will generated undelegated model/data files, and the second will generate XNNPACK-delegated model/data files.
```bash
./export_lora.sh
```
Expect the files `lora.pte` and `lora.ptd`.

Note:
- PTE: contains the program execution logic.
- PTD: contains the constant tensors used by the PTE.

See [program-data-separation](../../program-data-separation/README.md) for instructions.

## Install runtime dependencies.
The ExecuTorch repository is configured as a git submodule at `~/executorch-examples/program-data-separation/cpp/executorch`.  To initialize it:
```bash
cd ~/executorch-examples/
git submodule sync
git submodule update --init --recursive
```
Install dev requirements for ExecuTorch

```bash
cd ~/executorch-examples/program-data-separation/cpp/executorch
pip install -r requirements-dev.txt
```

## Build the runtime.
Build the executable:
```bash
cd ~/executorch-examples/program-data-separation/cpp/lora_example
chmod +x build_example.sh
./build_example.sh
```

## Run the executable.
```
./build/bin/executorch_program_data_separation --model-path ../../models/linear.pte --data-path ../../models/linear.ptd

./build/bin/executorch_program_data_separation --model-path ../../models/linear_xnnpack.pte --data-path ../../models/linear_xnnpack.ptd
```

## Clean up.
rm -rf build
cd ~/executorch-examples/program-data-separation
rm -rf models
