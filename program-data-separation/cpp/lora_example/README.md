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
LoRA isn't available in the 0.7.0 release of ExecuTorch. Instead, please install from source until ExecuTorch 1.0 is released.

[Install ExecuTorch pip package from source](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html#install-executorch-pip-package-from-source).

Currently, the LoRA changes aren't in nightlies. Once they are in, you can also install from the nightly build.
```
pip install executorch==0.8.0.devYYYYMMDD --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Export the model/s.
Change into the program-data-separation directory and create a directory to hold exported artifacts.
```bash
cd ~/executorch-examples/program-data-separation
mkdir models
```

Export models into the `models` directory. The first command will generated undelegated model/data files, and the second will generate XNNPACK-delegated model/data files.
```bash
sh export_lora.sh
```
Expect the files:
- llama_3_2_1B.pte
- llama_3_2_1B.ptd
- llama_3_2_1B_lora.pte
- foundation_weights.ptd
- tokenizer.model

llama_3_2_1B.ptd and foundation_weights.ptd contain the same contents, and you can remove llama_3_2_1B.ptd.
tokenizer.model is copied from the temp directory where we downloaded the HF artifacts. It will be used at runtime.

Note:
- PTE: contains the program execution logic.
- PTD: contains the constant tensors used by the PTE.

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
Install some dependencies:
```bash
cd ~/executorch-examples/program-data-separation/cpp/executorch
sh examples/models/llama/install_requirements.sh
```

Build the executable:
```bash
cd ~/executorch-examples/program-data-separation/cpp/lora_example
sh build_example.sh
```

## Run the executable.
```
./build/bin/executorch_program_data_separation --lora_model_path=../../llama_3_2_1B_lora.pte --llama_model_path=../../llama_3_2_1B.pte --tokenizer_path=../../tokenizer.model --data_path=../../foundation.ptd
```

## Clean up.
rm -rf build
cd ~/executorch-examples/program-data-separation
rm -rf *.pte *.ptd tokenizer.model
