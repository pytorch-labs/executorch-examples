# Program Data Separation Examples

This directory provides an example of the Program Data Separation APIs in ExecuTorch.

The program-data separation APIs allow users to generate a separate data file when exporting and lowering a model. i.e., generate a PTE file containing the model execution program, and one (or more) [PTD](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/README.md) file/s containing only weights.

PTD files are used to store data outside of the PTE file. Some use-cases:
- On-device training: checkpointing for model weights.
- Deduplication: sharing model weights between multiple executable PTE files. This can significantly reduce binary file size and runtime memory usage.
- Flexible deployment: allow async updates between program and data, especially if they are updated with different cadences.

## LoRA
A major use-case that program-data separation enables is inference with multiple LoRA adapters. LoRA is a fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). LoRA fine-tuning produces lightweight 'adapter' weights that can be applied to an existing model to adapt it to a new task. LoRA adapters are typically small in comparison to LLM foundation weights. They are generally on the order of KB,MB, depending on the finetuning setup and model size.

With program-data separation, users can generate a PTE file containing the program and LoRA weights, and save the original foundation weights to a separate PTD file. Provided they are based on the same underlying model, multiple LoRA-adapted PTE files can share the same foundation weights. This means adding a model adapted to a new task incurs minimal binary size and runtime memory overhead; the cost of the lora adapter weights.

An example of this usage is coming soon.

## Virtual environment setup
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```
Or alternatively, [install conda on your machine](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
```bash
conda create -yn executorch-ptd python=3.10.0 && conda activate executorch-ptd
```

Install dependencies:

[Please install ExecuTorch pip package from source](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html#install-executorch-pip-package-from-source), until executorch==0.7.0 is released.

```
pip install executorch==0.7.0
```

## Export a model with program-data separation
To export a non-delegated linear model, into the current directory:
```python
python export.py --outdir .
```
Expect the files 'linear.pte' and 'linear.ptd'.

To export a linear model delegated to XNNPACK, into the current directory:
```python
python export.py --outdir . --xnnpack
```
Expect the files 'linear_xnnpack.pte' and 'linear_xnnpack.ptd'.

Note:
- PTE: contains the program execution logic.
- PTD: contains the constant tensors used by the PTE.

For more information on the PTD data format, please see the [flat_tensor](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/README.md) directory.

## Runtime (cpp)
The cpp/ directory contains the executorch submodule along with a main.cpp file that demonstrates how to load the PTE and PTD files and execute the program.

First, export your PTE and PTD files using the instructions above.

**Build instructions**

Change to the cpp directory.
```
cd cpp
```

Create build directory if it doesn't exist.
```
mkdir -p build
cd build
```

Configure CMake.
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Build the project.
```
cmake --build . -j$(nproc)
echo "Build complete! Executable located at: ./bin/executorch_program_data_separation"
```

Run the executable.
```
./bin/executorch_program_data_separation --model-path ../../linear.pte --data-path ../../linear.ptd

./bin/executorch_program_data_separation --model-path ../../linear_xnnpack.pte --data-path ../../linear_xnnpack.ptd
```
