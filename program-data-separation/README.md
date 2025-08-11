# Program Data Separation Examples

This directory provides an example of the Program Data Separation APIs in ExecuTorch. Specifically, it showcases:
1. Simple program data separation examples using the portable operators and XNNPACK.
2. LoRA inference example with a LoRA and non-LoRA model sharing foundation weights.

## Program Data Separation

The program-data separation APIs allow users to generate a separate data file when exporting and lowering a model. i.e., generate a PTE file containing the model execution program, and one (or more) [PTD](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/README.md) file/s containing only weights.

PTD files are used to store data outside of the PTE file. Some use-cases:
- On-device training: checkpointing for model weights.
- Deduplication: sharing model weights between multiple executable PTE files. This can significantly reduce binary file size and runtime memory usage.
- Flexible deployment: allow async updates between program and data, especially if they are updated with different cadences.

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
```
pip install executorch==0.7.0
```

## Export a model with program-data separation
To export a non-delegated linear model, into the current directory:
```python
python export_linear.py --outdir .
```
Expect the files 'linear.pte' and 'linear.ptd'.

To export a linear model delegated to XNNPACK, into the current directory:
```python
python export_linear.py --outdir . --xnnpack
```
Expect the files 'linear_xnnpack.pte' and 'linear_xnnpack.ptd'.

Note:
- PTE: contains the program execution logic.
- PTD: contains the constant tensors used by the PTE.

For more information on the PTD data format, please see the [flat_tensor](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/README.md) directory.

Please see [program-data-separation/cpp](cpp/) for instructions on running the exported models.

## Export a model with LoRA
A major use-case that program-data separation enables is inference with multiple LoRA adapters. LoRA is a fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). LoRA fine-tuning produces lightweight 'adapter' weights that can be applied to an existing model to adapt it to a new task. LoRA adapters are typically small in comparison to LLM foundation weights, on the order of KB-MB depending on the finetuning setup and model size.

To enable LoRA, we generate:
- PTE file/s: containing program and LoRA adapter weights.
- PTD file: containing foundation weights.

Multiple LoRA-adapted PTE files can share the same foundation weights and adding a model adapted to a new task incurs minimal binary size and runtime memory overhead.

### Requirements
LoRA is currently supported on executorch main. [Please install ExecuTorch pip package from source](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html#install-executorch-pip-package-from-source), until executorch==1.0 is released.
