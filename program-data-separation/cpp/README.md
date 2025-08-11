# ExecuTorch Program Data Separation Demo C++.

This directory contains the C++ code to run the examples generated in [program-data-separation](../program-data-separation/README.md).

## Build instructions
0. Export the model/s. See [program-data-separation](../program-data-separation/README.md) for instructions.
1. The ExecuTorch repository is configured as a git submodule at `~/executorch-examples/program-data-separation/cpp/executorch`.  To initialize it:
   ```bash
    cd ~/executorch-examples/
    git submodule sync
    git submodule update --init --recursive
   ```
2. Install dev requirements for ExecuTorch

    ```bash
    cd ~/executorch-examples/mv2/cpp/executorch
    pip install -r requirements-dev.txt
    ```

## Program-data separation demo
**Build instructions**

Build the executable:
```bash
cd ~/executorch-examples/program-data-separation/cpp
chmod +x build_example.sh
./build_example.sh
```

Run the executable.
```
./bin/executorch_program_data_separation --model-path ../../linear.pte --data-path ../../linear.ptd

./bin/executorch_program_data_separation --model-path ../../linear_xnnpack.pte --data-path ../../linear_xnnpack.ptd
```

## LoRA demo
