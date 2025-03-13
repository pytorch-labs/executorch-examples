#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.models as models
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.mps.partition import MPSPartitioner
from executorch.exir.backend.backend_details import CompileSpec 
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import ssl
import certifi


def main() -> None:
    model = models.mobilenet_v3_small(weights="DEFAULT").eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    et_program_coreml = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[CoreMLPartitioner()],
    ).to_executorch()

    et_program_mps = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[MPSPartitioner([CompileSpec("use_fp16", bytes([True]))])], 
    ).to_executorch()

    et_program_xnnpack = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    save_path = "executorch-examples/mv3/apple/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/"
    with open(save_path+"mv3_coreml_all.pte", "wb") as file:
        et_program_coreml.write_to_file(file)
    with open(save_path+"mv3_mps_float16.pte", "wb") as file:
        et_program_mps.write_to_file(file)
    with open(save_path+"mv3_xnnpack_fp32.pte", "wb") as file:
        et_program_xnnpack.write_to_file(file)


if __name__ == "__main__":
    main()
