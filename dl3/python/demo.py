#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.runtime import Runtime


def main() -> None:
    runtime = Runtime.get()
    program = runtime.load_program("dl3_xnnpack_fp32.pte")
    method = program.load_method("forward")
    result = method.execute((torch.randn(1, 3, 224, 224), ))
    # TODO: Load an image and show the output
    print(result)


if __name__ == "__main__":
    main()
