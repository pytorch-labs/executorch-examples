/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const DIMS = 1024;


let modelButton = null;
let imageButton = null;
let inference_button = null
let maskButton = null;
let canvasCtx = null;
let maskCanvas = null;
let maskCanvasCtx = null;
let pointerCanvas = null;
let pointerCanvasCtx = null;
let modelText = null;

var Module = {
  onRuntimeInitialized: function() {
    modelButton = document.getElementById("upload_model_button");
    modelButton.addEventListener("click", openFilePickerModel);

    imageButton = document.getElementById("upload_image_button");
    imageButton.addEventListener("click", openFilePickerImage);

    inferenceButton = document.getElementById("inference_button");
    inferenceButton.addEventListener("click", runModel);

    maskButton = document.getElementById("mask_button");
    maskButton.addEventListener("click", toggleMask);

    const canvas = document.getElementById("canvas");
    canvasCtx = canvas.getContext("2d", { willReadFrequently: true });

    maskCanvas = document.getElementById("mask_canvas");
    maskCanvasCtx = maskCanvas.getContext("2d");

    pointerCanvas = document.getElementById("pointer_canvas");
    pointerCanvas.addEventListener("click", canvasClick);
    pointerCanvasCtx = pointerCanvas.getContext("2d");

    modelText = document.getElementById("model_text");
  }
}
const et = Module;

let module = null;
let imageTensor = null;
let point = null;

function toggleMask(event) {
  if (maskCanvas.style.display === "none") {
    maskCanvas.style.display = "block";
    maskButton.textContent = "Hide Mask";
  } else {
    maskCanvas.style.display = "none";
    maskButton.textContent = "Show Mask";
  }
}

function runModel(event) {
  const pointTensor = et.Tensor.fromArray([1, 1, 1, 2], point);
  const labelTensor = et.Tensor.fromArray([1, 1, 1], [1]);

  const startTime = performance.now();
  console.log("Running model...");
  const output = module.forward([imageTensor, pointTensor, labelTensor]);
  const endTime = performance.now();
  console.log(((endTime - startTime)/1000).toFixed(2) + "s");

  const argmax = output[1].data.reduce((iMax, elem, i, arr) => elem > arr[iMax] ? i : iMax, 0);

  const imageData = maskCanvasCtx.createImageData(DIMS, DIMS);
  for (let i = 0; i < DIMS; i++) {
    for (let j = 0; j < DIMS; j++) {
      const idx = ((i * DIMS + j) * 4);
      const idx3 = (argmax * DIMS + i) * DIMS + j;
      imageData.data[idx + 2] = 255;
      imageData.data[idx + 3] = Math.min(1, output[0].data[idx3]) * 100;
    }
  }
  maskCanvasCtx.putImageData(imageData, 0, 0);

  maskCanvas.style.display = "block";
  maskButton.textContent = "Hide Mask";
  maskButton.disabled = false;
  inferenceButton.disabled = true;

  pointTensor.delete();
  labelTensor.delete();
  output[0].delete();
  output[1].delete();
}

function canvasClick(event) {
  if (module == null) {
    return;
  }

  const rect = pointerCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  pointerCanvasCtx.beginPath();
  pointerCanvasCtx.clearRect(0, 0, DIMS, DIMS);
  pointerCanvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
  pointerCanvasCtx.stroke();

  console.log("Clicked at: " + x + ", " + y);
  point = [x, y];
  inferenceButton.disabled = false;
}

function verifyModel(mod) {
  try {
    mod.loadMethod("forward");
  } catch (e) {
    modelText.textContent = "Failed to load forward method: " + e;
    modelText.style.color = "red";
    return false;
  }

  const methodMeta = mod.getMethodMeta("forward");
  if (methodMeta.inputTags.length != 3) {
    modelText.textContent = "Error: Expected input size of 3, got " + methodMeta.inputTags.length;
    modelText.style.color = "red";
    return false;
  }

  for (let i = 0; i < 3; i++) {
    if (methodMeta.inputTags[i] != et.Tag.Tensor) {
      modelText.textContent = "Error: Expected input " + i + " to be Tensor, got " + methodMeta.inputTags[i].name;
      modelText.style.color = "red";
      return false;
    }
  }

  const expectedInputSizes = [[1, 3, DIMS, DIMS], [1, 1, 1, 2], [1, 1, 1]];
  for (let i = 0; i < 3; i++) {
    const inputMeta = methodMeta.inputTensorMeta[i];
    if (inputMeta.sizes.length != expectedInputSizes[i].length) {
      modelText.textContent = "Error: Expected input " + i + " shape to be " + expectedInputSizes[i] + ", got " + inputMeta.sizes;
      modelText.style.color = "red";
      return false;
    }

    for (let j = 0; j < expectedInputSizes[i].length; j++) {
      if (inputMeta.sizes[j] != expectedInputSizes[i][j]) {
        modelText.textContent = "Error: Expected input " + i + " shape to be " + expectedInputSizes[i] + ", got " + inputMeta.sizes;
        modelText.style.color = "red";
        return false;
      }
    }

    if (inputMeta.scalarType != et.ScalarType.Float) {
      modelText.textContent = "Error: Expected input " + i + " type to be Float, got " + inputMeta.scalarType.name;
      modelText.style.color = "red";
      return false;
    }
  }

  if (methodMeta.outputTags.length != 2) {
    modelText.textContent = "Error: Expected output size of 2, got " + methodMeta.outputTags.length;
    modelText.style.color = "red";
    return false;
  }

  for (let i = 0; i < 2; i++) {
    if (methodMeta.outputTags[i] != et.Tag.Tensor) {
      modelText.textContent = "Error: Expected output " + i + " to be Tensor, got " + methodMeta.outputTags[i].name;
      modelText.style.color = "red";
      return false;
    }
  }

  const expectedOutputSizes = [[1, 1, 3, DIMS, DIMS], [1, 1, 3]];
  for (let i = 0; i < 2; i++) {
    const outputMeta = methodMeta.outputTensorMeta[i];
    if (outputMeta.sizes.length != expectedOutputSizes[i].length) {
      modelText.textContent = "Error: Expected output " + i + " shape to be " + expectedOutputSizes[i] + ", got " + outputMeta.sizes;
      modelText.style.color = "red";
      return false;
    }

    for (let j = 0; j < expectedOutputSizes[i].length; j++) {
      if (outputMeta.sizes[j] != expectedOutputSizes[i][j]) {
        modelText.textContent = "Error: Expected output " + i + " shape to be " + expectedOutputSizes[i] + ", got " + outputMeta.sizes;
        modelText.style.color = "red";
        return false;
      }
    }

    if (outputMeta.scalarType != et.ScalarType.Float) {
      modelText.textContent = "Error: Expected output " + i + " type to be Float, got " + outputMeta.scalarType.name;
      modelText.style.color = "red";
      return false;
    }
  }

  return true;
}

function loadModelFile(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
    const buffer = event.target.result;

    const mod = et.Module.load(buffer);

    if (verifyModel(mod)) {
      if (module != null) {
        module.delete();
      }
      module = mod;
      modelText.textContent = 'Uploaded model: ' + file.name;
      modelText.style.color = null;
      canvasCtx.clearRect(0, 0, DIMS, DIMS);
      upload_image_button.disabled = false;
    }
  };
  reader.readAsArrayBuffer(file);
}

function* generateTensorData(data) {
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < data.length; i += 4) {
      yield data[i + j] / 255.0;
    }
  }
}

function loadImageFile(file) {
    const img = new Image();
    img.onload = function() {
      canvasCtx.drawImage(img, 0, 0, DIMS, DIMS);
      const imageData = canvasCtx.getImageData(0, 0, DIMS, DIMS);
      if (imageTensor != null) {
        imageTensor.delete();
      }
      imageTensor = et.Tensor.fromIter([1, 3, DIMS, DIMS], generateTensorData(imageData.data));

      maskCanvas.style.display = "none";
      maskButton.textContent = "Hide Mask";
      maskButton.disabled = true;
    }
    img.src = URL.createObjectURL(file);
}

async function openFilePickerModel() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: 'Model Files',
        accept: {
          'application/octet-stream': ['.pte'],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    loadModelFile(file);
  } catch (err) {
    if (err.name === 'AbortError') {
      // Handle user abort silently
    } else {
      console.error('File picker error:', err);
    }
  }
}

async function openFilePickerImage() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: "Images",
        accept: {
          "image/*": [".png", ".gif", ".jpeg", ".jpg"],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    loadImageFile(file);
  } catch (err) {
    if (err.name === 'AbortError') {
      // Handle user abort silently
    } else {
      console.error('File picker error:', err);
    }
  }
}
