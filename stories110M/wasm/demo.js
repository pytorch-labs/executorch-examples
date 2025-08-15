/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const DIMS = 1024;


let modelButton = null;
let tokenizerButton = null;
let inferenceButton = null
let etdumpButton = null;
let modelText = null;
let tokenizerText = null;
let promptBox = null;
let temperatureSlider = null;

let tokenizer = null;
var Module = {
  onRuntimeInitialized: function() {
    loadTokenizer().then(tk => {
      tokenizer = new tk.SPTokenizer();

      modelButton = document.getElementById("upload_model_button");
      modelButton.addEventListener("click", openFilePickerModel);

      tokenizerButton = document.getElementById("upload_tokenizer_button");
      tokenizerButton.addEventListener("click", openFilePickerTokenizer);

      inferenceButton = document.getElementById("inference_button");
      inferenceButton.addEventListener("click", runModel);

      modelText = document.getElementById("model_text");
      tokenizerText = document.getElementById("tokenizer_text");
      promptBox = document.getElementById("prompt");

      etdumpButton = document.getElementById("etdump_button");
      etdumpButton.addEventListener("click", etdump);

      temperatureSlider = document.getElementById("temperature_slider");
      const temperatureText = document.getElementById("temperature_text");
      temperatureSlider.oninput = function() {
        temperatureText.textContent = "Temperature: " + (temperatureSlider.value / 10).toFixed(1);
      }
    });
  }
}
const et = Module;

let module = null;
let imageTensor = null;
let point = null;

function etdump() {
  if (module == null) {
    return;
  }

  const etdump = module.etdump();
  const blob = new Blob([etdump.buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "result.etdump";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  etdump.delete();
}

function sampleNextToken(logits, temperature) {
    if (temperature <= 0) {
        throw new Error("Temperature must be > 0");
    }
    // Convert logits to a regular array for easier manipulation
    const arr = Array.from(logits);
    // For numerical stability, subtract the max logit before exponentiating
    const maxLogit = Math.max(...arr);
    const scaled = arr.map(x => (x - maxLogit) / temperature);
    // Compute exponentials
    const exp = scaled.map(Math.exp);
    // Compute softmax probabilities
    const sumExp = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map(x => x / sumExp);
    // Sample from the probability distribution
    const r = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probs.length; i++) {
        cumSum += probs[i];
        if (r < cumSum) {
            return i;
        }
    }
    // Fallback: return last index (shouldn't happen unless numerical issues)
    return probs.length - 1;
}

async function* generateTokens(tokens, bos, temperature) {
  for (let kvInd = 0n; kvInd < 10n; kvInd++) {
    const input1 = et.Tensor.fromArray([1, tokens.length], tokens);
    const input2 = et.Tensor.fromArray([1], [0n]);

    const startTime = performance.now();
    const output = module.forward([input1, input2]);
    const endTime = performance.now();
    console.log(((endTime - startTime)/1000).toFixed(2) + "s");

    let token = 0;
    if (temperature == 0) {
      token = output[0].data.reduce((iMax, elem, i, arr) => elem > arr[iMax] ? i : iMax, 0);
    } else {
      token = sampleNextToken(output[0].data, temperature);
    }

    if (token == bos) {
      break;
    }

    const str = tokenizer.decode(tokens[tokens.length - 1], token);
    tokens.push(BigInt(token));

    input1.delete();
    input2.delete();
    output[0].delete();

    yield str;
  }
}

async function runModel(event) {
  let text = promptBox.value;
  const tokens = tokenizer.encode(text, 0, 0);
  const bos = tokenizer.bosTok;
  const temperature = temperatureSlider.value / 10;

  if (tokens.length == 0) {
    return;
  }

  promptBox.disabled = true;
  inferenceButton.disabled = true;
  temperatureSlider.disabled = true;

  try {
    for await (const str of generateTokens(tokens, bos, temperature)) {
      text += str;
      promptBox.value = text;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  } finally {
    promptBox.disabled = false;
    inferenceButton.disabled = false;
    temperatureSlider.disabled = false;
  }
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
  console.log(mod.getMethods());
  console.log(methodMeta);
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

      if (tokenizer.isLoaded) {
        inferenceButton.disabled = false;
      }
      etdumpButton.disabled = false;
    }
  };
  reader.readAsArrayBuffer(file);
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

function loadTokenizerFile(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
    const buffer = event.target.result;
    tokenizer.load(buffer);

    console.log("BOS: " + tokenizer.bosTok);
    console.log("EOS: " + tokenizer.eosTok);
    console.log("Vocab size: " + tokenizer.vocabSize);

    tokenizerText.textContent = 'Uploaded tokenizer: ' + file.name;
    tokenizerText.style.color = null;

    if (module != null) {
      inferenceButton.disabled = false;
    }
  };
  reader.readAsArrayBuffer(file);
}

async function openFilePickerTokenizer() {
  try {
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{
        description: 'Tokenizer Files',
        accept: {
          'application/octet-stream': ['.model'],
        },
      }],
      multiple: false, // Set to true for multiple file selection
    });
    const file = await fileHandle.getFile();
    loadTokenizerFile(file);
  } catch (err) {
    if (err.name === 'AbortError') {
      // Handle user abort silently
    } else {
      console.error('File picker error:', err);
    }
  }
}
