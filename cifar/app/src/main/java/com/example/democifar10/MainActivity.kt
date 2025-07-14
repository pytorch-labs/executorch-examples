package com.example.democifar10

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.facebook.soloader.SoLoader
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.TrainingModule
import org.pytorch.executorch.SGD
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import kotlin.jvm.Throws

class MainActivity : ComponentActivity() {
    private var tModule: TrainingModule? = null

    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)

        // Create parent directories if they don't exist
        if (!file.parentFile?.exists()!!) {
            file.parentFile?.mkdirs()
        }

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        try {
            resources.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            Log.e("AssetCopy", "Error copying asset $assetName: ${e.message}")
            throw e
        }
    }

    // Function to convert a tensor to a bitmap
    private fun tensorToBitmap(tensor: Tensor): Bitmap {
        val channels = 3
        val height = 32
        val width = 32

        // Get the tensor data as a float array
        val data = tensor.getDataAsFloatArray()

        // Create a bitmap with the appropriate dimensions
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Fill the bitmap with the tensor data
        for (y in 0 until height) {
            for (x in 0 until width) {
                // In PyTorch, the tensor is in CHW format (channels, height, width)
                // We need to convert it to RGB format for the bitmap
                val r = (data[0 * height * width + y * width + x] * 255).toInt().coerceIn(0, 255)
                val g = (data[1 * height * width + y * width + x] * 255).toInt().coerceIn(0, 255)
                val b = (data[2 * height * width + y * width + x] * 255).toInt().coerceIn(0, 255)

                // Set the pixel color in the bitmap
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }

        return bitmap
    }

    // Function to save a bitmap to a PNG file
    private fun saveBitmapToPng(bitmap: Bitmap, file: File) {
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.d("ExecuTorchApp", "Saved image to ${file.absolutePath}")
        } catch (e: IOException) {
            Log.e("ExecuTorchApp", "Error saving image", e)
        }
    }

    /**
     * Load and process CIFAR-10 data from multiple batch files
     *
     * @param trainBatchFiles List of training batch files (e.g., ["data_batch_1.bin", ..., "data_batch_5.bin"])
     * @param testBatchFiles List of testing batch files (e.g., ["test_batch.bin"])
     * @param trainSize Number of images to use for training
     * @param testSize Number of images to use for testing
     * @return Pair of image and label Pairs: ((trainImgData, trainLblData), (testImgData, testLblData))
     */
    private fun loadCifar10Data(
        trainBatchFiles: List<String>,
        testBatchFiles: List<String>,
        trainSize: Int = 1000,
        testSize: Int = 1000
    ): Pair<Pair<ByteArray, ByteArray>, Pair<ByteArray, ByteArray>>? {
        try {
            val width = 32
            val height = 32
            val channels = 3
            val labelSize = 1
            val imageSize = width * height * channels
            val recordSize = labelSize + imageSize
            val imagesPerBatch = 10000 // Each CIFAR-10 batch file contains 10,000 images

            // Create arrays to hold the data
            val trainImgData = ByteArray(trainSize * imageSize)
            val trainLblData = ByteArray(trainSize)
            val testImgData = ByteArray(testSize * imageSize)
            val testLblData = ByteArray(testSize)

            val cifar10Dir = "cifar-10-batches-bin"

            // Load training data from multiple batch files
            var trainImagesLoaded = 0
            for (batchFile in trainBatchFiles) {
                if (trainImagesLoaded >= trainSize) break

                // Get the path to the batch file in the assets directory
                val batchFilePath = "$cifar10Dir/$batchFile"

                // Copy the file from assets to internal storage
                val internalPath = assetFilePath(batchFilePath)

                // Open the batch file
                val file = File(internalPath)
                val totalSize = file.length()
                val totalRecords = (totalSize / recordSize).toInt()

                // Calculate how many images to load from this batch
                val imagesToLoad = minOf(imagesPerBatch, trainSize - trainImagesLoaded)

                Log.d(
                    "Cifar10Loader", "Loading $imagesToLoad training images from $batchFile"
                )

                // Read the batch file
                DataInputStream(FileInputStream(file)).use { input ->
                    for (i in 0 until imagesToLoad) {
                        // Read label (1 byte) - use readUnsignedByte to get 0-255 range
                        trainLblData[trainImagesLoaded + i] = input.readUnsignedByte().toByte()

                        // Read image data (3072 bytes)
                        val tempImgData = ByteArray(imageSize)
                        input.readFully(tempImgData)

                        // Copy image data to trainImgData
                        System.arraycopy(
                            tempImgData,
                            0,
                            trainImgData,
                            (trainImagesLoaded + i) * imageSize,
                            imageSize
                        )
                    }
                }

                trainImagesLoaded += imagesToLoad
                Log.d("Cifar10Loader", "Total training images loaded so far: $trainImagesLoaded")
            }

            // Load testing data from multiple batch files
            var testImagesLoaded = 0
            for (batchFile in testBatchFiles) {
                if (testImagesLoaded >= testSize) break

                // Get the path to the batch file in the assets directory
                val batchFilePath = "$cifar10Dir/$batchFile"

                // Copy the file from assets to internal storage
                val internalPath = assetFilePath(batchFilePath)

                // Open the batch file
                val file = File(internalPath)
                val totalSize = file.length()
                val totalRecords = (totalSize / recordSize).toInt()

                // Calculate how many images to load from this batch
                val imagesToLoad = minOf(imagesPerBatch, testSize - testImagesLoaded)

                Log.d(
                    "Cifar10Loader", "Loading $imagesToLoad testing images from $batchFile"
                )

                // Read the batch file
                DataInputStream(FileInputStream(file)).use { input ->
                    for (i in 0 until imagesToLoad) {
                        // Read label (1 byte) - use readUnsignedByte to get 0-255 range
                        testLblData[testImagesLoaded + i] = input.readUnsignedByte().toByte()

                        // Read image data (3072 bytes)
                        val tempImgData = ByteArray(imageSize)
                        input.readFully(tempImgData)

                        // Copy image data to testImgData
                        System.arraycopy(
                            tempImgData,
                            0,
                            testImgData,
                            (testImagesLoaded + i) * imageSize,
                            imageSize
                        )
                    }
                }

                testImagesLoaded += imagesToLoad
                Log.d("Cifar10Loader", "Total testing images loaded so far: $testImagesLoaded")
            }

            Log.d(
                "Cifar10Loader",
                "Successfully loaded $trainImagesLoaded training and $testImagesLoaded testing images"
            )
            Log.d(
                "Cifar10Loader",
                "Training data: ${trainImgData.size} bytes, ${trainLblData.size} labels"
            )
            Log.d(
                "Cifar10Loader",
                "Testing data: ${testImgData.size} bytes, ${testLblData.size} labels"
            )

            return Pair(Pair(trainImgData, trainLblData), Pair(testImgData, testLblData))

        } catch (e: IOException) {
            Log.e("Cifar10Loader", "Error loading CIFAR-10 data: ${e.message}")
            e.printStackTrace()
            return null
        }
    }

    private val classes =
        arrayOf("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    override fun onCreate(savedInstanceState: Bundle?) {
        val batchSize = 1
        val width = 32
        val height = 32
        val channels = 3
        val pixelsPerImage = width * height * channels
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize SoLoader for loading native libraries
        SoLoader.init(this, false)

        // Load CIFAR-10 data from multiple batch files
        // Use data_batch_1.bin through data_batch_4.bin for training (32,000 images)
        // Use test_batch.bin for testing (8,000 images)
        val trainBatchFiles = listOf("train_data.bin")
//            listOf("data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin")
//        val testBatchFiles = listOf("test_batch.bin")
        val testBatchFiles = listOf("test_data.bin")
//        val cifar10Data = loadCifar10Data(trainBatchFiles, testBatchFiles, 32000, 8000)
        val cifar10Data = loadCifar10Data(trainBatchFiles, testBatchFiles, 1000, 1000)

        // Extract training and testing data
        val trainData = cifar10Data?.first
        val testData = cifar10Data?.second

        val trnImgData = trainData?.first
        val trnLblData = trainData?.second
        val tstImgData = testData?.first
        val tstLblData = testData?.second

        Log.d(
            "ExecuTorchApp", "Loaded ${trnImgData?.size?.div(pixelsPerImage) ?: 0} training images"
        )
        Log.d(
            "ExecuTorchApp", "Loaded ${tstImgData?.size?.div(pixelsPerImage) ?: 0} testing images"
        )

        // Get a batch of 4 images from the test image and labels data for testing
        val imgData = tstImgData?.take(batchSize * height * width * channels)
        val lblData = tstLblData?.take(batchSize)

        // Create a direct ByteBuffer as required by Tensor.fromBlob
        val buffer = Tensor.allocateFloatBuffer(imgData?.size ?: 0)
        // Convert ByteArray to FloatArray before putting into FloatBuffer
        val floatArray = FloatArray(imgData?.size ?: 0)
        imgData?.forEachIndexed { index, byte -> floatArray[index] = byte.toFloat() }
        buffer.put(floatArray)
        buffer.rewind() // Reset position to the beginning of the buffer

        val batchTestImageTensor = Tensor.fromBlob(
            buffer,
            longArrayOf(batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong())
        )
        Log.d(
            "ExecuTorchApp",
            "Image tensor shape from test bin: ${batchTestImageTensor.shape().joinToString(", ")}"
        )

        // Save the images in the batchTestImageTensor to png or jpg files
        val imageDir = File(applicationContext.filesDir, "images")
        if (!imageDir.exists()) {
            imageDir.mkdir()
        }

        val batchTestLabelBuffer = LongArray(batchSize) { lblData?.get(it.toInt())?.toLong() ?: 0 }

        // Debug: Check label values to ensure they're in valid range [0, 9] for CIFAR-10
        Log.d("ExecuTorchApp", "Label values: ${batchTestLabelBuffer.contentToString()}")
        for (i in batchTestLabelBuffer.indices) {
            if (batchTestLabelBuffer[i] < 0 || batchTestLabelBuffer[i] >= 10) {
                Log.e(
                    "ExecuTorchApp",
                    "Invalid label value at index $i: ${batchTestLabelBuffer[i]} (should be 0-9)"
                )
            }
        }

        val testLabelBuffer = Tensor.fromBlob(
            batchTestLabelBuffer, longArrayOf(batchSize.toLong())
        )

        try {
//            val modelPath = assetFilePath("et_cifar10_model_without_ptd.pte")
//            val dataPath = assetFilePath("generic_cifar_ptd_old")
            val modelPath = assetFilePath("cifar10_model.pte")
            val dataPath = assetFilePath("generic_cifar.ptd")

            // Verify files exist and are readable
            val modelFile = File(modelPath)
            val dataFile = File(dataPath)

            Log.d(
                "ExecuTorchApp",
                "Model file exists: ${modelFile.exists()}, readable: ${modelFile.canRead()}, size: ${modelFile.length()}"
            )
            Log.d(
                "ExecuTorchApp",
                "Data file exists: ${dataFile.exists()}, readable: ${dataFile.canRead()}, size: ${dataFile.length()}"
            )

            // Load the training module
            tModule = TrainingModule.load(modelPath, dataPath)
            Log.d("ExecuTorchApp", "TrainingModule loaded successfully")
        } catch (e: Exception) {
            Log.e("ExecuTorchApp", "Error loading the pte or the ptd model: ${e.message}", e)
            e.printStackTrace()
            finish()
        }

        // Let's run the inference using the batchTestImageTensor and testLabelBuffer and log the results
        val inputEValues = arrayOf(EValue.from(batchTestImageTensor), EValue.from(testLabelBuffer))
        val outputEValues = tModule?.executeForwardBackward("forward", *inputEValues)
            ?: throw IllegalStateException("Execution module is not loaded.")
        for (i in outputEValues.indices) {
            try {
                val tensor = outputEValues[i].toTensor()
                val data = tensor.getDataAsFloatArray()
                val shape = tensor.shape()
                Log.d(
                    "ExecuTorchApp",
                    "EValue[$i]: shape=${shape.contentToString()}, first few values=${data.take(10)}"
                )
            } catch (e: Exception) {
                Log.d("ExecuTorchApp", "EValue[$i]: Unable to extract tensor data: ${e.message}")
            }
        }

        if (testLabelBuffer != null) {
            val resultTextView: TextView = findViewById(R.id.resultTextView)
            val truthTextView: TextView = findViewById(R.id.truthTextView)
            resultTextView.text = "Loss: ${0.00}"
            truthTextView.text = "Ground truth: ${"None"}"
        } else {
            Log.e("ExecuTorchApp", "Error loading test label buffer")
        }

        // Set up button click listeners
        val fineTuneButton: Button = findViewById(R.id.fineTuneButton)
        val evaluateButton: Button = findViewById(R.id.evaluateButton)

        fineTuneButton.setOnClickListener {
            Log.d("Button", "Fine-tune button clicked")

            // Show progress container immediately on the UI thread
            runOnUiThread {
                try {
                    val statusText = findViewById<TextView>(R.id.statusText)
                    statusText.text = "Preparing for training..."
                    Log.d("StatusText", "Set progress container visible in button click")
                } catch (e: Exception) {
                    Log.e("StatusText", "Error in button click: ${e.message}")
                    e.printStackTrace()
                }
            }

            // Start fine-tuning with 5 epochs
            trainModel(
                tModule!!, trnImgData!!, trnLblData!!, tstImgData!!, tstLblData!!, 100, batchSize
            )
        }

        evaluateButton.setOnClickListener {
            // Evaluate the model
            evaluateModel(tModule!!, tstImgData!!, tstLblData!!, batchSize)
        }
    }

    private fun evaluateModel(
        model: TrainingModule,
        tstImgData: ByteArray,
        tstLblData: ByteArray,
        batchSize: Int,
        maxImagesToEvaluate: Int = 8000,
        updateUI: Boolean = true // Add parameter to control UI updates
    ): Double {
        val width = 32
        val height = 32
        val channels = 3

        // Start timing the evaluation process
        val startTime = System.currentTimeMillis()
        Log.d("ExecuTorchApp", "Starting evaluation")

        // Only update UI if not called from trainModel
        if (updateUI) {
            val statusText = findViewById<TextView>(R.id.statusText)

            // Make sure we're on the UI thread
            runOnUiThread {
                try {
                    statusText.text = "Evaluating model..."
                } catch (e: Exception) {
                    Log.e("StatusText", "Error showing progress container: ${e.message}")
                }
            }
        }

        var val_loss = 0.0
        var val_correct = 0
        var val_total = 0
        val tot_test_samples = tstImgData.size / (width * height * channels)
        val limited_test_samples = minOf(maxImagesToEvaluate, tot_test_samples)
        val num_test_batches = limited_test_samples / batchSize

        Log.d(
            "ExecuTorchApp",
            "Evaluating model on ${limited_test_samples} test images (${num_test_batches} batches) out of ${tot_test_samples} total images"
        )

        for (batch in 0 until num_test_batches) {
            // No need to update UI for each batch
            val startIdx = batch * batchSize * height * width * channels
            val endIdx = minOf((batch + 1) * batchSize * height * width * channels, tstImgData.size)
            val imgData = tstImgData.slice(startIdx until endIdx)

            val lblStartIdx = batch * batchSize
            val lblEndIdx = minOf((batch + 1) * batchSize, tstLblData.size)
            val lblData = tstLblData.slice(lblStartIdx until lblEndIdx)

            // Apply test transformations to the evaluation data
            // This includes normalization but no data augmentation
            val transformedData = ImageTransformations.applyBatchTestTransformations(
                imgData.toByteArray(), batchSize, width, height, channels
            )

            // Create a direct FloatBuffer as required by Tensor.fromBlob
            val buffer = Tensor.allocateFloatBuffer(transformedData.size)
            buffer.put(transformedData)
            buffer.rewind() // Reset position to the beginning of the buffer

            val batchTestImageTensor = Tensor.fromBlob(
                buffer, longArrayOf(
                    batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong()
                )
            )

            val batchTestLabelBuffer = LongArray(batchSize) { lblData[it].toLong() }
            val testLabelBuffer = Tensor.fromBlob(
                batchTestLabelBuffer, longArrayOf(batchSize.toLong())
            )

            val inputEValues =
                arrayOf(EValue.from(batchTestImageTensor), EValue.from(testLabelBuffer))
            val outputEValues = model.executeForwardBackward("forward", *inputEValues)
                ?: throw IllegalStateException("Execution module is not loaded.")

            // Extract the loss and prediction from the outputEValues
            val loss = outputEValues[0].toTensor().getDataAsFloatArray()[0]
            val predictions = outputEValues[1].toTensor().getDataAsLongArray()

            // Calculate accuracy
            for (i in predictions.indices) {
                if (predictions[i] == batchTestLabelBuffer[i]) {
                    val_correct += 1
                }
                val_total += 1
            }
            val_loss += loss
        }

        val val_accuracy = 100.0 * val_correct / val_total

        // Calculate evaluation execution time
        val endTime = System.currentTimeMillis()
        val duration = (endTime - startTime) / 1000.0 // Convert to seconds

        Log.d(
            "ExecuTorchApp",
            "Evaluation complete - Loss: ${val_loss / val_total * batchSize}, Accuracy: $val_accuracy%, " + "Time: ${
                String.format(
                    "%.2f", duration
                )
            } s, " + "Time per image: ${
                String.format(
                    "%.2f", (duration / val_total) * 1000
                )
            } ms"
        )

        // Update UI with the evaluation results and hide progress container only if updateUI is true
        if (updateUI) {
            runOnUiThread {
                try {
                    val resultTextView: TextView = findViewById(R.id.resultTextView)
                    resultTextView.text = "Evaluation Loss: ${val_loss / val_total * batchSize}"
                    val truthTextView: TextView = findViewById(R.id.truthTextView)
                    truthTextView.text = "Evaluation Accuracy: $val_accuracy%"
                } catch (e: Exception) {
                    Log.e("ResultTextView", "Error updating results: ${e.message}")
                }
            }
        }
        return val_accuracy
    }

    private fun trainModel(
        model: TrainingModule,
        trnImgData: ByteArray,
        trnLblData: ByteArray,
        tstImgData: ByteArray,
        tstLblData: ByteArray,
        epochs: Int,
        batchSize: Int,
        learningRate: Double = 0.001,
        momentum: Double = 0.9
    ) {

        val statusText = findViewById<TextView>(R.id.statusText)

        // Make sure we're on the UI thread
        runOnUiThread {
            try {
                statusText.text = "Training: Starting Epoch 1/$epochs"
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        val width = 32
        val height = 32
        val channels = 3
        val pixelsPerImage = width * height * channels
        val tot_train_samples = trnImgData.size / pixelsPerImage
        val num_batches = tot_train_samples / batchSize

        // Create a list of indices that we'll shuffle for each epoch
        val indices = (0 until tot_train_samples).toMutableList()

        for (epoch in 1..epochs) {
            // Start timing the epoch
            val epochStartTime = System.currentTimeMillis()

            // Shuffle the indices at the beginning of each epoch
            indices.shuffle()

            Log.d("ExecuTorchApp", "Starting Epoch $epoch/$epochs")

            // Update status text for new epoch
            runOnUiThread {
                try {
                    statusText.text = "Training: Epoch $epoch/$epochs"
                } catch (e: Exception) {
                    Log.e("ExecuTorchApp", "Error updating status text: ${e.message}")
                }
            }
            var epoch_loss = 0.0
            var train_correct = 0
            var train_total = 0

            Log.d(
                "ExecuTorchApp",
                "Total images to be trained: ${tot_train_samples}, Number of batches: ${num_batches}"
            )

            for (batch in 0 until num_batches) {
                val batchImgData = ByteArray(batchSize * pixelsPerImage)
                val batchLblData = ByteArray(batchSize)

                // Fill the batch with shuffled data
                for (i in 0 until batchSize) {
                    val idx = indices[batch * batchSize + i]

                    // Copy image data
                    System.arraycopy(
                        trnImgData,
                        idx * pixelsPerImage,
                        batchImgData,
                        i * pixelsPerImage,
                        pixelsPerImage
                    )

                    // Copy label data
                    batchLblData[i] = trnLblData[idx]
                }

                val imgData = batchImgData
                val lblData = batchLblData

                // Apply data augmentation transformations to the training data
                // This includes padding, random crop, horizontal flip, and normalization
                val transformedData = ImageTransformations.applyBatchTransformations(
                    imgData, batchSize, width, height, channels
                )

                // Create a direct FloatBuffer as required by Tensor.fromBlob
                val buffer = Tensor.allocateFloatBuffer(transformedData.size)
                buffer.put(transformedData)
                buffer.rewind() // Reset position to the beginning of the buffer
                val batchTrainImageTensor = Tensor.fromBlob(
                    buffer, longArrayOf(
                        batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong()
                    )
                )
                val batchTrainLabelBuffer =
                    LongArray(batchSize) { lblData?.get(it.toInt())?.toLong() ?: 0 }
                val trainLabelBuffer = Tensor.fromBlob(
                    batchTrainLabelBuffer, longArrayOf(batchSize.toLong())
                )
                val inputEValues =
                    arrayOf(EValue.from(batchTrainImageTensor), EValue.from(trainLabelBuffer))
                val outputEValues = model.executeForwardBackward("forward", *inputEValues)
                    ?: throw IllegalStateException("Execution module is not loaded.")
                // Extract the loss and prediction from the outputEValues
                val loss = outputEValues[0].toTensor().getDataAsFloatArray()[0]
                val predictions = outputEValues[1].toTensor().getDataAsLongArray()
                val parameters: Map<String, Tensor> = model.namedParameters("forward")
                val sgd = SGD.create(parameters, learningRate, momentum, 0.0, 0.0, true)
                val gradients: Map<String, Tensor> = model.namedGradients("forward")
                sgd.step(gradients)
                // Calculate accuracy
                for (i in predictions.indices) {
                    if (predictions[i] == batchTrainLabelBuffer[i]) {
                        train_correct += 1
                    }
                    train_total += 1
                }
                epoch_loss += loss
            }

            // Calculate epoch execution time
            val epochEndTime = System.currentTimeMillis()
            val epochDuration = (epochEndTime - epochStartTime) / 1000.0 // Convert to seconds

            Log.d(
                "ExecuTorchApp",
                "Epoch [$epoch/$epochs] Loss: ${epoch_loss / tot_train_samples * batchSize}, " + "Accuracy: ${100 * train_correct / train_total}%, " + "Time: ${
                    String.format(
                        "%.2f",
                        epochDuration
                    )
                } s, " + "Time per image: ${
                    String.format(
                        "%.2f", 1000 * (epochDuration / tot_train_samples)
                    )
                } ms"
            )

            // Evaluate the model using the test data and the evaluateModel function
            // Pass false for updateUI to prevent nested UI updates
            val val_accuracy = evaluateModel(model, tstImgData, tstLblData, batchSize, 1000, false)

            // Update UI with epoch results
            runOnUiThread {
                try {
                    val resultTextView: TextView = findViewById(R.id.resultTextView)
                    val truthTextView: TextView = findViewById(R.id.truthTextView)
                    resultTextView.text = "Training Accuracy: ${100 * train_correct / train_total}%"
                    truthTextView.text = "Validation Accuracy: $val_accuracy%"

                    // Update status text to show progress through epochs
                    statusText.text = "Completed Epoch $epoch/$epochs"
                } catch (e: Exception) {
                    Log.e("ResultTextView", "Error updating results: ${e.message}")
                }
            }
        }
    }
}
