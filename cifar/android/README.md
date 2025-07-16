# ExecuTorch Model Fine-Tuning on Android App

In this tutorial, we will be fine-tuning a CIFAR 10 model on an android app using ExecuTorch.

## Environment Setup

For the android environment setup, follow these steps:

1. Install Open JDK using: `brew install openjdk@17`
2. Download and install the latest version of [Android Studio](https://developer.android.com/studio/)  .
3. Start Android Studio and open the **Settings** dialog (gear icon on the bottom left).
4. Navigate to **Languages & Frameworks**, then **Android SDK**.
5. In the **SDK Platforms** tab, check **Android 14.0 (“UpsideDownCake”)** and **Android API 36**.
6. In the **SDK Tools** tab, check **Android SDK Build-Tools 36**, **NDK (Side by side)**, **Android SDK Command-line Tools (latest)**, **CMake**, **Android Emulator**, and **Android SDK Platform-Tools**.
7. Export the paths for these to your environment or add them to the `.bashrc` or `.zshrc` files:
```bash
# JAVA config
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/
export PATH="$JAVA_HOME/bin:$PATH"

# Android dev config
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
export PATH=$ANDROID_HOME/platform-tools:$PATH
export PATH=$ANDROID_HOME/emulator:$PATH
export PATH=$ANDROID_HOME/build-tools/latest:$PATH
export ANDROID_NDK=$ANDROID_HOME/ndk/28.0.12433566/

# Cmake config
export PATH=$ANDROID_HOME/cmake/4.0.2/bin:$PATH
```

**NOTE** For the updated steps for building the dependencies refer to the official repository over [here](https://github.com/pytorch/executorch/blob/main/extension/android/README.md).

To install ExecuTorch in a python environment we can use the following commands in a new terminal:

```bash
$ git clone https://github.com/pytorch/executorch.git
$ cd executorch
$ uv venv --seed --prompt et --python 3.10
$ source .venv/bin/activate
$ which python
$ git fetch origin
$ git submodule sync --recursive
$ git submodule update --init --recursive
$ ./install_requirements.sh
$ ./install_executorch.sh
$ sh ./scripts/build_android_library.sh
$ echo $ANDROID_HOME
```

If you run into errors for sdk path, complete the above steps in the [Trouble Shooting](#trouble-shooting) section before proceeding with the following steps:

```bash
$ sh ./scripts/build_android_library.sh
$ sh ./extension/android/executorch_android/android_test_setup.sh
$ ls ./extension/android/executorch_android/
$ cd extension/android
$ ./gradlew :executorch_android:testDebugUnitTest
$ ./gradlew :executorch_android:connectedAndroidTest
```

**NOTE:** This fails without a connected android device. If you run into this issue, launch your emulator or connect the android device to your laptop.

```bash
$ ./gradlew :executorch_android:connectedAndroidTest
```

Finally, the `.aar` file can be found here:

```bash
<PARENT_DIRECTORY>/executorch/extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar
```

**Note:** We will rename this file to `executorch.aar` and copy it into the `libs` directory of the android app.

## Creation of Android App

1. Start with a new empty project in android studio:
    ![](./images/Pasted%20image%2020250709162820.png)

2. Set the following configurations for the project:
    ![](./images/Pasted%20image%2020250709163001.png)

3. You'll be presented with a setup looking somewhat like this (after the build is complete and you select Project from the dropdown menu on the top left):
    ![](./images/Pasted%20image%2020250709163834.png)

4. Create a new directory named '`layout`' inside '`res`' and then create '`activity_main.xml`' file in the layout directory with [this](./app/src/main/res/layout/activity_main.xml) content.
    - Copy the following content of [this](./app/src/main/res/layout/activity_main.xml) XML script into the `activity_main.xml` file to get the following layout with two buttons for Fine-Tuning and Model Evaluation:
    ![](./images/Pasted%20image%2020250709164234.png)

5. Create a new directory named assets in the [main](./app/src/main) directory (you should automatically be presented with the option to select the assets directory from the gradle source set when you create the new directory) ![Image](./images/Pasted%20image%2020250709164842.png)

6. Copy the binary files (`train_data.bin` and `test_data.bin`) generated during the execution of the CIFAR 10 example on [ExecuTorch official repo](https://github.com/pytorch/executorch/tree/main/extension/training/examples/CIFAR) into this [directory](./app/src/main/assets/cifar-10-batches-bin) using the following command:

    ```bash
    (base) USERNAME@USERNAME-mbp ~ % cp -r cifar-10-batches-bin /Users/<USERNAME>/AndroidStudioProjects/DemoCIFAR10/app/src/main/assets
    ```
    **Note:** The example code can be run with the following command:
    ```bash
    python3 main.py --model-path cifar10_model.pth --pte-model-path cifar10_model_pte_only.pte --split-pte-model-path cifar10_model.pte --save-pt-json cifar10_pt.json --save-et-json cifar10_et.json --ptd-model-dir . --epochs 10 --fine-tune-epochs 50
    ```

7. Edit the `build.gradle.kts` file inside the [app](./app) directory to have [this](./app/build.gradle.kts) content:

8. Create the `ImageTransformations.kt` object file inside the [java/com/example/democifar10](./app/src/main/java/com/example/democifar10/) directory as shown here:
    ![ImageTransformations](./images/Pasted%20image%2020250709165757.png)

- Content for the `ImageTransformations.kt` file can be found [here](app/src/main/java/com/example/democifar10/ImageTransformations.kt).

9. Create the `Cifar10ImageExtractor.kt` class file in the [java/com/example/democifar10](./app/src/main/java/com/example/democifar10) directory![](./images/Pasted%20image%2020250709170006.png)

- Content for the `Cifar10ImageExtractor.kt` class can be found [here](./app/src/main/java/com/example/democifar10/Cifar10ImageExtractor.kt).

10. Copy the assets generated in during the execution of the [CIFAR 10 example](https://github.com/pytorch/executorch/tree/main/extension/training/examples/CIFAR) into the assets directory using the following commands:

    ```bash
    (base) USERNAME@USERNAME-mbp ~ % cp generic_cifar.ptd /Users/<USERNAME>/AndroidStudioProjects/DemoCIFAR10/app/src/main/assets

    (base) USERNAME@USERNAME-mbp ~ % cp generic_cifar.pte /Users/<USERNAME>/AndroidStudioProjects/DemoCIFAR10/app/src/main/assets

    (base) USERNAME@USERNAME-mbp ~ % cp executorch.aar /Users/<USERNAME>/AndroidStudioProjects/DemoCIFAR10/app/libs
    ```

11. Finally edit the `MainActivity.kt` file with the code from [here](./app/src/main/java/com/example/democifar10/MainActivity.kt):

12. Sync your Gradle build: ![](./images/Pasted%20image%2020250709170528.png)


- You'll be notified that the build was successful: ![](./images/Pasted%20image%2020250709171142.png)


13. Click on run to proceed: ![](./images/Pasted%20image%2020250709170837.png)

14. Click on the fine-tune button and the training begins for `150` epochs![[Pasted image 20250709171210.png]]

    **Note:** The training parameters can be tweaked in `MainActivity.kt`

### Summary:

We trained the PyTorch model for `1 epoch` and exported the `.pte` and `.ptd` files. We started with training loss of `2.159132883071899`, training accuracy of `18%`, testing loss of `2.1072397136688235`, and testing accuracy of `29%`

```log
2025-07-09 17:11:46.309 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting Epoch 1/150
2025-07-09 17:11:46.309 13277-13277 ExecuTorchApp           com.example.democifar10              D  Total images to be trained: 500, Number of batches: 125
2025-07-09 17:11:47.237 13277-13428 EGL_emulation           com.example.democifar10              D  app_time_stats: avg=7.23ms min=2.75ms max=25.70ms count=60
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Epoch [1/150] Loss: 2.159132883071899, Accuracy: 18%, Time: 1.41 s, Time per image: 2.81 ms
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting evaluation
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluating model on 100 test images (25 batches) out of 100 total images
2025-07-09 17:11:47.989 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluation complete - Loss: 2.1072397136688235, Accuracy: 29.0%, Time: 0.27 s, Time per image: 2.73 ms
```

We reached a training loss of `1.7837489886283875`, training accuracy of `35%`, testing loss of `2.0501016211509704`, and testing accuracy of `38%` after `150 epochs`

```log
2025-07-09 17:13:59.889 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting Epoch 150/150
2025-07-09 17:13:59.889 13277-13277 ExecuTorchApp           com.example.democifar10              D  Total images to be trained: 500, Number of batches: 125
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Epoch [150/150] Loss: 1.7837489886283875, Accuracy: 35%, Time: 0.78 s, Time per image: 1.56 ms
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting evaluation
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluating model on 100 test images (25 batches) out of 100 total images
2025-07-09 17:14:00.809 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluation complete - Loss: 1.7236163129806519, Accuracy: 38%, Time: 0.67 s, Time per image: 1.35 ms
```

### Trouble Shooting

Check if the `local.properties` file is present in the `extension/android` directory:

```bash
$ cat <PARENT_DIRECTORY>/executorch/extension/android/local.properties
```

**NOTE:** If this file (`local.properties`) is missing, the build process will fail because the script can't retrieve the path for the android sdk from the env variables. If you encounter this issue, please follow the following steps:

```bash
$ touch <PARENT_DIRECTORY>/executorch/extension/android/local.properties
$ vim <PARENT_DIRECTORY>/executorch/extension/android/local.properties # Add the path to your sdk directory into this file like: sdk.dir=/Users/<USERNAME>/Library/Android/sdk
```
