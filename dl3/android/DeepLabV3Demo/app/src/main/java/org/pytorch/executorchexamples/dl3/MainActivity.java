/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorchexamples.dl3;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

public class MainActivity extends Activity implements Runnable {
  private ImageView mImageView;
  private Button mButtonXnnpack;
  private ProgressBar mProgressBar;
  private Bitmap mBitmap = null;
  private Module mModule = null;
  private String mImagename = "corgi.jpeg";

  private String[] mImageFiles;
  private int mCurrentImageIndex = 0;
  // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of
  // classes with indexes
  private static final int CLASSNUM = 21;
  private static final int DOG = 12;
  private static final int PERSON = 15;
  private static final int SHEEP = 17;

  private void populateImage() {
    try {
      mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
      mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
      mImageView.setImageBitmap(mBitmap);
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
    }
  }

  private void populateImagePathFromAssets() {
    try {
          String[] allFiles = getAssets().list("");
          ArrayList<String> imageList = new ArrayList<>();
          for (String file : allFiles) {
            if (file.endsWith(".jpg") || file.endsWith(".jpeg") || file.endsWith(".png")) {
              imageList.add(file);
            }
          }
          mImageFiles = imageList.toArray(new String[0]);
          mCurrentImageIndex = 0;
          mImagename = mImageFiles.length > 0 ? mImageFiles[0] : null;
        } catch (IOException e) {
          Log.e("ImageSegmentation", "Error listing assets", e);
          finish();
        }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    populateImagePathFromAssets();

    try {
      mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename), null, null);
      mBitmap = Bitmap.createScaledBitmap(mBitmap, 224, 224, true);
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error reading assets", e);
      finish();
    }

    mModule = Module.load("/data/local/tmp/dl3_xnnpack_fp32.pte");

    mImageView = findViewById(R.id.imageView);
    mImageView.setImageBitmap(mBitmap);

    final Button buttonNext = findViewById(R.id.nextButton);
    buttonNext.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            if (mImageFiles == null || mImageFiles.length == 0) {
              // No images available
              return;
            }
            // Move to the next image, wrap around if at the end
            mCurrentImageIndex = (mCurrentImageIndex + 1) % mImageFiles.length;
            mImagename = mImageFiles[mCurrentImageIndex];
            populateImage();
          }
        });

    mButtonXnnpack = findViewById(R.id.xnnpackButton);
    mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
    mButtonXnnpack.setOnClickListener(
        new View.OnClickListener() {
          public void onClick(View v) {
            mModule.destroy();
            mModule = Module.load("/data/local/tmp/dl3_xnnpack_fp32.pte");
            mButtonXnnpack.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonXnnpack.setText(getString(R.string.run_model));

            Thread thread = new Thread(MainActivity.this);
            thread.start();
          }
        });

    final Button resetImage = findViewById(R.id.resetImage);
    resetImage.setOnClickListener(
            v -> populateImage());
  }

  @Override
  public void run() {
    final Tensor inputTensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            mBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    final long startTime = SystemClock.elapsedRealtime();
    Tensor outputTensor = mModule.forward(EValue.from(inputTensor))[0].toTensor();
    final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
    Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

    final float[] scores = outputTensor.getDataAsFloatArray();
    int width = mBitmap.getWidth();
    int height = mBitmap.getHeight();

    int[] intValues = new int[width * height];
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int maxi = 0, maxj = 0, maxk = 0;
        double maxnum = -Double.MAX_VALUE;
        for (int i = 0; i < CLASSNUM; i++) {
          float score = scores[i * (width * height) + j * width + k];
          if (score > maxnum) {
            maxnum = score;
            maxi = i;
            maxj = j;
            maxk = k;
          }
        }
        if (maxi == PERSON) intValues[maxj * width + maxk] = 0xFFFF0000; // R
        else if (maxi == DOG) intValues[maxj * width + maxk] = 0xFF00FF00; // G
        else if (maxi == SHEEP) intValues[maxj * width + maxk] = 0xFF0000FF; // B
        else intValues[maxj * width + maxk] = 0xFF000000;
      }
    }

    Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
    Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    outputBitmap.setPixels(
        intValues,
        0,
        outputBitmap.getWidth(),
        0,
        0,
        outputBitmap.getWidth(),
        outputBitmap.getHeight());
    final Bitmap transferredBitmap =
        Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

    runOnUiThread(
            () -> {
              mImageView.setImageBitmap(transferredBitmap);
              mButtonXnnpack.setEnabled(true);
              mButtonXnnpack.setText(R.string.run_xnnpack);
              mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            });
  }
}
