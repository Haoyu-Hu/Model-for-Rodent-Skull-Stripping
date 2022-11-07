# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pxg4xg0r8f1mz5AKsH8GSbZ4D2C4HCHi
"""

#!/usr/bin/env python

import SimpleITK as sitk 
import os
import sys

def uint_baco(args):
  if not args.input or not args.output:
      print("Usage: Directory of N4BiasFieldCorrection inputImage (folder/image name) " +
            "Directory of outputImage [shrinkFactor] [maskImage] [numberOfIterations] " +
            "[numberOfFittingLevels]")
      print("please input the path of input image and output image!")
      sys.exit(1)

  inputImage = sitk.ReadImage(args.input, sitk.sitkFloat32)
  image = inputImage

  if args.mask:
      maskImage = sitk.ReadImage(args.mask)
  else:
      maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

  if args.sf:
      image = sitk.Shrink(inputImage, [int(args.sf)] * inputImage.GetDimension())
      maskImage = sitk.Shrink(maskImage, [int(args.sf)] * inputImage.GetDimension())

  corrector = sitk.N4BiasFieldCorrectionImageFilter()

  numberFittingLevels = 4

  if args.nfl:
      numberFittingLevels = int(args.nfl)

  if args.iters:
      corrector.SetMaximumNumberOfIterations([int(args.iters)] * numberFittingLevels)

  corrected_image = corrector.Execute(image, maskImage)


  log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

  corrected_image_full_resolution = inputImage / sitk.Exp( log_bias_field )

  sitk.WriteImage(corrected_image, args.output)

  # if ("SITK_NOSHOW" not in os.environ):
  #     sitk.Show(corrected_image, "N4 Corrected")

def batching_baco(args):
  if isinstance(args.input, str):
    if os.path.isfile(args.input):
      path_input, img_name = os.path.split(args.input)
      print("Processing", img_name)
      if not isinstance(args.output, str) or not os.path.exists(args.output):
        if not os.path.exists(os.path.join(args.input, 'baco_img')):
          os.makedirs(os.path.join(args.input, 'baco_img'))
        args.output = os.path.join(args.input, 'baco_img/baco_'+img_name)
      uint_baco(args)
      return args.output
    elif os.path.isdir(args.input):
      input_path = args.input
      output_path = args.output
      mask_path = args.mask
      file_input = os.listdir(input_path)
      file_input.sort()
      if isinstance(mask_path, str):
        if os.path.isdir(mask_path):
          file_mask = os.listdir(mask_path)
          file_mask.sort()
      else:
        file_mask = None
      counter = 0
      for file in file_input:
        if ".nii" in file:
          print("Processing", file)
          args.input = os.path.join(input_path, file)
          if not isinstance(output_path, str):
            if not os.path.exists(os.path.join(input_path, 'baco_img')):
              os.makedirs(os.path.join(input_path, 'baco_img'))
            output_path = os.path.join(input_path, 'baco_img')
            args.output = os.path.join(input_path, 'baco_img/baco_'+file)
          else:
            args.output = os.path.join(output_path, 'baco_'+file)
          if file_mask:
            args.mask = os.path.join(mask_path, file_mask[counter])
          counter = counter+1
          uint_baco(args)
      return output_path
    else:
      print("Invalid input!")
      sys.exit(1)
  else:
    print("Invalid Input Path!")
    sys.exit(1)