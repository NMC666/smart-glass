/***************************************************************************//**
 * @file
 * @brief Audio classifier application config
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc.  Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement.  This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/

#ifndef AUDIO_CLASSIFIER_CONFIG_H
#define AUDIO_CLASSIFIER_CONFIG_H

#if __has_include("sl_tflite_micro_model_parameters.h")
  #include "sl_tflite_micro_model_parameters.h"
#endif


#define SMOOTHING_WINDOW_DURATION_MS 600
#define MINIMUM_DETECTION_COUNT     3
#define DETECTION_THRESHOLD         235
#define SUPPRESSION_TIME_MS         1500

#define SENSITIVITY .5f


#define IGNORE_UNDERSCORE_LABELS 1

#define DETECTION_LED sl_led_led1

#define ACTIVITY_LED sl_led_led0

#define VERBOSE_MODEL_OUTPUT_LOGS 1


#define INFERENCE_INTERVAL_MS 200

#define MAX_CATEGORY_COUNT    16


#define MAX_RESULT_COUNT      50

#define TASK_STACK_SIZE      512

#define TASK_PRIORITY         20


#if defined(SL_TFLITE_MODEL_CLASSES)
  #define CATEGORY_LABELS SL_TFLITE_MODEL_CLASSES
#else

  #define CATEGORY_LABELS { "hey_snips", "_unknown_", "_silence_" }
#endif


#endif
