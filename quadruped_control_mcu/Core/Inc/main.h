/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */
#define __round_uint(x) ((x) >= 0 ? (uint16_t)((x) + (float32_t)0.5) : (uint16_t)((x) - (float32_t)0.5))
#define __round_int(x) ((x) >= 0 ? (int16_t)((x) + (float32_t)0.5) : (int16_t)((x) - (float32_t)0.5))
/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define USER_Btn_Pin GPIO_PIN_13
#define USER_Btn_GPIO_Port GPIOC
#define USER_Btn_EXTI_IRQn EXTI15_10_IRQn
#define MCO_Pin GPIO_PIN_0
#define MCO_GPIO_Port GPIOH
#define SPI_Ready_Pin GPIO_PIN_0
#define SPI_Ready_GPIO_Port GPIOA
#define LD1_Pin GPIO_PIN_0
#define LD1_GPIO_Port GPIOB
#define LD3_Pin GPIO_PIN_14
#define LD3_GPIO_Port GPIOB
#define STLK_RX_Pin GPIO_PIN_8
#define STLK_RX_GPIO_Port GPIOD
#define STLK_TX_Pin GPIO_PIN_9
#define STLK_TX_GPIO_Port GPIOD
#define TMS_Pin GPIO_PIN_13
#define TMS_GPIO_Port GPIOA
#define TCK_Pin GPIO_PIN_14
#define TCK_GPIO_Port GPIOA
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
#define LD2_Pin GPIO_PIN_7
#define LD2_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */
// Number of joints
#define JOINTS 12

//Ranges for each type of joint
#define BODY_RANGE_MAX 15.0
#define BODY_RANGE_MIN -10.0
#define FEMUR_RANGE_MAX 30.0
#define FEMUR_RANGE_MIN -20.0
#define TIBIA_RANGE_MAX 15.0
#define TIBIA_RANGE_MIN -15.0

//delta angle between joint reference position of robot and simulation/solidworks
#define DELTA_REF_FEMUR 42.65021f
#define DELTA_REF_TIBIA 18.34979f
#define DELTA_REF_BODY  0

#define MID_POINT_BFR 96.1285553f
#define MID_POINT_BBR 80.5f
#define MID_POINT_BBL 85.4869385f
#define MID_POINT_TFR (69.0105972f + DELTA_REF_TIBIA)
#define MID_POINT_FFR (71.9034348f + DELTA_REF_FEMUR)
#define MID_POINT_BFL 93.9473724f
#define MID_POINT_FFL (134.817856f - DELTA_REF_FEMUR)
#define MID_POINT_TFL (120.135033f - DELTA_REF_TIBIA)
#define MID_POINT_FBL (130.440994f - DELTA_REF_FEMUR)
#define MID_POINT_TBL (122.615303f - DELTA_REF_TIBIA)
#define MID_POINT_TBR (55.4315529f + DELTA_REF_TIBIA)
#define MID_POINT_FBR (55.8714142f + DELTA_REF_FEMUR)

// 0PWM_BFR,1PWM_BBR, 2PWM_BBL, 3PWM_TFR, 4PWM_FFR, 5PWM_BFL, 6PWM_FFL, 7PWM_TFL, 8PWM_FBL, 9PWM_TBL, 10PWM_TBR, 11PWM_FBR

// WITH PID
// #define MID_POINT_BFR 93
// #define MID_POINT_BBR 81
// #define MID_POINT_BBL 84
// #define MID_POINT_TFR 85
// #define MID_POINT_FFR 82
// #define MID_POINT_BFL 90
// #define MID_POINT_FFL 84
// #define MID_POINT_TFL 93
// #define MID_POINT_FBL 91
// #define MID_POINT_TBL 93
// #define MID_POINT_TBR 79
// #define MID_POINT_FBR 81

//WITHOUT PID (comparing with later plastic reference)
// #define MID_POINT_BFR 94 96.1285553,
// #define MID_POINT_BBR 87 80.5,
// #define MID_POINT_BBL 86 85.4869385,
// #define MID_POINT_TFR 89 87.3603897,
// #define MID_POINT_FFR 87 114.55365,
// #define MID_POINT_BFL 94 93.9473724,
// #define MID_POINT_FFL 87 92.1676483,
// #define MID_POINT_TFL 93 101.78524,
// #define MID_POINT_FBL 94 87.7907867,
// #define MID_POINT_TBL 96 104.265511,
// #define MID_POINT_TBR 80 73.7813416,
// #define MID_POINT_FBR 85 98.5216217


#define TIM9_TICK (float32_t)0.01 //s
#define TIM5_TICK (float32_t)0.5 //ms

#define ONE_SECOND (float32_t)1000/TIM5_TICK  //milliseconds

// Parameters to transform Degrees(°) values to PWM (Ton in microseconds)
#define K_TON (float32_t)(1000.0 / 140.0)
#define BIAS_TON (float)500

// Parameters of the actuation and time-out algorithm
#define DEAD_BANDWIDTH_SERVO 3	// Degrees
#define MAX_DELTA_TARGET 2    // Degrees

#define TIMEOUT (float32_t)1000/TIM5_TICK    // miliseconds

#define ALL_FINISHED (uint16_t)0xFFF //(12 ones)

//Median Filter
#define WINDOW_SIZE 80
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
