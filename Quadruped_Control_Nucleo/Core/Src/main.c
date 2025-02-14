/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <string.h>
#include <arm_math.h>
#include "calibration_ADC.h"

#include "SignalProcessing.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define M_PI (float32_t)3.14159265358979323846

// Number of joints
#define JOINTS 12

//Ranges for each type of joint
#define BODY_RANGE_MAX 15.0
#define BODY_RANGE_MIN -10.0
#define FEMUR_RANGE_MAX 30.0
#define FEMUR_RANGE_MIN -20.0
#define TIBIA_RANGE_MAX 15.0
#define TIBIA_RANGE_MIN -15.0
// 0PWM_BFR,1PWM_BBR, 2PWM_BBL, 3PWM_TFR, 4PWM_FFR, 5PWM_BFL, 6PWM_FFL, 7PWM_TFL, 8PWM_FBL, 9PWM_TBL, 10PWM_TBR, 11PWM_FBR
#define MID_POINT_BFR 93
#define MID_POINT_BBR 81
#define MID_POINT_BBL 84
#define MID_POINT_TFR 85
#define MID_POINT_FFR 82
#define MID_POINT_BFL 90
#define MID_POINT_FFL 84
#define MID_POINT_TFL 93
#define MID_POINT_FBL 91
#define MID_POINT_TBL 93
#define MID_POINT_TBR 79
#define MID_POINT_FBR 81


// States for Main State Machine
#define RESET 0
#define TX_RASPBERRY 1
#define RX_RASPBERRY 2
#define ACTUATION 3
#define ERROR 4

// States for Actuation State Machine
#define RESET_ACTUATION 0
#define DELAY_STATE 1
#define COMPARE_MEASURE 2
#define TIMEOUT_STATE 3

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
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define __round_uint(x) ((x) >= 0 ? (uint16_t)((x) + (float32_t)0.5) : (uint16_t)((x) - (float32_t)0.5))
#define __round_int(x) ((x) >= 0 ? (int16_t)((x) + (float32_t)0.5) : (int16_t)((x) - (float32_t)0.5))
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

SPI_HandleTypeDef hspi3;
DMA_HandleTypeDef hdma_spi3_tx;
DMA_HandleTypeDef hdma_spi3_rx;

TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim9;

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
// For tx floats under UART (Serial plot)
typedef union float2byte
{
  float angle;
  uint8_t angle_bytes[4];
} float2byte;

//Median filter
volatile float32_t filter_in_arm_angle[12];
volatile float32_t median_filteredValue[JOINTS];
volatile spMedianFilter median_filter[JOINTS];

//SPI
volatile uint8_t spi_rx_cplt = 0;
//uint8_t spi_tx_cplt = 0;

// PID Variables
volatile float32_t error = 0;
volatile float32_t previous_error[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile float32_t control_signal = 0;
volatile float32_t error_acum[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile float32_t error_dif = 0;
volatile uint16_t pwm_pid_out[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// PID constants
float32_t kp[JOINTS] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
float32_t kd[JOINTS] = {0,0,0,0,0,0,0,0,0,0,0,0};
float32_t ki[JOINTS] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};

// Offset of each joint
const float32_t mid_point_joints[12] = {MID_POINT_BFR, MID_POINT_BBR, MID_POINT_BBL, MID_POINT_TFR, MID_POINT_FFR, MID_POINT_BFL, MID_POINT_FFL, MID_POINT_TFL, MID_POINT_FBL, MID_POINT_TBL, MID_POINT_TBR, MID_POINT_FBR};

// Global Flags for state machines
volatile uint8_t up_down_vector[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // To know if the servos have to go up or down: 1 is up, 0 is down

// Arrays used to set the pwm of each servo in a for loop
TIM_HandleTypeDef* htim[3];
uint32_t channel[4] = {TIM_CHANNEL_1, TIM_CHANNEL_2, TIM_CHANNEL_3, TIM_CHANNEL_4};

const float32_t joint_range[12][2] = {{BODY_RANGE_MIN, BODY_RANGE_MAX},
                                    {-BODY_RANGE_MAX, -BODY_RANGE_MIN},
                                    {-BODY_RANGE_MAX, -BODY_RANGE_MIN},
                                    {TIBIA_RANGE_MIN, TIBIA_RANGE_MAX},
                                    {FEMUR_RANGE_MIN, FEMUR_RANGE_MAX},
                                    {BODY_RANGE_MIN, BODY_RANGE_MAX},
                                    {-FEMUR_RANGE_MAX, -FEMUR_RANGE_MIN},
                                    {-TIBIA_RANGE_MAX, -TIBIA_RANGE_MIN},
                                    {-FEMUR_RANGE_MAX, -FEMUR_RANGE_MIN},
                                    {-TIBIA_RANGE_MAX, -TIBIA_RANGE_MIN},
                                    {TIBIA_RANGE_MIN, TIBIA_RANGE_MAX},
                                    {FEMUR_RANGE_MIN, FEMUR_RANGE_MAX}};

// Global Buffers
// Buffer rx SPI
volatile float32_t target_joint[12] ={MID_POINT_BFR, MID_POINT_BBR, MID_POINT_BBL, MID_POINT_TFR, MID_POINT_FFR, MID_POINT_BFL, MID_POINT_FFL, MID_POINT_TFL, MID_POINT_FBL, MID_POINT_TBL, MID_POINT_TBR, MID_POINT_FBR};
volatile uint16_t raw_angle_ADC[2][12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // ADC measurement with DMA

volatile uint8_t current_buffer = 0;

// Ticks of timer
volatile uint8_t pid_sample = 0;
volatile uint8_t send_uart = 0;
volatile uint16_t timeout = 0;
volatile uint16_t time_debounce = 0;  //To debounce user button

// Init program at reset with user button
volatile uint8_t init_nucleo = 0;

// Calibration variables
const float32_t *calibration_table[12] = {&ADC_VALUES_SERVO_0[0][0], &ADC_VALUES_SERVO_1[0][0], &ADC_VALUES_SERVO_2[0][0], &ADC_VALUES_SERVO_3[0][0], &ADC_VALUES_SERVO_4[0][0], &ADC_VALUES_SERVO_5[0][0], &ADC_VALUES_SERVO_6[0][0], &ADC_VALUES_SERVO_7[0][0], &ADC_VALUES_SERVO_8[0][0], &ADC_VALUES_SERVO_9[0][0], &ADC_VALUES_SERVO_10[0][0], &ADC_VALUES_SERVO_11[0][0]};

/////////////////////////////

//Para pruebas
// volatile uint16_t debug_delay = 0;
volatile uint32_t delta_time;
float test_quadruped_nucleo[12] = {MID_POINT_BFR, MID_POINT_BBR, MID_POINT_BBL, MID_POINT_TFR, MID_POINT_FFR, MID_POINT_BFL, MID_POINT_FFL, MID_POINT_TFL, MID_POINT_FBL, MID_POINT_TBL, MID_POINT_TBR, MID_POINT_FBR};

// VARIABLES QUE NO DEBEN SER GLOBALES, pero es mas cómodo para debuggear

// podrían ser locales
volatile float32_t delta_target = 0;
volatile float32_t f_joint_angle[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile float32_t spi_transmit_rpi[JOINTS] =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile uint8_t uart_tx_buffer[2 + 12 * 4 + 12 * 4 + 12 * 4 + 12 * 4];

// podrían ser estáticas locales
volatile uint8_t state_actuation = RESET_ACTUATION;
volatile uint16_t joints_finished = ALL_FINISHED; // Each bit is a flag for a joint: 1-Finished, 0-Unfinished
volatile int8_t step_complete = 1;           // returned by State_Machine_Actuation

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_SPI3_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM9_Init(void);
/* USER CODE BEGIN PFP */
float adc2angle(uint16_t adc_value, uint8_t up_down, const float32_t *table_ptr);
uint16_t angle2ton_us(float);
float ton_us2angle(uint16_t ton_us);
void move_servos(uint8_t joint, uint16_t ton);

int8_t State_Machine_Actuation(void);
void State_Machine_Control(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// CAMBIAR NOMBRES PARA QUE SEA MAS DESCRIPTIVO
float2byte u_dummy1;
float2byte u_dummy2;
float2byte u_dummy3;
float2byte u_dummy4;
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM2_Init();
  MX_USART3_UART_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_SPI3_Init();
  MX_TIM5_Init();
  MX_TIM9_Init();
  /* USER CODE BEGIN 2 */
	while(init_nucleo == 0) HAL_Delay(10);

  HAL_TIM_Base_Start(&htim2);
  HAL_TIM_Base_Start(&htim2);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);

  HAL_TIM_Base_Start(&htim3);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);

  HAL_TIM_Base_Start(&htim4);
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_4);

  for (uint8_t joint = 0; joint < 12; joint++)
  {
		median_filter[joint] = spCreateMedianFilter(WINDOW_SIZE);
		pwm_pid_out[joint] = angle2ton_us(mid_point_joints[joint]);
  }
    

  while (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)raw_angle_ADC, 12 * 2) == HAL_BUSY);

  // Initialize htim vector used to set pwm in for loops
  htim[0] = &htim2;
  htim[1] = &htim3;
  htim[2] = &htim4;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  // Main state
  uart_tx_buffer[0] = 0xFF;
  uart_tx_buffer[1] = 0xFF;

  // Joints set to default initial angle
  for (uint8_t joint = 0; joint < JOINTS; joint++)
  {
    move_servos(joint, angle2ton_us(mid_point_joints[joint]));
  }

  HAL_Delay(5000);

  while (HAL_TIM_Base_Start_IT(&htim5) == HAL_BUSY);
  
  while (HAL_TIM_Base_Start_IT(&htim9) == HAL_BUSY);
  
  while (1)
  {
//    for (uint8_t joint = 0; joint < JOINTS; joint++)
//			__HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], angle2ton_us(test_quadruped_nucleo[joint]));


		//memcpy(target_joint, test_quadruped_nucleo, sizeof(test_quadruped_nucleo));

    //memcpy(dummy1, &raw_angle_ADC[buffer_to_copy][0], sizeof(dummy1));
    
    //Serial Plot
    if (send_uart)
    {
      for (int i = 0; i < JOINTS; i++)
      {
				u_dummy1.angle = target_joint[i];
        //u_dummy2.angle = adc2angle(median_filteredValue[i], up_down_vector[i], calibration_table[i]);
				//u_dummy3.angle = adc2angle(filter_in_arm_angle[i], up_down_vector[i], calibration_table[i]);
        u_dummy2.angle = ton_us2angle(pwm_pid_out[i]);
				u_dummy3.angle = error_acum[i];
        u_dummy4.angle = f_joint_angle[i];

        uart_tx_buffer[16 * i + 2] = u_dummy1.angle_bytes[3];
        uart_tx_buffer[16 * i + 3] = u_dummy1.angle_bytes[2];
        uart_tx_buffer[16 * i + 4] = u_dummy1.angle_bytes[1];
        uart_tx_buffer[16 * i + 5] = u_dummy1.angle_bytes[0];

        uart_tx_buffer[16 * i + 6] = u_dummy2.angle_bytes[3];
        uart_tx_buffer[16 * i + 7] = u_dummy2.angle_bytes[2];
        uart_tx_buffer[16 * i + 8] = u_dummy2.angle_bytes[1];
        uart_tx_buffer[16 * i + 9] = u_dummy2.angle_bytes[0];
				
				uart_tx_buffer[16 * i + 10] = u_dummy3.angle_bytes[3];
        uart_tx_buffer[16 * i + 11] = u_dummy3.angle_bytes[2];
        uart_tx_buffer[16 * i + 12] = u_dummy3.angle_bytes[1];
        uart_tx_buffer[16 * i + 13] = u_dummy3.angle_bytes[0];
				
				uart_tx_buffer[16 * i + 14] = u_dummy4.angle_bytes[3];
        uart_tx_buffer[16 * i + 15] = u_dummy4.angle_bytes[2];
        uart_tx_buffer[16 * i + 16] = u_dummy4.angle_bytes[1];
        uart_tx_buffer[16 * i + 17] = u_dummy4.angle_bytes[0];
      }
			HAL_UART_Transmit(&huart3, (const uint8_t*) uart_tx_buffer, 16 * JOINTS + 2, HAL_MAX_DELAY);

      send_uart = 0;
    }
		
    //HAL_Delay(1); // Adjust delay as necessary

    State_Machine_Control();
  }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 96;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ENABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 12;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SEQ_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_4;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_13;
  sConfig.Rank = 2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = 3;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_11;
  sConfig.Rank = 4;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_14;
  sConfig.Rank = 5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = 6;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_5;
  sConfig.Rank = 7;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_6;
  sConfig.Rank = 8;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_10;
  sConfig.Rank = 9;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = 10;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_12;
  sConfig.Rank = 11;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_9;
  sConfig.Rank = 12;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief SPI3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI3_Init(void)
{

  /* USER CODE BEGIN SPI3_Init 0 */

  /* USER CODE END SPI3_Init 0 */

  /* USER CODE BEGIN SPI3_Init 1 */

  /* USER CODE END SPI3_Init 1 */
  /* SPI3 parameter configuration*/
  hspi3.Instance = SPI3;
  hspi3.Init.Mode = SPI_MODE_SLAVE;
  hspi3.Init.Direction = SPI_DIRECTION_2LINES;
  hspi3.Init.DataSize = SPI_DATASIZE_16BIT;
  hspi3.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi3.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi3.Init.NSS = SPI_NSS_SOFT;
  hspi3.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi3.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi3.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi3.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI3_Init 2 */

  /* USER CODE END SPI3_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 96-1;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 3100;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_ENABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 96-1;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 3100;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_ENABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 96-1;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 3100;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_ENABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */
  HAL_TIM_MspPostInit(&htim4);

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 96-1;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 500;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief TIM9 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM9_Init(void)
{

  /* USER CODE BEGIN TIM9_Init 0 */

  /* USER CODE END TIM9_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};

  /* USER CODE BEGIN TIM9_Init 1 */

  /* USER CODE END TIM9_Init 1 */
  htim9.Instance = TIM9;
  htim9.Init.Prescaler = 96-1;
  htim9.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim9.Init.Period = 10000;
  htim9.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim9.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim9) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim9, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM9_Init 2 */

  /* USER CODE END TIM9_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 256000;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream2_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream2_IRQn);
  /* DMA1_Stream7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream7_IRQn);
  /* DMA2_Stream4_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream4_IRQn, 2, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream4_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SPI_Ready_Pin */
  GPIO_InitStruct.Pin = SPI_Ready_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(SPI_Ready_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
  current_buffer = 0;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
  current_buffer = 1;
}

void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
  if(hspi == &hspi3)
  {
    spi_rx_cplt = 1;
  }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	if(GPIO_Pin == USER_Btn_Pin && init_nucleo == 0)
	{
		init_nucleo = 1;
    time_debounce = ONE_SECOND;
	}else if(GPIO_Pin == USER_Btn_Pin && init_nucleo == 1 && time_debounce == 0)
  {
    step_complete = 1;  //disable PID

		for (uint8_t joint = 0; joint < JOINTS; joint++)
		
			__HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], angle2ton_us(mid_point_joints[joint]));


		HAL_Delay(1000);  //Delay to ignore bouncing of user button, and establish the default position before reset

		//Turn off pwm without interrupting ton.
		for(uint8_t i= 0; i<JOINTS;i++)
		{  
			//wait for ton to finish
			while(__HAL_TIM_GET_COUNTER(htim[i / 4]) < __HAL_TIM_GET_COMPARE(htim[i / 4], channel[i % 4]));
			
			HAL_TIM_PWM_Stop(htim[i / 4], channel[i % 4]);
		}

		HAL_NVIC_SystemReset();
  }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *timer)
{
  // static float32_t t_step = 0;
	//static int sign = 1;

  if (timer == &htim5)
  {
    if (timeout > 0)
      timeout--;

    if (time_debounce > 0)
      time_debounce--;
		
		// if (debug_delay > 0)
		// 	debug_delay--;

    uint8_t buffer_to_copy = current_buffer;

		// uint32_t init_value = __HAL_TIM_GetCounter(&htim5);

    for (uint8_t joint = 0; joint < 12; joint++)
    {
      filter_in_arm_angle[joint] = raw_angle_ADC[buffer_to_copy][joint];
			median_filteredValue[joint] = spMedianFilterInsert(median_filter[joint], filter_in_arm_angle[joint]);  //Median Filter
    }
		// uint32_t final_value = __HAL_TIM_GetCounter(&htim5);
		
		// delta_time = final_value - init_value;
    // for controlling trajectory
    // t_step += 0.0005;
		
//		if(t_step >=3)
//		{
//			sign = - sign;
//			t_step = 0.0;
//			target_joint[10] = -5 * sign+ mid_point_joints[10];
//			//target_joint[11] = 5 * sign + mid_point_joints[11];
//			target_joint[3] = -5 * sign + mid_point_joints[3];
//			//target_joint[4] = 5 * sign + mid_point_joints[4];
//			target_joint[9] = 5 * sign + mid_point_joints[9];
//			//target_joint[8] = -5 * sign + mid_point_joints[8];
//			target_joint[7] = 5 * sign+ mid_point_joints[7];
//			//target_joint[6] = -5 * sign+ mid_point_joints[6];
//			for(uint8_t i = 0;i<JOINTS;i++)		
//				move_servos(i, angle2ton_us(target_joint[i]));
//		}	
		
			
//			// // Tibia   Back    Right
//			target_joint[10] = -10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[10];

//			// // Femur   Back    Right
//			target_joint[11] = 10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[11];

//			// // Tibia Front Right
//			target_joint[3] = -10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[3];
//			// //Femur Front Right
//			target_joint[4] = 10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[4];

//			// //Tibia Back left
//			target_joint[9] = 10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[9];

//			// //Femur Back left
//			target_joint[8] = -10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[8];

//			// //Tibia Front left
//			target_joint[7] = 10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[7];
//			// //Femur Front left
//			target_joint[6] = -10 * sin(2 * M_PI * 0.5 * t_step) + mid_point_joints[6];
//			
//			for(uint8_t i = 0;i<JOINTS;i++)		
//				move_servos(i, angle2ton_us(target_joint[i]));
		

  }

  if (timer == &htim9)
  {
		// Move servos
		for (uint8_t joint = 0; joint < 12; joint++)
		{

			f_joint_angle[joint] = adc2angle(median_filteredValue[joint], up_down_vector[joint], calibration_table[joint]);
			
			if (step_complete == 0)
			{
				error = target_joint[joint] - f_joint_angle[joint];
				error_acum[joint] += error * TIM9_TICK;
				error_dif = (error - previous_error[joint]) / TIM9_TICK; //(divido tick de tim9)

				previous_error[joint] = error;
        
				control_signal = kp[joint] * error + ki[joint] * error_acum[joint] + kd[joint] * error_dif; //Delta_angle
				up_down_vector[joint] = control_signal > 0;

        pwm_pid_out[joint] = pwm_pid_out[joint] + __round_int(control_signal);
				
        //If pid_out is below min range, move to min range
				if(ton_us2angle(pwm_pid_out[joint]) < (mid_point_joints[joint] + joint_range[joint][0]))
					move_servos(joint, angle2ton_us(mid_point_joints[joint] + joint_range[joint][0]));

        //If pid_out is above max range, move to max range
				else if(ton_us2angle(pwm_pid_out[joint]) > (mid_point_joints[joint] + joint_range[joint][1]))
					move_servos(joint, angle2ton_us(mid_point_joints[joint] + joint_range[joint][1]));
        
        //Else move it to pid_out
				else
					move_servos(joint, pwm_pid_out[joint]);
			}
		
		}
    pid_sample = 1;
    send_uart = 1;
	}
}

// Convert ADC_value of ACCUMULATED measurement to angle in degrees using linear interpolation
float adc2angle(uint16_t adc_value, uint8_t up_down, const float32_t *table_ptr)
{
  uint16_t index = 0;

  const float32_t *ADC_VALUES = table_ptr;

  while (adc_value > ADC_VALUES[up_down * MED_CALIB + index] / 5) // Busco el c�digo en la tabla obtenida en la calibracion
  {
    index++;
    if (index > MED_CALIB - 1)
      return 266; //adc_value greater than maximum value of table
  }

  if (adc_value == ADC_VALUES[up_down * MED_CALIB + index] / 5) // Si el c�digo est� en la tabla, devuelvo el angulo correspondiente
    return ANGLES[index];

  else if (index == 0)
    return 0; // adc_value lesser than minimum value of table

  else // Si es menor al codigo de la tabla, realizo una interpolacion lineal
    return (ANGLES[index - 1] * (ADC_VALUES[up_down * MED_CALIB + index] / 5 - adc_value) + ANGLES[index] * (adc_value - ADC_VALUES[up_down * MED_CALIB + index - 1] / 5)) / (ADC_VALUES[up_down * MED_CALIB + index] / 5 - ADC_VALUES[up_down * MED_CALIB + index - 1] / 5);
}

uint16_t angle2ton_us(float angle_value)
{
  uint16_t ton_us = 0;

  ton_us = __round_uint(angle_value * K_TON + BIAS_TON);

  return ton_us;
}
float ton_us2angle(uint16_t ton_us)
{
  float angle_value = 0.0;

  angle_value = (ton_us - BIAS_TON) / K_TON;

  return angle_value;
}
void move_servos(uint8_t joint, uint16_t ton)
{
  __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton);
}



void State_Machine_Control(void)
{
  static uint8_t state = TX_RASPBERRY;
	// static int sign = 1;

  switch (state)
  {
    case TX_RASPBERRY:
			// if(debug_delay == 0)
			{
				// Request master to transmit target step rotation and joint angles
				// debug_delay = 3000/TIM5_TICK;
				
				for(uint8_t joint = 0; joint<JOINTS ; joint++)
					spi_transmit_rpi[joint] = f_joint_angle[joint] - mid_point_joints[joint];
        
				HAL_SPI_Transmit_DMA(&hspi3, (uint8_t*)spi_transmit_rpi, 12*2);
				HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
				HAL_Delay(1);
				HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);

				state = RX_RASPBERRY;
			}
      break;
    case RX_RASPBERRY:
      //As reference:
      // Order of joints, Nucleo:
      // 0 - BFR
      // 1 - BBR
      // 2 - BBL
      // 3 - TFR
      // 4 - FFR
      // 5 - BFL
      // 6 - FFL
      // 7 - TFL
      // 8 - FBL
      // 9 - TBL
      // 10 - TBR
      // 11 - FBR

      // Order of joints, Raspberry:
      // 0 - BFR
      // 1 - FFR
      // 2 - TFR
      // 3 - BFL
      // 4 - FFL
      // 5 - TFL
      // 6 - BBR
      // 7 - FBR
      // 8 - TBR
      // 9 - BBL
      // 10 - FBL
      // 11 - TBL

      // Request master to receive the next action
      
      while(HAL_SPI_Receive_DMA(&hspi3, (uint8_t*)target_joint, 12*2) == HAL_BUSY);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
      HAL_Delay(1);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);

      while(spi_rx_cplt == 0);
      spi_rx_cplt = 0;
			
      //Add offset to each joint
			for (uint8_t joint = 0; joint < JOINTS; joint++)
			{
				target_joint[joint] = target_joint[joint] + mid_point_joints[joint];
			}
      
      HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, (GPIO_PinState)0);
      HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, (GPIO_PinState)0);
      
      //*Only for testing without raspberry
			// sign = - sign;
			// target_joint[10] = -5 * sign+ mid_point_joints[10];
			// //target_joint[11] = 5 * sign + mid_point_joints[11];
			// target_joint[3] = -5 * sign + mid_point_joints[3];
			// //target_joint[4] = 5 * sign + mid_point_joints[4];
			// target_joint[9] = 5 * sign + mid_point_joints[9];
			// //target_joint[8] = -5 * sign + mid_point_joints[8];
			// target_joint[7] = 5 * sign+ mid_point_joints[7];
			// //target_joint[6] = -5 * sign+ mid_point_joints[6];
			

      state = ACTUATION;

      break;
    case ACTUATION:

      step_complete = State_Machine_Actuation();

      if (step_complete == 1)
				state = TX_RASPBERRY;
			
      else if (step_complete == -1)
        //state = ERROR;
				state = TX_RASPBERRY;
        //state = TX_ERROR;   // TODO: ESTADO STOP DE EMERGENCIA EN EL FUTURO
			
      break;
    case ERROR:
      HAL_Delay(1);
      break;
  }
}

int8_t State_Machine_Actuation(void)
{
  switch (state_actuation)
  {
  case RESET_ACTUATION:

    // Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
    // Therefore consider the joint already finished
    for (uint8_t joint = 0; joint < JOINTS; joint++)
    {
      delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
      
      if (delta_target >= DEAD_BANDWIDTH_SERVO)
      {
        // Evaluate if servo has to go up or down to get a better measurement
        up_down_vector[joint] = (f_joint_angle[joint] < target_joint[joint]);
        
        joints_finished &= ~(1 << joint); // Set respective bit in 0

        //*Solo si no se usa PID
        //move_servos(joint, angle2ton_us(target_joint[joint]));
      }
    }
    if (joints_finished == ALL_FINISHED) // If no joint has to move we skip the actuation machine
    {
      HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, (GPIO_PinState)0);
      HAL_GPIO_WritePin(LD1_GPIO_Port, LD3_Pin, (GPIO_PinState)1);
      return 1;
    }
    else
    {
      // Clear PID buffers
      memset((void*)error_acum, 0, sizeof(error_acum));
      memset((void*)previous_error, 0, sizeof(previous_error));

      state_actuation = DELAY_STATE;
      timeout = TIMEOUT;
      pid_sample = 0;
      step_complete = 0;
    }
    break;

  case DELAY_STATE:
    if (timeout == 0)
      state_actuation = TIMEOUT_STATE;
    else if (pid_sample)
      state_actuation = COMPARE_MEASURE;

    break;
  case COMPARE_MEASURE:
    if (timeout == 0)
      state_actuation = TIMEOUT_STATE;

    else
    {
      for (uint8_t joint = 0; joint < JOINTS; joint++)
      {
        if ((joints_finished & (1 << joint)) == 0) // Ignore joints that finished
        {
          delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);

          if (delta_target < MAX_DELTA_TARGET) // If the stable joint reached target
          {
            joints_finished |= 1 << joint; // Set respective bit in 1

            if (joints_finished == ALL_FINISHED)
            {
              state_actuation = RESET_ACTUATION;
              HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, (GPIO_PinState)0);
              HAL_GPIO_WritePin(LD1_GPIO_Port, LD3_Pin, (GPIO_PinState)1);
              return 1;
            }
          }
        }
      }
      pid_sample = 0;
      state_actuation = DELAY_STATE;
      // state_actuation = COMPARE_MEASURE;
    }

    break;
  case TIMEOUT_STATE:
    state_actuation = RESET_ACTUATION;

    HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, (GPIO_PinState)0);
    HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, (GPIO_PinState)1);

    // Reset all joints
    // joints_finished = ALL_FINISHED;
    // for (uint8_t joint = 0; joint < JOINTS; joint++)
    // {
    //   __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], angle2ton_us(90));
    // }
    return -1;
  }
  return 0;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
