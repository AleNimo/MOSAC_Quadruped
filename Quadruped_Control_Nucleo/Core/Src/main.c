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
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define M_PI 3.14159265358979323846

//Number of joints
#define JOINTS 12

//Number of ADC measurements for average
#define N_SAMPLES 10

//States for Main State Machine
#define RESET 						0
#define TX_RASPBERRY 			1
#define RX_RASPBERRY 			2
#define ACTUATION		 			3

//States for Step State Machine
#define RESET_ACTUATION		0
#define DELAY_STATE 			1
#define COMPARE_MEASURE		2
#define TIMEOUT_STATE     3

//Parameters to transform Degrees(�) values to PWM (Ton in microseconds)
#define K_TON_1 (float)(100.0/9.0)
#define BIAS_TON_1	(float)500
	
#define K_TON_2 (float)(38.0/3.0)
#define BIAS_TON_2	(float)(350.0)

//Parameters of the time-out algorithm
//(150 microseconds is the delay between 2 average measurements)

#define DEAD_BANDWIDTH_SERVO 3  //Degrees
#define MAX_DELTA_ANGLE	2	//Degrees		(960 micro radians is the quantum of the ADC in angle. 680 micro radians is the minimum change detectable measuring with 150us of delay and the speed of the servo)
#define MAX_DELTA_SAMPLE	0.5	

#define SAMPLE_TIME 50	//miliseconds
#define TIMEOUT 1000 //miliseconds

#define UMBRAL_DONE 100

#define ALL_FINISHED pow(2,JOINTS)-1

//For calibration
#define PWM_STEP 95
#define PWM_MIN 350  //FOR PRO Servo
#define PWM_MAX 2630 //FOR PRO Servo
#define MED_CALIB (int)__round(((float)(PWM_MAX-PWM_MIN)/PWM_STEP)+1)
#define UPDATE_PWM 0
#define MEASURE 1

#define LENGTH_TABLE_1 21
#define LENGTH_TABLE_2 25

#define NUM_TAPS 200

#define NUM_STAGE_IIR 1
#define NUM_ORDER_IIR (NUM_STAGE_IIR*2)
#define NUM_STD_COEFS 5 // b0, b1, b2, a1, a2
#define ALPHA 0.01



/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define __round(x) ((x)>=0?(uint16_t)((x)+0.5):(uint16_t)((x)-0.5))
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
DMA_HandleTypeDef hdma_usart3_tx;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */
//FIR FILTER Variables
arm_fir_instance_f32 fir_instance[12];
float32_t fir_in_arm[12], fir_state[12][NUM_TAPS];

float32_t fir_coeff[NUM_TAPS] = {0};

//IIR FILTER Variables
static float32_t iirState[12][NUM_ORDER_IIR];

static float32_t iirCoeffs[NUM_STAGE_IIR * NUM_STD_COEFS] = 
{
    ALPHA, 0, 0, -(ALPHA-1), 0
};

arm_biquad_cascade_df2T_instance_f32 S [12];
	

//Global Flags
uint8_t conv_cplt = 1;
uint8_t angulo_ready = 0;

uint8_t up_down_ADC[JOINTS] = {0};

//Global Buffers
//	SPI (SI NO USAMOS DMA NO HACE FALTA GLOBAL)
float target_joint[12] = {0};	//Buffer in SPI

//Arrays for changing pwm ton
TIM_HandleTypeDef* htim[3];
uint32_t channel[4] = {TIM_CHANNEL_1,TIM_CHANNEL_2, TIM_CHANNEL_3, TIM_CHANNEL_4};//{TIM_CHANNEL_1, TIM_CHANNEL_2, TIM_CHANNEL_3, TIM_CHANNEL_4};

//	ADC-DMA
uint16_t joint_angle_dma[2][12] = {0};	//ADC measurement with DMA
float32_t joint_angle[12] = {0};				//Accumulator for average
uint8_t current_buffer = 0;
//Ticks of timer
uint16_t sample_time = 0;
uint16_t timeout = 0;
uint16_t time = 0; // for calibrating Servo

//VARIABLES PARA DEBUG///////
#define ANGULO_INICIAL 0
HAL_StatusTypeDef status_spi;
uint16_t delay_button = 0;
//uint32_t counter = 0;
//uint32_t last_counter = 0;
//uint32_t delay_counter = 0;
float resultado;

uint16_t ton[JOINTS] = {0};

uint8_t joint = 0;

//Variables de Calibracion
uint16_t calibrated_joints_up[MED_CALIB][12] = {0};  //For storing the results of calibration up
uint16_t calibrated_joints_down[MED_CALIB][12] = {0};  //For storing the results of calibration down

uint8_t joint_adc = 0;

const float32_t *calibration_table[12] = {&ADC_VALUES_SERVO_9[0][0],&ADC_VALUES_SERVO_10[0][0],&ADC_VALUES_SERVO_11[0][0],&ADC_VALUES_SERVO_12[0][0],&ADC_VALUES_SERVO_1[0][0],&ADC_VALUES_SERVO_2[0][0],&ADC_VALUES_SERVO_3[0][0],&ADC_VALUES_SERVO_4[0][0],&ADC_VALUES_SERVO_5[0][0],&ADC_VALUES_SERVO_6[0][0],&ADC_VALUES_SERVO_7[0][0],&ADC_VALUES_SERVO_8[0][0]};
uint8_t col_sel[12] = {LENGTH_TABLE_1, LENGTH_TABLE_2, LENGTH_TABLE_2, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1, LENGTH_TABLE_1};

/////////////////////////////



//VARIABLES QUE NO DEBEN SER GLOBALES

float delta_sample = 0;
float delta_target = 0;
float f_joint_angle[JOINTS] = {ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL};//, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL, ANGULO_INICIAL};
//float f_joint_angle[JOINTS] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.11, 11.12, 12.13};
float f_last_joint[JOINTS] = {0};
float f_joint_angle_aux[JOINTS] = {0};
uint8_t uart_tx_buffer[12*2+12*2+2];

uint8_t state_actuation = RESET_ACTUATION;

uint8_t count_stable_signal[JOINTS] = {0};

uint16_t joints_finished = 0x000 ;	//Each bit is a flag for a joint






/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_SPI3_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM9_Init(void);
/* USER CODE BEGIN PFP */
float adc2angle(uint16_t adc_value, uint8_t up_down,const float32_t* table_ptr,uint8_t col_sel);
uint16_t angle2ton_us(float,uint8_t);

int8_t State_Machine_Actuation(void);
void State_Machine_Calibration(void);
void State_Machine_Control(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

uint16_t dummy1[12];
uint16_t dummy2[12];
uint8_t buffer_ready = 0;
uint16_t ton_dummy = PWM_MIN;
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
  MX_USB_OTG_FS_PCD_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_SPI3_Init();
  MX_TIM5_Init();
  MX_TIM9_Init();
  /* USER CODE BEGIN 2 */
	HAL_TIM_Base_Start(&htim2);
	HAL_TIM_Base_Start(&htim2);
  HAL_TIM_PWM_Start(&htim2,TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim2,TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim2,TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim2,TIM_CHANNEL_4);

  HAL_TIM_Base_Start(&htim3);
  HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_4);

  HAL_TIM_Base_Start(&htim4);
  HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_4);
	/*
	for(uint16_t taps = 0; taps<NUM_TAPS; taps++)
		fir_coeff[taps] = (float)1/NUM_TAPS;
	*/
	for(uint8_t joint = 0; joint<12;joint++)
		//arm_fir_init_f32(&(fir_instance[joint]),NUM_TAPS,fir_coeff,fir_state[joint],1);
		arm_biquad_cascade_df2T_init_f32(&S[joint], NUM_STAGE_IIR, &iirCoeffs[0], &iirState[joint][0]);
	
	

	
	while(HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12*2)==HAL_BUSY);
	//while(HAL_UART_Transmit_DMA(&huart3, uart_tx_buffer, 1 * 2) == HAL_BUSY);

	while(HAL_TIM_Base_Start_IT(&htim5)==HAL_BUSY);
	while(HAL_TIM_Base_Start_IT(&htim9)==HAL_BUSY);

  htim[0] = &htim2;
  htim[1] = &htim3;
  htim[2] = &htim4;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	
	//Main state
	for(joint = 0; joint < JOINTS; joint++)
		__HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], PWM_MIN);
	
	HAL_Delay(2000);	

	// ton = angle2ton_us(ANGULO_INICIAL);
	
	 //First measure of joints before transmition
	conv_cplt = 0;
	memset(joint_angle, 0, sizeof(joint_angle));

	
	
  uint8_t samples = 0;
	uint8_t buffer_to_copy = 0;
	uint8_t current_buffer = 0;
	uart_tx_buffer[0] = 0xFF;
	uart_tx_buffer[1] = 0xFF;
	
  while(1)
  {
		/*
		for(joint = 0; joint < JOINTS; joint++)
		__HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton_dummy);
		*/
		//__HAL_TIM_SET_COMPARE(htim[1 / 4], channel[1 % 4], ton_dummy);
    //State_Machine_Calibration();
		/*
    if(samples == N_SAMPLES)
    {
      samples = 0;
      conv_cplt = 1;
    }
    else if(conv_cplt == 0)
    {
      for(uint8_t joint = 0; joint < JOINTS;joint++) joint_angle[joint] += joint_angle_dma[joint];
      samples++;
    }*/


		
		memcpy(dummy1,&joint_angle_dma[buffer_to_copy][0],sizeof(dummy1));
		//memcpy(dummy2,(uint16_t*)joint_angle,sizeof(dummy2));
		for(uint8_t joint = 0; joint<12;joint++) 
		{
			//f_joint_angle_aux[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);

			dummy2[joint] = (uint16_t)joint_angle[joint];
			//dummy1[i] = joint_angle_dma[buffer_to_copy][i];
		}
		
		if(sample_time == 0)
		{
			for (int i = 0; i < 4; i++) 
			{
					uart_tx_buffer[4*i+2] = (dummy1[i] >> 8) & 0xFF;
					uart_tx_buffer[4*i + 3] = dummy1[i] & 0xFF;
					uart_tx_buffer[4*i + 4] = (dummy2[i] >> 8) & 0xFF;
					uart_tx_buffer[4*i + 5] = dummy2[i] & 0xFF;
					
			}
			
			HAL_UART_Transmit(&huart3,uart_tx_buffer,4*4+2,HAL_MAX_DELAY);
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
  sConfig.Channel = ADC_CHANNEL_14;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = 2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = 3;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_4;
  sConfig.Rank = 4;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_5;
  sConfig.Rank = 5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_6;
  sConfig.Rank = 6;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = 7;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_9;
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
  sConfig.Channel = ADC_CHANNEL_11;
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
  sConfig.Channel = ADC_CHANNEL_13;
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
  htim2.Init.Period = 20000;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
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
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.Pulse = 0;
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
  htim3.Init.Period = 20000;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
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
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
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
  htim4.Init.Period = 20000;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
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
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
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
  htim5.Init.Period = 1000;
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
  htim9.Init.Period = 100;
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
  huart3.Init.BaudRate = 115200;
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
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_FS_PCD_Init(void)
{

  /* USER CODE BEGIN USB_OTG_FS_Init 0 */

  /* USER CODE END USB_OTG_FS_Init 0 */

  /* USER CODE BEGIN USB_OTG_FS_Init 1 */

  /* USER CODE END USB_OTG_FS_Init 1 */
  hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
  hpcd_USB_OTG_FS.Init.dev_endpoints = 6;
  hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.battery_charging_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_OTG_FS_Init 2 */

  /* USER CODE END USB_OTG_FS_Init 2 */

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
  /* DMA1_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
  /* DMA1_Stream7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream7_IRQn);
  /* DMA2_Stream4_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream4_IRQn, 0, 0);
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
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);

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

  /*Configure GPIO pin : PG6 */
  GPIO_InitStruct.Pin = GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc)
{
	current_buffer = 0;
	//memcpy(dummy1,&joint_angle_dma[0][0],sizeof(dummy1));
	/*
	uart_tx_buffer[0] = (dummy1[0] >> 8) & 0xFF;
	uart_tx_buffer[1] = dummy1[0] & 0xFF;
	HAL_UART_Transmit_DMA(&huart3, uart_tx_buffer, 1 * 2);*/
	
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
	current_buffer = 1;
	//memcpy(dummy1,&joint_angle_dma[1][0],sizeof(dummy1));
	/*
	uart_tx_buffer[0] = (dummy1[0] >> 8) & 0xFF;
	uart_tx_buffer[1] = dummy1[0] & 0xFF;
	HAL_UART_Transmit_DMA(&huart3, uart_tx_buffer, 1 * 2);*/
}

/*
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if(GPIO_Pin == USER_Btn_Pin)
    {
			if(delay_button == 0)
			{
				if(ton >= PWM_MAX) 
				{

					sign = -1;
				}
				else if(ton <= PWM_MIN) 
				{
					sign = 1;
				}
				ton += sign*PWM_STEP;

        for(joint = 0; joint < JOINTS; joint++)
		      __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton);

				//__HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, ton);
				
				time = 1000;
				delay_button = 500;
				angulo_ready = 1;
			}
    }
}
*/

// void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
// {
// 	if(hadc == &hadc1)
// 	{	
// 		cant_med++;
		
// 		for(uint8_t joint=0; joint<JOINTS; joint++) joint_angle[joint] += joint_angle_dma[joint];

// 		if(cant_med == N_SAMPLES)
// 		{
			
// 			//Para medir tiempo entre muestras promediadas en microsegundos
// 			//counter = __HAL_TIM_GetCounter(&htim5);
// 			//delay_counter = counter - last_counter;
// 			//last_counter = counter;
			

// 			//reset counter and accumulator
// 			cant_med = 0;

// 			conv_cplt = 1;
// 		}
// 		else
// 		{
// 			HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
// 		}
// 	}
// }



void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	static float angulo_salida = 0;
	static int8_t sign = 1;
	static uint8_t step = 10;
	
    if(GPIO_Pin == USER_Btn_Pin)
    {
			if(delay_button == 0)
			{
				
				if(angulo_salida >= 180)
				{
					sign = -1;
					angulo_salida = 180;
				}
				else if(angulo_salida <= 0)
				{
					sign = 1;
					angulo_salida = 0;
				}
				angulo_salida += sign*step;
				
				if(angulo_salida >= 180)
					angulo_salida = 180;
				else if(angulo_salida <= 0)
					angulo_salida = 0;
				
				//target_joint[0] = angulo_salida;
				//target_joint[1] = angulo_salida;
				//target_joint[2] = angulo_salida;
        //target_joint[3] = angulo_salida;
				
				angulo_ready = 1;
				
				delay_button = 500;
			}
    }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
	if(htim == &htim5)
	{
		if(sample_time > 0) sample_time--;
		if(timeout > 0) timeout--;
		
		//for debugging
		if(delay_button > 0) delay_button--;

    //for calibration
    if(time>0)time--;
		
		//for(uint8_t joint = 0; joint<12;joint++) f_joint_angle_aux[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
		uint8_t buffer_to_copy = current_buffer;
		
		//memcpy(fir_in_arm,(float32_t*)&joint_angle_dma[buffer_to_copy][0],12);
		
		for(uint8_t joint = 0; joint<12;joint++)
		{
			fir_in_arm[joint] = joint_angle_dma[buffer_to_copy][joint];
			arm_biquad_cascade_df2T_f32 (&S[joint], &fir_in_arm[joint], &joint_angle[joint] , 1);    // perform filtering
			//arm_fir_f32(&fir_instance[joint],&fir_in_arm[joint],&joint_angle[joint],1);	
		}

	}
	if(htim == &htim9)
	{
		
		
		
	}

}

//Convert ADC_value of ACCUMULATED measurement to angle in degrees using linear interpolation
float adc2angle(uint16_t adc_value, uint8_t up_down,const float32_t* table_ptr,uint8_t col_sel)
{
	uint16_t index = 0;
	
	const float32_t *ADC_VALUES = table_ptr;

  if(col_sel == LENGTH_TABLE_2) // Servo Pro decreases adc values as angle increases
    while(adc_value < ADC_VALUES[up_down*col_sel + index]/10)	//Busco el c�digo en la tabla obtenida en la calibracion
    {
      index++;
      if(index > col_sel-1)
        return 180;	//adc_value greater than maximum value of table
    }
  else
    while(adc_value > ADC_VALUES[up_down*col_sel + index]/10)	//Busco el c�digo en la tabla obtenida en la calibracion
    {
      index++;
      if(index > col_sel-1)
        return 180;	//adc_value greater than maximum value of table
    }


	if(adc_value == ADC_VALUES[up_down*col_sel + index]/10)	//Si el c�digo est� en la tabla, devuelvo el angulo correspondiente
		if(col_sel==LENGTH_TABLE_1)
			return ANGLES_1[index];
		else
			return ANGLES_2[index];

	else if(index == 0) return 0;	//adc_value lesser than minimum value of table
	
	else	//Si es menor al codigo de la tabla, realizo una interpolacion lineal
		if(col_sel == LENGTH_TABLE_1)
			return (ANGLES_1[index-1] * (ADC_VALUES[up_down*col_sel + index]/10-adc_value) + ANGLES_1[index] * (adc_value-ADC_VALUES[up_down*col_sel + index-1]/10)) / (ADC_VALUES[up_down*col_sel + index]/10 - ADC_VALUES[up_down*col_sel + index-1]/10);
		else
			return (ANGLES_2[index-1] * (ADC_VALUES[up_down*col_sel + index]/10-adc_value) + ANGLES_2[index] * (adc_value-ADC_VALUES[up_down*col_sel + index-1]/10)) / (ADC_VALUES[up_down*col_sel + index]/10 - ADC_VALUES[up_down*col_sel + index-1]/10);

}

uint16_t angle2ton_us(float angle_value,uint8_t col_sel)
{
	uint16_t  ton_us = 0;
	
	if(col_sel == LENGTH_TABLE_1)
		ton_us = __round(angle_value * K_TON_1 + BIAS_TON_1);
	else if(col_sel == LENGTH_TABLE_2)
		ton_us = __round(angle_value * K_TON_2 + BIAS_TON_2);
	
	return ton_us;
}
/*
void State_Machine_Calibration(void)
{

	static uint8_t state = UPDATE_PWM;
	static uint8_t first_time = 1;
	static uint16_t pwm_value = 0;

  static int8_t sign = 1;
  static uint16_t ton = PWM_MIN;

  static uint8_t calibration_direction = 1;
	
	static uint8_t cant_med = 0;
	
  //static uint8_t cant_med = 0;

	//CALIBRACION ADC
	switch(state)
	{
		case UPDATE_PWM:
			if(first_time == 0)
			{
				if(ton >= PWM_MAX) 
				{
					calibration_direction = 0;
					sign = -1;
				}
				else if(ton <= PWM_MIN) 
				{
					calibration_direction = 1;
					sign = 1;
				}
				ton += sign*PWM_STEP;
				pwm_value+=sign;
			}
			first_time = 0;
			for(uint8_t joint = 0; joint < JOINTS; joint++)
				__HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton);

			time = 1000;
			
			state = MEASURE;
		case MEASURE:
      if(cant_med == N_SAMPLES)
			{
				cant_med = 0;
				if(calibration_direction == 1)
				{						
					memcpy(calibrated_joints_up[pwm_value], joint_angle, sizeof(joint_angle));
				}
				
				else
				{
					memcpy(calibrated_joints_down[pwm_value], joint_angle, sizeof(joint_angle));
				}
				
				memset(joint_angle, 0, sizeof(joint_angle));
				state = UPDATE_PWM;
			}
			else if(time == 0)
			{
				for(uint8_t joint = 0; joint < JOINTS;joint++) joint_angle[joint] += joint_angle_dma[joint];
				time = 1000;
				cant_med++;
			}
  } 

}*/

void State_Machine_Control(void)
{
	static uint8_t state = RESET;
	
	int8_t result = 0;
	
  switch(state)
  {
    case RESET:
								
			for(joint = 0; joint < JOINTS; joint++)
				f_joint_angle[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);	//there is no way to define the up_down vector, initial values are used
			state = TX_RASPBERRY;

      break;
    case TX_RASPBERRY:
      
      //Request master to transmit target step rotation and joint angles
      HAL_SPI_Transmit_DMA(&hspi3, (uint8_t*)f_joint_angle, 12*2);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
      HAL_Delay(1);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);	//POSIBLE PROBLEMA ACA (tiempo que tarda el master en iniciar desde que lee el 1 en el GPIO)
      
      state = RX_RASPBERRY;

      break;
    case RX_RASPBERRY:
      //Request master to receive the next action
      while(HAL_SPI_Receive_DMA(&hspi3, (uint8_t*)target_joint, 12*2) == HAL_BUSY);
      if(angulo_ready)
      {
        angulo_ready = 0;
        HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
        HAL_Delay(1);
        HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);	//IDEM PROBLEMA ANTERIOR
        joints_finished = 0;
        //Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
        //Therefore consider the joint already finished
				HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 0);
				HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 0);
        for(joint=0;joint<JOINTS;joint++)
        {
          delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
          if(delta_target < DEAD_BANDWIDTH_SERVO)
            joints_finished |= 1<<joint;	//Set respective bit in 1
          else
						ton[joint] = angle2ton_us(target_joint[joint],col_sel[joint]);
            __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton[joint]);  //Move the servomotor
        }
        
        //Start first measure of joints before ACTUATION
        //memset(joint_angle, 0, sizeof(joint_angle));
        //HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
      
        //Evaluate if servo has to go up or down to get a better measurement
        for(joint=0;joint<JOINTS;joint++)
          up_down_ADC[joint] = (f_joint_angle[joint] < target_joint[joint]);

        state = ACTUATION;
      }
      break;
    case ACTUATION:
      result = State_Machine_Actuation();
      
      if(result == 1)
        state = TX_RASPBERRY;
      else if(result == -1)
        state = TX_RASPBERRY;	//ESTADO STOP DE EMERGENCIA EN EL FUTURO
      break;
  }
}
//Nueva Version
int8_t State_Machine_Actuation(void)
{
	switch(state_actuation)
	{
		case RESET_ACTUATION:

			
			for(joint = 0; joint < JOINTS; joint++)
				f_last_joint[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
							
			sample_time = SAMPLE_TIME;
			state_actuation = DELAY_STATE;
			//state_actuation = COMPARE_MEASURE;
			timeout = TIMEOUT;

			break;
		
		case DELAY_STATE:
			if(timeout == 0)
			  state_actuation = TIMEOUT_STATE;
      else if(sample_time == 0)
			{
				//memset(joint_angle, 0, sizeof(joint_angle));
				//HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
				state_actuation = COMPARE_MEASURE;
			}
			break;
		case COMPARE_MEASURE:
			if(timeout == 0)
        state_actuation = TIMEOUT_STATE;

      else
			{
				for(joint = 0; joint < JOINTS; joint++)
				{
					if((joints_finished & (1<<joint)) == 0)	//Ignore joints that finished
					{
						f_joint_angle[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
						
						delta_sample = fabs(f_joint_angle[joint] - f_last_joint[joint]);
						
						if(delta_sample < MAX_DELTA_SAMPLE)//Not Moving	
						{						
							delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
							
							if(delta_target < MAX_DELTA_ANGLE)  //If the stable joint reached target
							{
								joints_finished |= 1<<joint;	//Set respective bit in 1
								
								if(joints_finished == ALL_FINISHED)
								{
									state_actuation = RESET_ACTUATION;
									HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 0);
									HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 1);
									return 1;
								}
							}
						}
						else
							
							timeout = TIMEOUT;

					}
				}

				memcpy(f_last_joint, f_joint_angle, sizeof(f_joint_angle));	//Remember joint

				sample_time = SAMPLE_TIME;
				state_actuation = DELAY_STATE;
				//state_actuation = COMPARE_MEASURE;
			}
			
			break;
    case TIMEOUT_STATE:
      state_actuation = RESET_ACTUATION;
      
      HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 0);
      HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 1);
				
      //Turn off joints that got stuck
      for(joint=0; joint<JOINTS; joint++)
      {
        if((joints_finished & (1<<joint)) == 0)
          __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], 0);
      }
      return -1;
	}
	return 0;
}
/*
int8_t State_Machine_Actuation(void)
{
	switch(state_actuation)
	{
		case RESET_ACTUATION:

			memset(count_stable_signal, 0, sizeof(count_stable_signal));
			
			for(joint = 0; joint < JOINTS; joint++)
				f_last_joint[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
							
			sample_time = SAMPLE_TIME;
			state_actuation = DELAY_STATE;
			timeout = TIMEOUT;

			break;
		case DELAY_STATE:
			if(timeout == 0)
			  state_actuation = TIMEOUT_STATE;
      else if(sample_time == 0)
			{
				//memset(joint_angle, 0, sizeof(joint_angle));
				//HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
				state_actuation = COMPARE_MEASURE;
			}
			break;
		case COMPARE_MEASURE:
			if(timeout == 0)
        state_actuation = TIMEOUT_STATE;

      else
			{
				for(joint = 0; joint < JOINTS; joint++)
				{
					if((joints_finished & (1<<joint)) == 0)	//Ignore joints that finished
					{
						f_joint_angle[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
						
						delta_sample = fabs(f_joint_angle[joint] - f_last_joint[joint]);
						
						if(delta_sample < MAX_DELTA_ANGLE)
						{
							count_stable_signal[joint]++;
							if(count_stable_signal[joint] == UMBRAL_DONE && timeout>0)  //If the joint signal is stable
							{
								count_stable_signal[joint] = 0;

                delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
								if(delta_target < MAX_DELTA_ANGLE)  //If the stable joint reached target
								{
									joints_finished |= 1<<joint;	//Set respective bit in 1
									
									if(joints_finished == ALL_FINISHED)
									{
										joints_finished = 0;
                    state_actuation = RESET_ACTUATION;
										HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 0);
                    HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 1);
										return 1;
									}
								}
							}
						}
						else
							count_stable_signal[joint] = 0;
					}
				}

				memcpy(f_last_joint, f_joint_angle, sizeof(f_joint_angle));	//Remember joint

				sample_time = SAMPLE_TIME;
				state_actuation = DELAY_STATE;
			}
			
			break;
    case TIMEOUT_STATE:
      state_actuation = RESET_ACTUATION;
      
      HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 0);
      HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 1);
				
      //Turn off joints that got stuck
      for(joint=0; joint<JOINTS; joint++)
      {
        if((joints_finished & (1<<joint)) == 0)
          __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], 0);
      }

      joints_finished = 0;
      return -1;
	}
	return 0;
}*/

/*
void State_Machine_Control(void)
{
	static uint8_t state = RESET;
	
	int8_t result = 0;
	
  switch(state)
  {
    case RESET:
      if(conv_cplt)
      {
        conv_cplt = 0;
                  
        for(joint = 0; joint < JOINTS; joint++)
          f_joint_angle[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);	//there is no way to define the up_down vector, initial values are used
        state = TX_RASPBERRY;
      }
      break;
    case TX_RASPBERRY:
      
      //Request master to transmit target step rotation and joint angles
      //HAL_SPI_Transmit_DMA(&hspi3, (uint8_t*)f_joint_angle, 12*2);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
      HAL_Delay(1);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);	//POSIBLE PROBLEMA ACA (tiempo que tarda el master en iniciar desde que lee el 1 en el GPIO)
      
      state = RX_RASPBERRY;

      break;
    case RX_RASPBERRY:
      //Request master to receive the next action
      //while(HAL_SPI_Receive_DMA(&hspi3, (uint8_t*)target_joint, 12*2) == HAL_BUSY);
      if(angulo_ready)
      {
        angulo_ready = 0;
        HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
        HAL_Delay(1);
        HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);	//IDEM PROBLEMA ANTERIOR
        
        //Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
        //Therefore consider the joint already finished
        for(joint=0;joint<JOINTS;joint++)
        {
          delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
          if(delta_target < DEAD_BANDWIDTH_SERVO)
            joints_finished |= 1<<joint;	//Set respective bit in 1
          else
            __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], angle2ton_us(target_joint[joint],col_sel[joint]));  //Move the servomotor
        }
        
        //Start first measure of joints before ACTUATION
        conv_cplt = 0;
        memset(joint_angle, 0, sizeof(joint_angle));
        //HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
      
        //Evaluate if servo has to go up or down to get a better measurement
        for(joint=0;joint<JOINTS;joint++)
          up_down_ADC[joint] = (f_joint_angle[joint] < target_joint[joint]);

        state = ACTUATION;
      }
      break;
    case ACTUATION:
      result = State_Machine_Actuation();
      
      if(result == 1)
        state = TX_RASPBERRY;
      else if(result == -1)
        state = TX_RASPBERRY;	//ESTADO STOP DE EMERGENCIA EN EL FUTURO
      break;
  }
}

int8_t State_Machine_Actuation(void)
{
	switch(state_actuation)
	{
		case RESET_ACTUATION:
			if(conv_cplt)
			{
				memset(count_stable_signal, 0, sizeof(count_stable_signal));
				
				for(joint = 0; joint < JOINTS; joint++)
					f_last_joint[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
								
				sample_time = SAMPLE_TIME;
				state_actuation = DELAY_STATE;
				timeout = TIMEOUT;
			}
			break;
		case DELAY_STATE:
			if(timeout == 0)
			  state_actuation = TIMEOUT_STATE;
      else if(sample_time == 0)
			{
				memset(joint_angle, 0, sizeof(joint_angle));
        conv_cplt = 0; // Init adc conversion
				//HAL_ADC_Start_DMA(&hadc1,(uint32_t*)joint_angle_dma,12);
				state_actuation = COMPARE_MEASURE;
			}
			break;
		case COMPARE_MEASURE:
			if(timeout == 0)
        state_actuation = TIMEOUT_STATE;

      else if(conv_cplt)
			{
				for(joint = 0; joint < JOINTS; joint++)
				{
					if((joints_finished & (1<<joint)) == 0)	//Ignore joints that finished
					{
						f_joint_angle[joint] = adc2angle(joint_angle[joint], up_down_ADC[joint],calibration_table[joint],col_sel[joint]);
						
						delta_sample = fabs(f_joint_angle[joint] - f_last_joint[joint]);
						
						if(delta_sample < MAX_DELTA_ANGLE)
						{
							count_stable_signal[joint]++;
							if(count_stable_signal[joint] == UMBRAL_DONE && timeout>0)  //If the joint signal is stable
							{
								count_stable_signal[joint] = 0;

                delta_target = fabs(f_joint_angle[joint] - target_joint[joint]);
								if(delta_target < MAX_DELTA_ANGLE)  //If the stable joint reached target
								{
									joints_finished |= 1<<joint;	//Set respective bit in 1
									
									if(joints_finished == ALL_FINISHED)
									{
										joints_finished = 0;
                    state_actuation = RESET_ACTUATION;
										HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 0);
                    HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 1);
										return 1;
									}
								}
							}
						}
						else
							count_stable_signal[joint] = 0;
					}
				}

				memcpy(f_last_joint, f_joint_angle, sizeof(f_joint_angle));	//Remember joint

				sample_time = SAMPLE_TIME;
				state_actuation = DELAY_STATE;
			}
			
			break;
    case TIMEOUT_STATE:
      state_actuation = RESET_ACTUATION;
      
      HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, 0);
      HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, 1);
				
      //Turn off joints that got stuck
      for(joint=0; joint<JOINTS; joint++)
      {
        if((joints_finished & (1<<joint)) == 0)
          __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], 0);
      }

      joints_finished = 0;
      return -1;
	}
	return 0;
}
*/
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
