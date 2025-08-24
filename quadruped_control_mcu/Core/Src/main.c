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
#include "adc.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "arm_math.h"
#include <math.h>
#include <string.h>

#include "conversions.h"
#include "state_machines.h"
#include "signal_processing.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// For tx floats under UART (Serial plot)
typedef union float2byte
{
  float angle;
  uint8_t angle_bytes[4];
} float2byte;

float2byte uart_conv_1;
float2byte uart_conv_2;
float2byte uart_conv_3;
float2byte uart_conv_4;

//Median filter
volatile float32_t filter_in_arm_angle[12];
volatile float32_t median_filteredValue[JOINTS];
volatile spMedianFilter median_filter[JOINTS];

//SPI
volatile uint8_t spi_rx_cplt = 0;

// Buffer rx SPI
volatile float32_t target_joint[12] ={MID_POINT_BFR, MID_POINT_BBR, MID_POINT_BBL, MID_POINT_TFR, MID_POINT_FFR, MID_POINT_BFL, MID_POINT_FFL, MID_POINT_TFL, MID_POINT_FBL, MID_POINT_TBL, MID_POINT_TBR, MID_POINT_FBR};

// PID Variables
volatile float32_t error = 0;
volatile float32_t previous_error[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile float32_t control_signal = 0;
volatile float32_t error_acum[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
volatile float32_t error_dif = 0;
volatile uint16_t pwm_pid_out[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// PID constants
const float32_t kp[JOINTS] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
const float32_t kd[JOINTS] = {0,0,0,0,0,0,0,0,0,0,0,0};
const float32_t ki[JOINTS] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1};

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



volatile uint16_t raw_angle_ADC[2][12] = {}; // ADC measurement with DMA

volatile uint8_t current_dma_buff = 0;

// Ticks of timer
volatile uint8_t pid_sample = 0;
volatile uint8_t send_uart = 0;
volatile uint16_t timeout = 0;
volatile uint16_t time_debounce = 0;  //To debounce user button

// Init program at reset with user button
volatile uint8_t init_nucleo = 0;


/////////////////////////////

volatile float32_t joint_angle[JOINTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

volatile uint8_t uart_tx_buffer[2 + 12 * 4 + 12 * 4 + 12 * 4 + 12 * 4];

volatile int8_t step_complete = 1;           // returned by State_Machine_Actuation

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

void move_servos(uint8_t joint, uint16_t ton);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  uint8_t joint;
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

  for (joint = 0; joint < JOINTS; joint++)
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

  while(init_nucleo == 0) HAL_Delay(10);

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
    //memcpy(dummy1, &raw_angle_ADC[buffer_to_copy][0], sizeof(dummy1));
    
    //Serial Plot
    // if (send_uart)
    // {
    //   for (int i = 0; i < JOINTS; i++)
    //   {
		// 		 uart_conv_1.angle = target_joint[i];
    //     //uart_conv_2.angle = adc2angle(median_filteredValue[i], up_down_vector[i], i);
		// 		 //uart_conv_3.angle = adc2angle(filter_in_arm_angle[i], up_down_vector[i], i);
    //     uart_conv_2.angle = ton_us2angle(pwm_pid_out[i]);
		// 		 uart_conv_3.angle = error_acum[i];
    //     uart_conv_4.angle = joint_angle[i];

    //     uart_tx_buffer[16 * i + 2] = uart_conv_1.angle_bytes[3];
    //     uart_tx_buffer[16 * i + 3] = uart_conv_1.angle_bytes[2];
    //     uart_tx_buffer[16 * i + 4] = uart_conv_1.angle_bytes[1];
    //     uart_tx_buffer[16 * i + 5] = uart_conv_1.angle_bytes[0];

    //     uart_tx_buffer[16 * i + 6] = uart_conv_2.angle_bytes[3];
    //     uart_tx_buffer[16 * i + 7] = uart_conv_2.angle_bytes[2];
    //     uart_tx_buffer[16 * i + 8] = uart_conv_2.angle_bytes[1];
    //     uart_tx_buffer[16 * i + 9] = uart_conv_2.angle_bytes[0];
				
		// 		 uart_tx_buffer[16 * i + 10] = uart_conv_3.angle_bytes[3];
    //     uart_tx_buffer[16 * i + 11] = uart_conv_3.angle_bytes[2];
    //     uart_tx_buffer[16 * i + 12] = uart_conv_3.angle_bytes[1];
    //     uart_tx_buffer[16 * i + 13] = uart_conv_3.angle_bytes[0];
				
		// 		 uart_tx_buffer[16 * i + 14] = uart_conv_4.angle_bytes[3];
    //     uart_tx_buffer[16 * i + 15] = uart_conv_4.angle_bytes[2];
    //     uart_tx_buffer[16 * i + 16] = uart_conv_4.angle_bytes[1];
    //     uart_tx_buffer[16 * i + 17] = uart_conv_4.angle_bytes[0];
    //   }
		//   HAL_UART_Transmit(&huart3, (const uint8_t*) uart_tx_buffer, 16 * JOINTS + 2, HAL_MAX_DELAY);

    //   send_uart = 0;
    // }
		
    // HAL_Delay(1000); // Adjust delay as necessary

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

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
  current_dma_buff = 0;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
  current_dma_buff = 1;
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

static inline void pid(uint8_t joint)
{
  if (step_complete == 0)
  {
    error = target_joint[joint] - joint_angle[joint];
    error_acum[joint] += error * TIM9_TICK;
    error_dif = (error - previous_error[joint]) / TIM9_TICK; //(divido tick de tim9)

    previous_error[joint] = error;
    
    control_signal = kp[joint] * error + ki[joint] * error_acum[joint] + kd[joint] * error_dif; //Delta_angle
    up_down_vector[joint] = control_signal > 0;

    pwm_pid_out[joint] = pwm_pid_out[joint] + __round_int(control_signal);
    
    move_servos(joint, pwm_pid_out[joint]);
  }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *timer)
{
  uint8_t joint;
  uint8_t freezed_dma_index;

  if (timer == &htim5)
  {
    if (timeout > 0)
      timeout--;

    if (time_debounce > 0)
      time_debounce--;

    // Copy the contents from the ping pong dma buffer to a stable buffer
    // - current_dma_buff selects the half of the buffer most recently completed (the other is being used)
    // - first keep the current_dma_buff index freezed until the end of the copy:
    freezed_dma_index = current_dma_buff;

    for (joint = 0; joint < 12; joint++)
      filter_in_arm_angle[joint] = raw_angle_ADC[freezed_dma_index][joint];
    
    // With the contents copied, apply the median filter
    for (joint = 0; joint < 12; joint++)
			median_filteredValue[joint] = spMedianFilterInsert(median_filter[joint], filter_in_arm_angle[joint]);
  }

  if (timer == &htim9)
  {
		// Move servos
		for (joint = 0; joint < 12; joint++)
		{

      // Convert the adc values to angles with the calibration tables
			joint_angle[joint] = adc2angle(median_filteredValue[joint], up_down_vector[joint], joint);
			
      // Apply the PID controller, and with the final pwm value move the servomotors
			pid(joint);
		
		}
    pid_sample = 1;
    send_uart = 1;
	}
}

void move_servos(uint8_t joint, uint16_t ton)
{
  //If ton is below min range, move to min range
  if(ton_us2angle(ton) < (mid_point_joints[joint] + joint_range[joint][0]))
    ton = angle2ton_us(mid_point_joints[joint] + joint_range[joint][0]);

  //If ton is above max range, move to max range
  else if(ton_us2angle(ton) > (mid_point_joints[joint] + joint_range[joint][1]))
    ton = angle2ton_us(mid_point_joints[joint] + joint_range[joint][1]);

  __HAL_TIM_SET_COMPARE(htim[joint / 4], channel[joint % 4], ton);
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
