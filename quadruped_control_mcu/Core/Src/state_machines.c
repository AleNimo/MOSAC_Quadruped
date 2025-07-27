#include "state_machines.h"
#include "arm_math.h"

#include "spi.h"

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

extern volatile float32_t joint_angle[JOINTS];
extern volatile float32_t target_joint[12];
extern const float32_t mid_point_joints[12];

extern volatile uint8_t spi_rx_cplt;

extern volatile int8_t step_complete;

extern volatile uint16_t timeout;
extern volatile uint8_t pid_sample;

extern volatile uint8_t up_down_vector[JOINTS];

extern volatile float32_t error_acum[JOINTS];
extern volatile float32_t previous_error[JOINTS];

extern void move_servos(uint8_t joint, uint16_t ton);
extern uint16_t angle2ton_us(float angle_value);


void State_Machine_Control(void)
{
  static uint8_t state = TX_RASPBERRY;

  uint8_t joint;

  float32_t spi_transmit_rpi[JOINTS];

  switch (state)
  {
    case TX_RASPBERRY:
		  // Request master to transmit target step rotation and joint angles				
      for(joint = 0; joint<JOINTS ; joint++)
        spi_transmit_rpi[joint] = joint_angle[joint] - mid_point_joints[joint];
      
      HAL_SPI_Transmit_DMA(&hspi3, (uint8_t*)spi_transmit_rpi, 12*2);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_RESET);
      HAL_Delay(1);
      HAL_GPIO_WritePin(SPI_Ready_GPIO_Port, SPI_Ready_Pin, GPIO_PIN_SET);

      state = RX_RASPBERRY;
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
  static uint8_t state_actuation = RESET_ACTUATION;

  static uint16_t joints_finished = ALL_FINISHED; // Each bit is a flag for a joint: 1-Finished, 0-Unfinished

  uint8_t joint;
  
  float32_t delta_target;

  switch (state_actuation)
  {
    case RESET_ACTUATION:

      // Check if the new target is too close to the actual position, in which case the servo wouldn't move because of the dead bandwidth
      // Therefore consider the joint already finished
      for (joint = 0; joint < JOINTS; joint++)
      {
        delta_target = fabs(joint_angle[joint] - target_joint[joint]);
        
        if (delta_target >= DEAD_BANDWIDTH_SERVO)
        {
          // Evaluate if servo has to go up or down to get a better measurement
          up_down_vector[joint] = (joint_angle[joint] < target_joint[joint]);
          
          joints_finished &= ~(1 << joint); // Set respective bit in 0

          //*Solo si no se usa PID
          move_servos(joint, angle2ton_us(target_joint[joint]));
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
            delta_target = fabs(joint_angle[joint] - target_joint[joint]);

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