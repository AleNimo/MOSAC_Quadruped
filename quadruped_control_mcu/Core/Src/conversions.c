#include "conversions.h"
#include "calibration_ADC.h"

#include "main.h"

const float32_t* calibration_table[12] = {&ADC_VALUES_SERVO_0[0][0], &ADC_VALUES_SERVO_1[0][0], &ADC_VALUES_SERVO_2[0][0], &ADC_VALUES_SERVO_3[0][0], &ADC_VALUES_SERVO_4[0][0], &ADC_VALUES_SERVO_5[0][0], &ADC_VALUES_SERVO_6[0][0], &ADC_VALUES_SERVO_7[0][0], &ADC_VALUES_SERVO_8[0][0], &ADC_VALUES_SERVO_9[0][0], &ADC_VALUES_SERVO_10[0][0], &ADC_VALUES_SERVO_11[0][0]};

// Convert ADC_value to angle in degrees using linear interpolation and calibration tables
float32_t adc2angle(uint16_t adc_value, uint8_t up_down, uint8_t joint)
{
  uint16_t index = 0;

  const float32_t *ADC_VALUES = calibration_table[joint];

  while (adc_value > ADC_VALUES[up_down * MED_CALIB + index] / MEAS_PER_ANG) // Search for the adc_value in the obtained calibration table
  {
    index++;
    if (index > MED_CALIB - 1)
      return MAX_TABLE_ANGLE; //adc_value greater than maximum value of table
  }

  if (adc_value == ADC_VALUES[up_down * MED_CALIB + index] / MEAS_PER_ANG) // If the exact value is in the table return the corresponding angle (pretty unlikely)
    return ANGLES[index];

  else if (index == 0)
    return 0; // adc_value lesser than minimum value of table

  else // If the value value is lower than theSi es menor al codigo de la tabla, realizo una interpolacion lineal
    return (ANGLES[index - 1] * (ADC_VALUES[up_down * MED_CALIB + index] / MEAS_PER_ANG - adc_value) + ANGLES[index] * (adc_value - ADC_VALUES[up_down * MED_CALIB + index - 1] / MEAS_PER_ANG)) / (ADC_VALUES[up_down * MED_CALIB + index] / MEAS_PER_ANG - ADC_VALUES[up_down * MED_CALIB + index - 1] / MEAS_PER_ANG);
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

