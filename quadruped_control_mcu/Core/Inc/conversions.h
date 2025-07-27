#ifndef __CONV_H__
#define __CONV_H__

#include "arm_math.h"

float32_t adc2angle(uint16_t adc_value, uint8_t up_down, uint8_t joint);
uint16_t angle2ton_us(float);
float32_t ton_us2angle(uint16_t ton_us);

#endif