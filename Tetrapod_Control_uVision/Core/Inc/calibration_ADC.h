#include "stdint.h"

#ifndef ADC_DATA_H_
#define ADC_DATA_H_

const uint16_t ADC_VALUES[2][21] =
{
	//ADC accumulated values for 10 measurements of servo going down
	{
		4051,
		5691,
		7391,
		9028,
		10787,
		12398,
		14109,
		15734,
		17496,
		19084,
		20780,
		22362,
		23911,
		25524,
		27051,
		28678,
		30290,
		31843,
		33341,
		35009
	},
	//servo going up
	{
		4075,
		5489,
		7198,
		8902,
		10575,
		12234,
		13995,
		15601,
		17266,
		18920,
		20562,
		22208,
		23783,
		25355,
		26916,
		28503,
		30084,
		31618,
		33219,
		34815,
		36182
	}
};

//Corresponding theoretical angles in degrees
const float ANGLES[21] = 
{
	0,
	9,
	18,
	27,
	36,
	45,
	54,
	63,
	72,
	81,
	90,
	99,
	108,
	117,
	126,
	135,
	144,
	153,
	162,
	171,
	180
};
#endif /* INC_TRANSFERENCIAADC_CALIBRADO_H_ */