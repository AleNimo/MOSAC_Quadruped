#include <BasicLinearAlgebra.h>
#include <ElementStorage.h>
#include <FIR.h>
#include <Wire.h>
//#include <curveFitting.h>


/*****************************************************************************/
//  HighLevelExample.ino
//  Hardware:      Grove - 6-Axis Accelerometer&Gyroscope
//	Arduino IDE:   Arduino-1.65
//	Author:	       Lambor
//	Date: 	       Oct,2015
//	Version:       v1.0
//
//  Modified by:
//  Data:
//  Description:
//
//	by www.seeedstudio.com
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
//
/*******************************************************************************/

#include "LSM6DS3.h"
#include <math.h>

#define G 9.80665f


using namespace BLA;

Matrix<3,1,double> EulerAccel(double ax,double ay,double az);
Matrix<3,1,double> EP2Euler321(Matrix<4,1,double> q);
Matrix<4,1,double>Euler3212EP(Matrix<3,1,double> e);

Matrix<4,4,double>I = { 1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1};
Matrix<4,4,double>H = I;
Matrix<4,4,double>K;
Matrix<4,4,double>A;
Matrix<4,4,double>Q = I * pow(1.2* PI/180,2);
Matrix<4,4,double>R = I * pow(0.002 * G,2);
//Matrix<4,4,double>Q = I * pow(0.028 * PI/180,2);
//Matrix<4,4,double>R = I * pow(0.0007 * G,2);
Matrix<4,4,double>P = I * 0.1;
Matrix<4,1,double>X = {1,
                       0,
                       0,
                       0};

Matrix<4,4,double>aux;

Matrix<4,1,double>Z;
Matrix<3,1,double>angles;

double ax_noisy;
double ay_noisy;
double az_noisy;

double wx_noisy;
double wy_noisy;
double wz_noisy;


double ax;
double ay;
double az;

double wx;
double wy;
double wz;

double yaw = 0;
double pitch = 0;
double roll = 0;

double dt = 0.001;


//Calibration
double gx_cal = 0;
double gy_cal = 0;
double gz_cal = 0;
/*
double ax_cal[900];
double ay_cal[900];
double az_cal[900];

double g_cal_x[900];
double g_cal_y[900];
double g_cal_z[900];
*/

//LPF
FIR<double,10>fir_ax;
FIR<double,10>fir_ay;
FIR<double,10>fir_az;
double coef_a[10] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};

FIR<double,3>fir_wx;
FIR<double,3>fir_wy;
FIR<double,3>fir_wz;
double coef_w[3] = {1.,1.,1.};

float imu_data[4] = {0};

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

double Time  = 0;
double Time_prev  = 0;

void setup() {
    // put your setup code here, to run once:
    //Serial.begin(9600);
    //while (!Serial);
    //Call .begin() to configure the IMUs
    if (myIMU.begin() != 0) {
        //Serial.println("Device error");
    } else {
        //Serial.println("Device OK!");
    }
    //Gyro Calibration
    for(unsigned int i = 0; i<5000;i++)
    {
      gx_cal += myIMU.readFloatGyroX();
      gy_cal += myIMU.readFloatGyroY();
      gz_cal += myIMU.readFloatGyroZ();
      delay(1);
    }

    gx_cal /=5000;
    gy_cal /=5000;
    gz_cal /=5000;
/*
    //Accelerometer Calibration
 

    //IMU Poiting Against gravity
    Serial.println("Point IMU against gravity");
    for(unsigned int i = 0; i<300;i++)
    {
      ax_cal[i] =  myIMU.readFloatAccelX();
      ay_cal[i] = myIMU.readFloatAccelY();
      az_cal[i] = myIMU.readFloatAccelZ();
      g_cal_x[i] =  0.0f;
      g_cal_y[i] =  0.0f;
      g_cal_z[i] =  1.0f;
      delay(10);
    }
    Serial.println("Point IMU towards gravity");
    delay(3000);
    //IMU Pointing Towards gravity    
    for(unsigned int i = 300; i<600;i++)
    {
      az_cal[i] = myIMU.readFloatAccelZ();
      g_cal_z[i] =  -1.0f;
      delay(10);
    }

    //IMU Poiting Perpendicular gravity (x poiting down)
    Serial.println("Point IMU perpendicular gravity(x poiting down)");
    delay(3000);    
    for(unsigned int i = 600; i<900;i++)
    {
      ax_cal[i] =  myIMU.readFloatAccelX();
      az_cal[i] = myIMU.readFloatAccelZ();

      g_cal_x[i] =  -1.0f;
      g_cal_z[i] =  0.0f;
      
      delay(10);
    }

    //IMU Poiting Perpendicular gravity (x poiting up)
    Serial.println("Point IMU perpendicular gravity(x poiting up)");
    delay(3000);    
    for(unsigned int i = 300; i<600;i++)
    {
      ax_cal[i] =  myIMU.readFloatAccelX();
      g_cal_x[i] =  1.0f;     
      delay(10);
    }
    //IMU Poiting Perpendicular gravity (y poiting down)
    Serial.println("Point IMU perpendicular gravity(y poiting down)");
    delay(3000);    
    for(unsigned int i = 600; i<900;i++)
    {
      ay_cal[i] =  myIMU.readFloatAccelY();
      g_cal_y[i] =  -1.0f;     
      delay(10);
    }
    //IMU Poiting Perpendicular gravity (y poiting up)
    Serial.println("Point IMU perpendicular gravity (y poiting up)");
    delay(3000);
    for(unsigned int i = 300; i<600;i++)
    {
      ay_cal[i] =  myIMU.readFloatAccelY();
      g_cal_y[i] =  1.0f;     
      delay(10);
    }

    double coeffs_x[2];
    double coeffs_y[2];
    double coeffs_z[2];
    fitCurve(1, 900, (double*)g_cal_x, (double*)ax_cal, 2, coeffs_x);
    fitCurve(1, 900, (double*)g_cal_y, (double*)ay_cal, 2, coeffs_y);
    fitCurve(1, 900, (double*)g_cal_z, (double*)az_cal, 2, coeffs_z);
    Serial.println("LinReg x=");
    Serial.print(coeffs_x[0]);
    Serial.print(",");
    Serial.print(coeffs_x[1]);
    Serial.print("\n");
    Serial.println("LinReg y=");
    Serial.print(coeffs_y[0]);
    Serial.print(",");
    Serial.print(coeffs_y[1]);
    Serial.print("\n");
    Serial.println("LinReg z=");
    Serial.print(coeffs_z[0]);
    Serial.print(",");
    Serial.print(coeffs_z[1]);
    Serial.print("\n");
   
*/

  fir_ax.setFilterCoeffs(coef_a);
  fir_ay.setFilterCoeffs(coef_a);
  fir_az.setFilterCoeffs(coef_a);

  fir_wx.setFilterCoeffs(coef_w);
  fir_wy.setFilterCoeffs(coef_w);
  fir_wz.setFilterCoeffs(coef_w);

  Wire.begin(0x1);            //I2C communication
  Wire.onRequest(Send_IMU);   //Function to call when master starts communication


}

void loop() {

     
  ax_noisy = -myIMU.readFloatAccelX()*G;
  ay_noisy = myIMU.readFloatAccelY()*G;
  az_noisy = -myIMU.readFloatAccelZ()*G;

  wx_noisy = -(myIMU.readFloatGyroX()-gx_cal)*PI/180;
  wy_noisy = (myIMU.readFloatGyroY()-gy_cal)*PI/180;
  wz_noisy = -(myIMU.readFloatGyroZ()-gz_cal)*PI/180;

  Time = micros()/1000000.0f;
  dt = Time - Time_prev;
  Time_prev = Time;
  //Serial.println(dt*1000000.0f);


  //IIR Filter First Order
  ax = fir_ax.processReading(ax_noisy);
  ay = fir_ay.processReading(ay_noisy);
  az = fir_az.processReading(az_noisy);

  wx = fir_wx.processReading(wx_noisy);
  wy = fir_wy.processReading(wy_noisy);
  wz = fir_wz.processReading(wz_noisy);
  
 
  
  A = { 0,   -wx,  -wy,  -wz,
        wx,  0,    wz,  -wy,
        wy,  -wz,   0,    wx,
        wz,   wy,  -wx,   0  };

  A = I + A*dt*0.5;

  X = A*X;
  P = A*P*(~A) + Q;

  angles = EP2Euler321(X);
  yaw = angles(0,0);

  Invert(H*P*(~H) + R,aux);
  K = P*(~H)*aux;

  angles = EulerAccel(ax, ay,az);
  angles(0,0) = yaw;

  Z = Euler3212EP(angles);

  X = X + K*(Z - H*X);   
  P = P - K*H*P;

  angles = EP2Euler321(X);

  yaw = angles(0,0);
  pitch = angles(1,0);
  roll = angles(2,0);

/*Yaw, Pitch & Roll prints*/
/*
  Serial.print(yaw*180/PI);
  Serial.print(",");
  Serial.print(pitch*180/PI);
  Serial.print(",");
  Serial.print(roll*180/PI);
  Serial.print(",");
  Serial.print("\n");

*/
  /*Noisy, filtered & corrected acc + Gyro + Velocity */
  /*
  Serial.print(ax_noisy);
  
  Serial.print(",");
  
  Serial.print(ay_noisy);
  Serial.print(",");
  Serial.print(az_noisy);
  Serial.print(",");
  Serial.print(wx_noisy);
  Serial.print(",");
  Serial.print(wy_noisy);
  Serial.print(",");
  Serial.print(wz_noisy);
  Serial.print(",");
  
  Serial.print(ax);
  Serial.print(",");
  
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print(",");
  Serial.print(wx);
  Serial.print(",");
  Serial.print(wy);
  Serial.print(",");
  Serial.print(wz);
  Serial.print(",");  
  */

/*
  Serial.print(dt);
  Serial.print("\n");
  */
   

  delay(1);
  
}

Matrix<3,1,double> EulerAccel(double ax, double ay, double az)
{

  float g = sqrt(pow(ax,2) + pow(ay,2) + pow(az,2));

  double theta = asin(ax/g);
  double phi   = atan2l(-ay,-az);
/*
  if(theta >= PI/2)
      theta -= PI;
  else if (theta <=- PI/2)
      theta += PI;  */
/*
  Serial.print("ax ");
  Serial.println(ax);
  Serial.print("az ");
  Serial.println(az);
  Serial.print("arctan2 ");
  Serial.println(theta);
  */
/*
  if(phi>=0)
    phi-=PI;
  else
    phi+=PI;  
*/
/*
  Serial.print("ay ");
  Serial.println(ay);
  Serial.print("az ");
  Serial.println(az);
  //Serial.println(phi*180/PI);
  */
  Matrix<3,1,double> angles = {0,theta,phi};

  return angles;
}

Matrix<3,1,double> EP2Euler321(Matrix<4,1,double> q)
{

  /*E = EP2Euler321(Q) translates the Euler parameter vector
  Q into the corresponding (3-2-1) Euler angle set.
  */

  double q0 = q(0,0);
  double q1 = q(1,0);
  double q2 = q(2,0);
  double q3 = q(3,0);

  Matrix<3,1,double> e;


  e(0,0) = atan2(2*(q1*q2+q0*q3),q0*q0+q1*q1-q2*q2-q3*q3);
  e(1,0) = asin(-2*(q1*q3-q0*q2));
  e(2,0)= atan2(2*(q2*q3+q0*q1),q0*q0-q1*q1-q2*q2+q3*q3);

  return e;
}


Matrix<4,1,double>Euler3212EP(Matrix<3,1,double> e)
{
  /*
	Q = Euler3212EP(E) translates the 321 Euler angle
	vector E into the Euler parameter vector Q.
  */
  double c1 = cos(e(0,0)/2);
  double s1 = sin(e(0,0)/2);
  double c2 = cos(e(1,0)/2);
  double s2 = sin(e(1,0)/2);
  double c3 = cos(e(2,0)/2);
  double s3 = sin(e(2,0)/2);

  Matrix<4,1,double> q;

  q(0,0) = c1*c2*c3+s1*s2*s3;
  q(1,0) = c1*c2*s3-s1*s2*c3;
  q(2,0) = c1*s2*c3+s1*c2*s3;
  q(3,0) = s1*c2*c3-c1*s2*s3;

  return q;
}

void Send_IMU(void)
{
  
  imu_data[0] = pitch;
  imu_data[1] = roll;
  imu_data[2] = wy;
  imu_data[3] = wx;

/*
  Serial.print(imu_data[0],10);
  Serial.print(",");
  Serial.print(imu_data[1],10);
  Serial.print(",");
  Serial.print(imu_data[2],10);
  Serial.print(",");
  Serial.print(imu_data[3],10);
  Serial.print("\n");
 */

/*
  byte imu_data[16] = {0};
  for(char i = 0 ; i<16;i++) imu_data[i] = i;
  */
  Wire.write((byte*)imu_data,sizeof(imu_data));
}
