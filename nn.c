#include <stdio.h>
#include <math.h>

#define e 2.71
#define n 3


float sigmoid(float v){
    return 1/(1+pow(e,-v));
}

float sigmoidD(float v){
    return sigmoid(v) * (1-sigmoid(v));
}

float dot(float w[3], float x[3]){
    for(int i = 0; i<n; i++){
        float g = w[i] * x[i];
    }
   return g;
}

int main()
{
    float w[3] = {.3, -.9, .1};
   
    float t_x[3][n] = {{},{},{}};
    float t_y[n] = {};
   
   
    //forward
   sigmoid(dot(w, t_x));



    return 0;
}
