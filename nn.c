#include <stdio.h>
#include <math.h>

#define e 2.71
#define n 3


float sigmoid(float v){
    return 1/(1+pow(e,-v));
}

float sigmoid_d(float v){
    return sigmoid(v) * (1-sigmoid(v));
}

float dot(float w[3], float x[3]){
    float out = 0;
    for(int i = 0; i<n; i++){
        out += w[i] * x[i];
    }
   return out;
}

float forward(float w[3], float x[3]){
    return sigmoid(dot(w, x));
}

float loss_partial(int r, float w[3], float t_x[3][n], float t_y[n]){
    float p = 0;
    for(int i = 0; i < 3; i++){
        float t = forward(w, t_x[i]);
        p += 2 * (sigmoid(t) - t_y[i]) *  sigmoid_d(t) *  t_x[i][r];      

    }

    return p / 3;
}



int main()
{
    float w[3] = {.3, -.9, .1};
   
    float t_x[3][n] = {{1, 0, 0},{1, 1, 0},{0, 1, 0}};
    float t_y[n] = {1, 1, 0};
   
   
    //forward
   printf("%f", forward(w, t_x[0]));

   //train
    int iterations = 100;
    float learning_rate = .01;
    for(int i = 0; i < iterations; i++){

        for(int j = 0; j < 3; j++){
            w[j] = w[j] - learning_rate * loss_partial(j, w, t_x, t_y);
        }

    }


    return 0;
}
