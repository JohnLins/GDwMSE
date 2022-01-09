#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>



#define e 2.71
#define n 3
#define l 3


float sigmoid(float v){
    return 1/(1+powf(e,-v));
}

float sigmoid_d(float v){
    return sigmoid(v) * (1-sigmoid(v));
}

float dot(float w[l], float x[l]){
    float out = 0;
    for(int i = 0; i<n; i++){
        out += w[i] * x[i];
    }
   return out;
}

float forward(float w[l], float x[l]){
    return sigmoid(dot(w, x));
}

float loss_partial(int r, float w[l], float t_x[n][l], float t_y[n], float (*s)(), float (*f)()){
    float p = 0;
    for(int i = 0; i < n; i++){
        float t = dot(w, t_x[i]);
        p += 2 * (sigmoid(t) - t_y[i]) *  sigmoid_d(t) *  t_x[i][r];      

    }

    return p / n;
}

float MSE(float w[l], float t_x[n][l], float t_y[n], float (*f)()){

    float out = 0;
    for(int i = 0; i < n; i++){
        out += powf(((*f)(w, t_x[i]) - t_y[i]), 2);
    }

    return out / n;

}

void train(int *iterations, float *lr, float (*lp)(), float (*f)(), float w[l], float t_x[n][l], float t_y[n]){

    for(int i = 0; i < *iterations; i++){

        for(int j = 0; j < 3; j++){
            w[j] = w[j] - (*lr) * (*lp)(j, w, t_x, t_y, f);
        }

        printf("loss: %f \n", MSE(w, t_x, t_y, f));

    }

}


int main()
{

    srand(time(0));

    float w[l] = {.3, -.9, .1};

    for(int i = 0; i < l; i++){
        w[i] = 1.0 * rand() / (RAND_MAX / 2) - 1;
    }
   
    float t_x[n][l] = {{1, 0, 0},{1, 1, 0},{0, 1, 0}};
    float t_y[n] = {1, 1, 0};
   
    int iterations = 500;
    float learning_rate = .01;
    
    train(&iterations, &learning_rate, &loss_partial, &forward, w, t_x, t_y);


    float test1[3] = {1, 0, 0};
    float test2[3] = {0, 1, 0};
    float test3[3] = {1, 0, 1};

    printf("Result (1): %f \n", forward(w, test1));
    printf("Result (0): %f \n", forward(w, test2));
    printf("Result (1): %f \n", forward(w, test3));

    return 0;
}
