/*
avgada 
Averages the learning rate for all weights (more efficient and adds noise)

WORKS
*/



#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>



#include "../../macros.h"


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

float loss_partial(int r, float w[l], float t_x[n][l], float t_y[n], float (*s)()){
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

void train(int iterations, float lr, float (*lp)(), float (*f)(), float w[l], float t_x[n][l], float t_y[n]){



double small = 0.000000001;
float decay_rate = .9;
float cashe = 0;


    for(int i = 0; i < iterations; i++){

        for(int j = 0; j < 3; j++){
           
           
           
            float gradient_j = (*lp)(j, w, t_x, t_y);
            cashe += pow(gradient_j,2);
           
            //s[j] = decay_rate * s[j] + (1-decay_rate) * pow(gradient_j,2);

                //3 = len(w)
            w[j] = w[j] - ((lr)/sqrt((  cashe/3  ) + small)) * gradient_j;

            
        }

        printf("loss: %f , l: %f\n", MSE(w, t_x, t_y, f), ((lr)/sqrt((  cashe/3  ) + small)));

    }

}





int main()
{

    srand(time(0));

    float w[l];

    for(int i = 0; i < l; i++){
        w[i] = 1.0 * rand() / (RAND_MAX / 2) - 1;
    }
   
    float t_x[n][l] = {{1, 0.1, 0.1, 1, 0.1, 1, 1, 0.1}, //happy
                        {1, 0.1, 0.1, 1, 1, 1, 1, 1},
                        {1, 0.1, 0.1, 1, 0.1, 1, 1, 1},
                        {1, 0.1, 0.1, 1, 1, 1, 1, 0.1},


                        {0.1, 1, 1, 0.1, 1, 0.1, 0.1, 1}, //sad
                        {1, 1, 1, 1, 1, 0.1, 0.1, 1},
                        {0.1, 1, 1, 1, 1, 0.1, 0.1, 1},
                        {1, 1, 1, 0.1, 1, 0.1, 0.1, 1}};



    float t_y[n] = {1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1};
   
    int iterations = 1000;
   float learning_rate = .01;
    
    train(iterations, learning_rate, &loss_partial, &forward, w, t_x, t_y);


    float test1[l] = {1, 0.1, 0.1, 1, 0.1, 1, 1, 0.1};
    float test2[l] = {0.1, 1, 1, 1, 1, 0.1, 0.1, 1};

    printf("Result (1): %f \n", forward(w, test1));
    printf("Result (0): %f \n", forward(w, test2));


    return 0;
}