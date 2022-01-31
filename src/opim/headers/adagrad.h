#include <math.h>
#include "../../macros.h"


void adagrad_train(int iterations, float lr, float (*lp)(int r, float w[l], float t_x[n][l], float t_y[n]), float w[l], float t_x[n][l], float t_y[n]){


double small = 0.000000001;
float s[l];


    for(int i = 0; i < iterations; i++){

        for(int j = 0; j < l; j++){
            
            float gradient_j = (*lp)(j, w, t_x, t_y); 
            s[j] += pow(gradient_j,2);

            w[j] = w[j] - ((lr)/sqrt(s[j] + small)) * gradient_j;
        }

       // printf("loss: %f \n", MSE(w, t_x, t_y, f));

    }

}