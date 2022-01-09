#include <stdio.h>
#include <math.h>

#define e 2.71

float sigmoid(float v){
    return 1/(1+powf(e,-v));
}

float sigmoid_d(float v){
    return sigmoid(v) * (1-sigmoid(v));
}

float forward(float x, float *w){
  return sigmoid((x)*(*w));
}

float loss_partial(float *w, float t_x[10], float t_y[10]){
    float p = 0;
    for(int i = 0; i < 10; i++){
        float t = t_x[i] * (*w);
        p += 2 * (sigmoid(t) - t_y[i]) *  sigmoid_d(t) *  t_x[i];      

    }

    return p / 10;
}

float MSE(float t_x[10], float t_y[10], float *w){

   float out = 0;
    for(int i = 0; i < 10; i++){
        out += powf((forward(t_x[i], w) - t_y[i]), 2);
    }

    return out / 10;

}




int main(void) {

  float t_x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  float t_y[10] = {.45, .4, .354, .31, .2689, .23, .1978, .1679, .1418, .119};

  float w = 1;
  //-.2

  int iterations = 1000;
  float learning_rate = .01;
    
 //train

float test = 2;

 for(int i = 0; i < iterations; i++){

        w -= learning_rate * loss_partial(&w, t_x, t_y);
        
        printf("loss: %f f(): %f w: %f\n ", MSE(t_x, t_y, &w), forward(test, &w), w);

    }
  

  
  //printf("(2): %f", forward(test, &w));

  
  return 0;
}