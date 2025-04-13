#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},    
};

#define train_count sizeof(train) / sizeof(train[0])

float rand_float()
{
    return (float) rand() / (float) RAND_MAX; // return random float from 0 to 1
}

float cost(float w)
{
    float result = 0.0f;
    for (int i = 0; i < train_count; i++)
    {
        float x = train[i][0];
        float y = train[i][1];
        float d = w*x - y;
        result += d*d;
        // printf("actual: %f, expected: %f \n", y, train[i][1]);
    }
    result /= train_count; // Mean Square Error 
    return result;
}

float dcost(float w) // derivative of cost
{
    float result = 0.0f;
    size_t n = train_count;
    for (int i = 0; i < n; i++)
    {
        float x = train[i][0];
        float y = train[i][1];
        float d = (x*w - y)*x; // gradient descant
        result += d;
    }
    result *= 2;
    result /= n;
    return result;
}

// y = w*x;
int main() 
{
    srand(time(0));
    float w = rand_float() * 10.0f; // weight initialized
    // float b = rand_float() * 5.0f; // bias initialized

    float eps = 1e-3;
    float rate = 1e-1; // learning rate

    for (int i = 0; i < 6; i++)
    {
#if 0
        float c = cost(w);

        // approximation of derivative: finite difference
        float dw = (cost(w + eps) - c) / eps;
#else
        // gradient descent: derivative of the cost function
        float dw = dcost(w);
#endif

        // gradient descant
        w -= rate * dw;
        // moving in opposite direction of dcost to minimize loss

        printf("cost = %f, w = %f \n", cost(w), w);
    }

    printf("---------------------\n");
    printf("w: %f \n", w);

    return 0;
}