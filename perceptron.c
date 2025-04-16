#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float or_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float train_count = sizeof(or_train) / sizeof(or_train[0]);

float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float cost(float w1, float w2, float b)
{
    float result = 0.0f;
    size_t n = train_count;
    for (size_t i = 0; i < n; i++)
    {
        float x1 = or_train[i][0];
        float x2 = or_train[i][1];
        float y = or_train[i][2];

        float d = sigmoidf((w1*x1) + (w2*x2) + b) - y;
        result += d*d;
    }
    result /= n;
    return result;
}

void dcost(float eps, float w1, float w2, float b, float *dw1, float *dw2, float* db)
{
    float c = cost(w1, w2, b);

    *dw1 = (cost(w1 + eps, w2, b) - c)/eps;
    *dw2 = (cost(w1, w2 + eps, b) - c)/eps;
    *db = (cost(w1, w2, b + eps) - c)/eps;
}

void gcost(float w1, float w2, float b, float *dw1, float *dw2, float* db)
{
    *dw1 = 0.0f;
    *dw2 = 0.0f;
    *db = 0.0f;

    size_t n = train_count;
    for (size_t i = 0; i < n; i++)
    {
        float x1 = or_train[i][0];
        float x2 = or_train[i][1];
        float y = or_train[i][2];
        float ai = sigmoidf(w1*x1 + w2*x2 + b);
        float di = (ai - y) * ai * (1 - ai); // derivative of cost 

        *dw1 += di* x1;
        *dw2 += di * x2;
        *db += di;
    }

    *dw1 *= (float) 2/n;
    *dw2 *= (float) 2/n;
    *db *= (float) 2/n;
}

int main(void)
{
    srand(time(0));

    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float rate = 1e-1;



    printf("cost = %f \n", cost(w1, w2, b));
    for (size_t i = 0; i < 50*100; i++)
    {
        float dw1, dw2, db;

#if 0
        float eps = 1e-1;
        // finite difference
        dcost(eps, w1, w2, b, &dw1, &dw2, &db);

#else
        // gradient descent
        gcost(w1, w2, b, &dw1, &dw2, &db);
#endif
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
    }
    printf("cost = %f \n", cost(w1, w2, b));

    printf("--------------------\n");
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("%d | %d = %f \n", i, j, sigmoidf(i*w1 + j*w2 + b));
        }
    }
}