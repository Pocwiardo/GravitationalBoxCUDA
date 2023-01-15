
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <fstream>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

const double AU = 1.496e+11;
const double G = 6.67428e-11;
const double SCALE = 250 / AU;  // 1AU = 100 pixels
const double TIMESTEP = 3600 * 24; // 1 day

class Planet
{
    
public:
    float mass;
    float x;
    float y;
    float vx;
    float vy;
    Planet() : mass(1E+23), vx(AU), vy(AU) {}
    __host__ __device__ Planet(float mass, float x, float y)
    {
        this->mass = mass;
        this->x = x;
        this->y = y;
        this->vx = 0;
        this->vy = 0;
    }
    __host__ __device__ void updatePosition(float dt)
    {
        // Implementacja ruchu planet
    }
    __host__ __device__ float getMass()
    {
        return mass;
    }
    __host__ __device__ float getX()
    {
        return x;
    }
    __host__ __device__ float getY()
    {
        return x;
    }
private:
};

__global__ void updatePositions(Planet* planets, int n, int t, double* x_coordinates, double* y_coordinates)
{
    // Indeks wątku
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pamięć dzielona dla przechowywania pozycji planety
    __shared__ double pos[2];

    // Jeśli indeks wątku jest mniejszy niż liczba planet
    if (i < n)
    {
        // Pobranie pozycji planety do pamięci dzielonej
        //pos[0] = planets[i].x; //planets[i].x is impostor, coś z pamięcią nie teges
        //pos[1] = planets[i].y;

        // Synchronizacja wątków w bloku
        __syncthreads();

        // Obliczanie nowych pozycji dla planety
        double Fx = 0;
        double Fy = 0;
        /*for (int j = 0; j < n; j++)
        {
            // Obliczanie siły działającej na planetę
            double r = distance(pos[0], pos[1], planets[j].x, planets[j].y);
            double F = G * planets[i].mass * planets[j].mass / (r * r);
            double theta = atan2(planets[j].y - pos[1], planets[j].x - pos[0]);
            Fx += F * cos(theta);
            Fy += F * sin(theta);
        }*/

        // Aktualizacja pozycji planety
        //planets[i].x += Fx / planets[i].mass * dt;
        //planets[i].y += Fy / planets[i].mass * dt;
        //planets[i].x += 1e+12;
        //planets[i].y += 1e+11;

        // Zapis koordynatów do tablic
        x_coordinates[t*n + i] = 10;
        y_coordinates[t*n + i] = 10;
    }
}



int main()
{
    const int n = 1000;
    const int units = 10; //jednostki czasowe
    const double MIN_X = -100.0 * AU;
    const double MAX_X = 100.0 * AU;
    const double MIN_Y = -100.0 * AU;
    const double MAX_Y = 100.0 * AU;
    const double MIN_MASS = 1E+23;
    const double MAX_MASS = 1E+25;
    std::uniform_real_distribution<> y_dis(MIN_Y, MAX_Y);
    std::uniform_real_distribution<> mass_dis(MIN_MASS, MAX_MASS);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dis(MIN_X, MAX_X);
    Planet* planets;
    Planet* d_planets;
    planets = (Planet*)malloc(n);
    cudaMalloc((void**)&d_planets, n);
    //Planet planets[n];
    for (int i = 0; i < n; i++)
    {
        double x = x_dis(gen);
        double y = y_dis(gen);
        double mass = mass_dis(gen);
        planets[i] = Planet(mass,x, y);
    }
    double* x_coordinates, * y_coordinates;
    double* d_x_coordinates, * d_y_coordinates;
    int size = n * units * sizeof(double);
    cudaMalloc((void**)&d_x_coordinates, size);
    cudaMalloc((void**)&d_y_coordinates, size);
    x_coordinates = (double*)malloc(size);
    y_coordinates = (double*)malloc(size);
    for (int t = 0; t < units; t++) {

        cudaMemcpy(d_planets, planets, n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_coordinates, x_coordinates, size, cudaMemcpyHostToDevice);
        updatePositions << <1000, 1000 >> > (d_planets, n, t, d_x_coordinates, d_y_coordinates);
        cudaMemcpy(planets, d_planets, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(x_coordinates, d_x_coordinates, size, cudaMemcpyDeviceToHost);
        //cudaFree(d_planets);
        cudaThreadSynchronize();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        printf("x = %.1f\n", x_coordinates[t * n]);
        printf("x = %.1f\n", x_coordinates[t * n + 1]);
        printf("x = %.1f\n", x_coordinates[t * n + 2]);
        printf("x = %.1f\n", x_coordinates[t * n + 3]);
        printf("x = %.1f\n", x_coordinates[t * n + 4]);

    }
    
    printf("x = %.1f\n", x_coordinates[2 * n + 1]);
    /*std::ofstream file("planets_coordinates2.csv");
    file << "x,y" << std::endl;
    for (int i = 0; i < n; i++)
    {
        file << planets[i].x << "," << planets[i].y << std::endl;
    }
    file.close();
    */
    cudaFree(d_planets);
    return 0;
}

