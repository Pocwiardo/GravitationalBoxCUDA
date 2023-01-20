
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <fstream>
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

const double AU = 1.496e+11;
__device__ const double G = 6.67428e-11;
const double SCALE = 250 / AU;  // 1AU = 100 pixels
__device__ const double TIMESTEP = 3600 * 24 * 100; // 1 day


class Planet
{
    
public:
    double mass;
    double x;
    double y;
    double vx;
    double vy;
    Planet() : mass(1E+23), x(AU), y(AU) {}
    CUDA_CALLABLE_MEMBER Planet(double mass, double x, double y)
    {
        this->mass = mass;
        this->x = x;
        this->y = y;
        this->vx = 0;
        this->vy = 0;
    }
    CUDA_CALLABLE_MEMBER ~Planet()
    {
        //cudaFree(mass);
    }

private:
};
class GravitySource
{

public:
    double mass;
    double x;
    double y;
    double vx;
    double vy;
    GravitySource() : mass(1E+30), x(0), y(0) {}
    CUDA_CALLABLE_MEMBER GravitySource(double mass, double x, double y)
    {
        this->mass = mass;
        this->x = x;
        this->y = y;
        this->vx = 0;
        this->vy = 0;
    }
    CUDA_CALLABLE_MEMBER ~GravitySource()
    {
    }

private:
};

__global__ void updatePositions(Planet* planets, int n, int t, double* x_coordinates, double* y_coordinates, GravitySource sun)
{
    // Indeks wątku
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pamięć dzielona dla przechowywania pozycji planety
    __shared__ double x;
    __shared__ double y;

    // Jeśli indeks wątku jest mniejszy niż liczba planet
    if (i < n)
    {
        // Pobranie pozycji planety do pamięci dzielonej
        x = planets[i].x; 
        y = planets[i].y;
        // Synchronizacja wątków w bloku
        __syncthreads();
        

        // Obliczanie nowych pozycji dla planety
        //double F = 0;
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
        double distance_x = sun.x - planets[i].x;
        double distance_y = sun.y - planets[i].y;
        double r = sqrt(distance_x * distance_x + distance_y * distance_y);
        double theta = atan2(distance_y, distance_x);
        double F = G * sun.mass * planets[i].mass / (r * r);
        Fx = F * cos(theta);
        Fy = F * sin(theta); //z ta matma trzeba sprawdzic że sie wektor nie odwraca //to += chyb chociaz chuj wi
       
        // Aktualizacja pozycji planety
        planets[i].vx += Fx / planets[i].mass * TIMESTEP;
        planets[i].vy += Fy / planets[i].mass * TIMESTEP;
        planets[i].x += planets[i].vx * TIMESTEP;
        planets[i].y += planets[i].vy * TIMESTEP;
        //planets[i].x += 1e+12;
        //planets[i].y += 1e+11;

        // Zapis koordynatów do tablic
        //x_coordinates[t*n + i] = planets[i].x;
        //y_coordinates[t*n + i] = planets[i].y;
        x_coordinates[t * n + i] = planets[i].x;
        y_coordinates[t * n + i] = planets[i].y;
    }
}



int main()
{
    const int n = 100;
    const int units = 50; //jednostki czasowe
    const double MIN_X = -10.0 * AU;
    const double MAX_X = 10.0 * AU;
    const double MIN_Y = -10.0 * AU;
    const double MAX_Y = 10.0 * AU;
    const double MIN_MASS = 1E+23;
    const double MAX_MASS = 1E+25;
    std::uniform_real_distribution<> y_dis(MIN_Y, MAX_Y);
    std::uniform_real_distribution<> mass_dis(MIN_MASS, MAX_MASS);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dis(MIN_X, MAX_X);
    Planet* planets = new Planet[n];;
    Planet* d_planets;
    GravitySource sun;
    sun = GravitySource(1E+30, 0, 0);
    //printf("masa = %.30f\n", G);
    //GravitySource* d_sun;
    //cudaMalloc((void**)&d_sun, n);
    //planets = (Planet*)malloc(n);
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
    int size = (n * units) * sizeof(double);
    cudaMalloc((void**)&d_x_coordinates, size);
    cudaMalloc((void**)&d_y_coordinates, size);
    x_coordinates = (double*)malloc(size);
    y_coordinates = (double*)malloc(size);
    std::ofstream file("planets_coordinates3.csv");
    file << "x,y" << std::endl;
    for (int t = 0; t < units; t++) {

        cudaMemcpy(d_planets, planets, n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_coordinates, x_coordinates, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_coordinates, y_coordinates, size, cudaMemcpyHostToDevice);
        updatePositions << <1000, 1000 >> > (d_planets, n, t, d_x_coordinates, d_y_coordinates, sun);
        cudaMemcpy(planets, d_planets, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(x_coordinates, d_x_coordinates, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(y_coordinates, d_y_coordinates, size, cudaMemcpyDeviceToHost);
        //cudaFree(d_planets);
        cudaThreadSynchronize();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        printf("x = %.1f ", x_coordinates[t * n]);
        printf("y = %.1f\n", y_coordinates[t * n]);
        file << x_coordinates[t * n] << "," << y_coordinates[t * n] << std::endl;
        //printf("x = %.1f\n", x_coordinates[t * n + 1]);
        //printf("x = %.1f\n", x_coordinates[t * n + 2]);
        //printf("x = %.1f\n", x_coordinates[t * n + 3]);
        //printf("x = %.1f\n", x_coordinates[t * n + 4]);

    }
    
    //printf("x = %.1f\n", x_coordinates[2 * n + 1]);
    /*std::ofstream file("planets_coordinates2.csv");
    file << "x,y" << std::endl;
    for (int i = 0; i < n; i++)
    {
        file << planets[i].x << "," << planets[i].y << std::endl;
    }
    file.close();
    */
    printf("aaaaa");
    cudaFree(d_planets);
    cudaFree(d_x_coordinates);
    cudaFree(d_y_coordinates);
    //free(planets);
    free(x_coordinates);
    free(y_coordinates);
    //for (int i = 0; i < n; i++)
    //{
    //    cudaFree(d_planets + i);
    //    free(planets + i);
    //}
    delete[] planets;
    return 0;
}

