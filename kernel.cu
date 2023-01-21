
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h> 
#include <random>
#include <fstream>
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#define THREADS_PER_BLOCK 512

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

const double AU = 1.496e+11;
__device__ const double G = 6.67428e-11;
__device__ const double TIMESTEP = 3600 * 24; // 1 day


class Planet
{
    
public:
    double mass;
    double x;
    double y;
    double vx;
    double vy;
    Planet() : mass(1E+23), x(AU), y(AU), vx(0),vy(0) {}
    CUDA_CALLABLE_MEMBER Planet(double mass, double x, double y, double vx, double vy)
    {
        this->mass = mass;
        this->x = x;
        this->y = y;
        this->vx = vx;
        this->vy = vy;
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
    GravitySource() : mass(2E+30), x(0), y(0),vx(0),vy(0) {}
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
    //__shared__ double x;
    //__shared__ double y;

    // Jeśli indeks wątku jest mniejszy niż liczba planet
    if (i < n)
    {
        // Pobranie pozycji planety do pamięci dzielonej
        double x = planets[i].x; 
        double y = planets[i].y;
        //if(i==0) printf("  F = %.10e\n ", y);
        double vx = planets[i].vx;
        double vy = planets[i].vy;
        double mass = planets[i].mass;
        // Synchronizacja wątków w bloku
        //__syncthreads();
        /*if (i == 0) {
            printf("Przed: \n");
            printf("  x = %.10e ", x);
            printf("  y = %.10e\n", y);
            printf("  vx = %.10e ", planets[i].vx);
            printf("  vy = %.10e\n ", planets[i].vy);
            //printf("  xsh = %.10e", vx);
            //printf("  ysh = %.10e\n ", vy);
        }*/

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
        for (int j = 0; j < 20; j++) {
            double distance_x = sun.x - x;
            double distance_y = sun.y - y;
            double r = sqrt(distance_x * distance_x + distance_y * distance_y);
            //if(i==0) printf("  r = %.10e\n", r);
            double theta = atan2(distance_y, distance_x);
            double F = G * sun.mass * mass / (r * r);
            Fx = F * cos(theta);
            Fy = F * sin(theta); 

            // Aktualizacja pozycji planety
            vx += (Fx / mass) * TIMESTEP;
            vy += (Fy / mass) * TIMESTEP;
            x += vx * TIMESTEP;
            y += vy * TIMESTEP;
            /*if (i == 6) {
                printf("Po: \n");
                printf("  distx = %.10e ", distance_x);
                printf("  r = %.10e\n", r);
                printf("  theta = %.10e ", theta);
                printf("  F = %.10e\n ", F);
                printf("  Fx = %.10e\n ", Fx);
                printf("  x = %.10e\n ", x);
                printf("  vx = %.10e\n ", vx);
                printf("out: \n");
                //printf("  xsh = %.10e", vx);
                //printf("  ysh = %.10e\n ", vy);
            }*/
        }
        //__syncthreads();
        //planets[i].x += 1e+12;
        //planets[i].y += 1e+11;

        // Zapis koordynatów do tablic
        //x_coordinates[t*n + i] = planets[i].x;
        //y_coordinates[t*n + i] = planets[i].y;
        planets[i].x = x;
        planets[i].y = y;
        planets[i].vx = vx;
        planets[i].vy = vy;
        //x_coordinates[t * n + i] = x;
        //y_coordinates[t * n + i] = y;
        __syncthreads();
        x_coordinates[i] = x;
        y_coordinates[i] = y;
        //printf("i = %d \n", i);
        /*if (i == 0) {
            printf("Po: \n");
            printf("  Fx = %.10e ", Fx);
            printf("  Fy = %.10e\n", Fy);
            printf("  vx = %.10e ", planets[i].vx);
            printf("  vy = %.10e\n ", planets[i].vy);
            //printf("  xsh = %.10e", vx);
            //printf("  ysh = %.10e\n ", vy);
        }*/
    }
}



int main()
{
    const int n = 20000;
    const int units = 100; //jednostki czasowe
    const double X_COR = 10.0 * AU;
    const double Y_COR = 0.0 * AU;
    const double MIN_MASS = 1E+23;
    const double MAX_MASS = 1E+25;
    const double X_VEL = 0;
    const double Y_VEL = 50000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> mass_dis(MIN_MASS, MAX_MASS);
    std::uniform_real_distribution<> vel_disX(-X_VEL, X_VEL);
    std::uniform_real_distribution<> vel_disY(-Y_VEL, Y_VEL);
    std::uniform_real_distribution<> x_dis(-X_COR, X_COR);
    std::uniform_real_distribution<> y_dis(-Y_COR, Y_COR);
    Planet* planets = new Planet[n];;
    Planet* d_planets;
    GravitySource sun;
    sun = GravitySource(2E+30, 0, 0);
    //printf("masa = %.30f\n", G);
    //GravitySource* d_sun;
    //cudaMalloc((void**)&d_sun, n);
    //planets = (Planet*)malloc(n);
    cudaMalloc((void**)&d_planets, n * sizeof(Planet));
    //Planet planets[n];
    for (int i = 0; i < n; i++)
    {
        double x = x_dis(gen);
        double y = y_dis(gen);
        double mass = mass_dis(gen);
        double vx = vel_disX(gen);
        double vy = vel_disY(gen);
        planets[i] = Planet(mass,x, y, vx, vy);
        
    }
    //printf("6 = %.1f\n", planets[6].x);
    double* x_coordinates, * y_coordinates;
    double* d_x_coordinates, * d_y_coordinates;
    //int size = (n * units) * sizeof(double);
    int size = n * sizeof(double);
    cudaMalloc((void**)&d_x_coordinates, size);
    cudaMalloc((void**)&d_y_coordinates, size);
    x_coordinates = (double*)malloc(size);
    y_coordinates = (double*)malloc(size);
    std::ofstream file("planets_coordinates.csv");
    //file << "x1,y1,x2,y2" << std::endl;
    unsigned int grid_size = ceil(n / THREADS_PER_BLOCK) + 1;
    for (int t = 0; t < units; t++) {

        cudaMemcpy(d_planets, planets, n * sizeof(Planet), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_coordinates, x_coordinates, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_coordinates, y_coordinates, size, cudaMemcpyHostToDevice);
        updatePositions << <grid_size, THREADS_PER_BLOCK >> > (d_planets, n, t, d_x_coordinates, d_y_coordinates, sun);
        cudaMemcpy(planets, d_planets, n * sizeof(Planet), cudaMemcpyDeviceToHost);
        cudaMemcpy(x_coordinates, d_x_coordinates, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(y_coordinates, d_y_coordinates, size, cudaMemcpyDeviceToHost);

        //cudaFree(d_planets);
        cudaThreadSynchronize();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        //printf("t = %d \n", t);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        //printf("x = %.10e ", x_coordinates[6]);
        //printf("y = %.10e\n", y_coordinates[6]);
        for (int k = 0; k < n; k++) {
            //file << x_coordinates[t * n + i] << "," << y_coordinates[t * n + i] << "," << x_coordinates[t * n + 1] << "," << y_coordinates[t * n + 1] << std::endl;
            //file << x_coordinates[t * n + k] << "," << y_coordinates[t * n + k];
            file << x_coordinates[k] << "," << y_coordinates[k];
            if(k==n-1) file << std::endl;
            else file << ",";
            //printf("x = %.1f\n", x_coordinates[t * n + 1]);
            //printf("x = %.1f\n", x_coordinates[t * n + 2]);
            //printf("x = %.1f\n", x_coordinates[t * n + 3]);
            //printf("x = %.1f\n", x_coordinates[t * n + 4]);
        }
    }
    //rintf("6 = %.10e ", planets[6].x);
    
    //printf("x = %.1f\n", x_coordinates[2 * n + 1]);
    /*std::ofstream file("planets_coordinates2.csv");
    file << "x,y" << std::endl;
    for (int i = 0; i < n; i++)
    {
        file << planets[i].x << "," << planets[i].y << std::endl;
    }
    file.close();
    */
    //printf("aaaaa");
    file.close();
    cudaFree(d_planets);
    cudaFree(d_x_coordinates);
    cudaFree(d_y_coordinates);
    //free(planets);
    free(x_coordinates);
    free(y_coordinates);
    
    delete[] planets;
    return 0;
}

