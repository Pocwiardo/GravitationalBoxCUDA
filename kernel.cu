
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h> 
#include <random>
#include <fstream>
#include <stdio.h>
#include <chrono>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#define THREADS_PER_BLOCK 512

#define INTERVAL 20 //kroki obliczeń wykonywane w jednym wątku (zanim dane zostaną zapisane do pliku), efektywnie ilość dni / krok symulacyjny
#define UNITS 1000 //jednostki czasowe (długość symulacji)
const double AU = 1.496e+11; //jednostka astronomiczna
__device__ const double G = 6.67428e-11;
__device__ const double TIMESTEP = 3600 * 24; // 1 dzień (krok symulacji)



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
    GravitySource() : mass(2E+30), x(0), y(0) {}
    CUDA_CALLABLE_MEMBER GravitySource(double mass, double x, double y)
    {
        this->mass = mass;
        this->x = x;
        this->y = y;
    }
    CUDA_CALLABLE_MEMBER ~GravitySource()
    {
    }

private:
};

__global__ void updatePositions(Planet* planets, int n, int n_source, double* x_coordinates, double* y_coordinates, GravitySource* source)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (i < n)
    {
       
        double x = planets[i].x; 
        double y = planets[i].y;
        double vx = planets[i].vx;
        double vy = planets[i].vy;
        double mass = planets[i].mass;
        
        for (int j = 0; j < INTERVAL; j++) {
            double Fx = 0;
            double Fy = 0;
            for (int k = 0; k < n_source; k++) {
                double distance_x = source[k].x - x;
                double distance_y = source[k].y - y;
                double r = sqrt(distance_x * distance_x + distance_y * distance_y);
                double theta = atan2(distance_y, distance_x);
                double F = G * source[k].mass * mass / (r * r);
                Fx += F * cos(theta);
                Fy += F * sin(theta);

            }
            vx += (Fx / mass) * TIMESTEP;
            vy += (Fy / mass) * TIMESTEP;
            x += vx * TIMESTEP;
            y += vy * TIMESTEP;
        }
        planets[i].x = x;
        planets[i].y = y;
        planets[i].vx = vx;
        planets[i].vy = vy;
        x_coordinates[i] = x;
        y_coordinates[i] = y;
        
    }
}
void updatePositionsCPU(Planet* planets, int n, int n_source, double* x_coordinates, double* y_coordinates, GravitySource* source) {
    for (int i = 0; i < n; i++) {
        double x = planets[i].x;
        double y = planets[i].y;
        double vx = planets[i].vx;
        double vy = planets[i].vy;
        double mass = planets[i].mass;

        for (int j = 0; j < INTERVAL; j++) {
            double Fx = 0;
            double Fy = 0;
            for (int k = 0; k < n_source; k++) {
                double distance_x = source[k].x - x;
                double distance_y = source[k].y - y;
                double r = sqrt(distance_x * distance_x + distance_y * distance_y);
                double theta = atan2(distance_y, distance_x);
                double F = G * source[k].mass * mass / (r * r);
                Fx += F * cos(theta);
                Fy += F * sin(theta);

            }
            vx += (Fx / mass) * TIMESTEP;
            vy += (Fy / mass) * TIMESTEP;
            x += vx * TIMESTEP;
            y += vy * TIMESTEP;
        }
        planets[i].x = x;
        planets[i].y = y;
        planets[i].vx = vx;
        planets[i].vy = vy;
        x_coordinates[i] = x;
        y_coordinates[i] = y;

    }
}



int main()
{
    const int n = 20000;
    
    const double X_COR = 10.0 * AU;
    const double Y_COR = 0.0 * AU;
    const double MIN_MASS = 1E+23;
    const double MAX_MASS = 1E+25;
    const double X_VEL = 0;
    const double Y_VEL = 50000;
    const double SOURCE_MIN_MASS = 1E+30;
    const double SOURCE_MAX_MASS = 1E+31;
    const double SOURCE_COR = 20.0 * AU;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> mass_dis(MIN_MASS, MAX_MASS);
    std::uniform_real_distribution<> source_mass_dis(SOURCE_MIN_MASS, SOURCE_MAX_MASS);
    std::uniform_real_distribution<> vel_disX(-X_VEL, X_VEL);
    std::uniform_real_distribution<> vel_disY(-Y_VEL, Y_VEL);
    std::uniform_real_distribution<> x_dis(-X_COR, X_COR);
    std::uniform_real_distribution<> y_dis(-Y_COR, Y_COR);
    std::uniform_real_distribution<> source_x_dis(-SOURCE_COR, SOURCE_COR);
    std::uniform_real_distribution<> source_y_dis(-SOURCE_COR, SOURCE_COR);
    //double source_masses[3] = { 2E+30, 8E+30, 1E+30 };

    Planet* planets = new Planet[n];
    Planet* d_planets;
    int n_source;
    printf("Enter number of gravity sources: ");
    scanf("%d", &n_source);
    GravitySource* source = new GravitySource[n_source];
    GravitySource* d_source;

    cudaMalloc((void**)&d_source, n_source * sizeof(GravitySource));
    //source[0] = GravitySource(2E+30, 0, 0);
    //source[1] = GravitySource(8E+30, 10*AU, 10*AU);
    //source[2] = GravitySource(2E+30, -AU, 0);
    std::ofstream source_file("sources_coordinates.csv");
    printf("Sources' coordinates: \n");
    for (int i = 0; i < n_source; i++) {
        double cor;
        if (i < 1) cor = pow(10, 4-4*i);
        else cor = 1;
        double source_x = source_x_dis(gen) / cor;
        double source_y = source_y_dis(gen) / cor;
        double source_mass = source_mass_dis(gen);
        source[i] = GravitySource(source_mass, source_x, source_y);
        printf("x = %.10e ", source_x);
        printf("y = %.10e ", source_y);
        printf("mass = %.10e\n", source_mass);
        source_file << source_x << "," << source_y;
        if (i == n_source - 1) source_file << std::endl;
        else source_file << ",";
    }
    source_file.close();
    printf("Calculating... \n");
    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&d_planets, n * sizeof(Planet));
    for (int i = 0; i < n; i++)
    {
        double x = x_dis(gen);
        double y = y_dis(gen);
        double mass = mass_dis(gen);
        double vx = vel_disX(gen);
        double vy = vel_disY(gen);
        planets[i] = Planet(mass,x, y, vx, vy);
        
    }
    double* x_coordinates, * y_coordinates;
    double* d_x_coordinates, * d_y_coordinates;
    int size = n * sizeof(double);
    cudaMalloc((void**)&d_x_coordinates, size);
    cudaMalloc((void**)&d_y_coordinates, size);
    x_coordinates = (double*)malloc(size);
    y_coordinates = (double*)malloc(size);
    std::ofstream file("planets_coordinates.csv");
    
    //file << "x1,y1,x2,y2" << std::endl;
    unsigned int grid_size = ceil(n / THREADS_PER_BLOCK) + 1;
    for (int t = 0; t < UNITS; t++) {
        /*
        cudaMemcpy(d_source, source, n_source * sizeof(GravitySource), cudaMemcpyHostToDevice);
        cudaMemcpy(d_planets, planets, n * sizeof(Planet), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_coordinates, x_coordinates, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_coordinates, y_coordinates, size, cudaMemcpyHostToDevice);
        updatePositions << <grid_size, THREADS_PER_BLOCK >> > (d_planets, n, n_source, d_x_coordinates, d_y_coordinates, d_source);
        cudaMemcpy(planets, d_planets, n * sizeof(Planet), cudaMemcpyDeviceToHost);
        cudaMemcpy(source, d_source, n_source * sizeof(GravitySource), cudaMemcpyDeviceToHost);
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
        */
        updatePositionsCPU(planets, n, n_source, x_coordinates, y_coordinates, source);

        for (int k = 0; k < n; k++) {
            file << x_coordinates[k] << "," << y_coordinates[k];
            if(k==n-1) file << std::endl;
            else file << ",";
        }
    }
    
    file.close();
    cudaFree(d_planets);
    cudaFree(d_source);
    cudaFree(d_x_coordinates);
    cudaFree(d_y_coordinates);
    free(x_coordinates);
    free(y_coordinates);
    
    delete[] planets;
    delete[] source;
    auto now = std::chrono::high_resolution_clock::now() - start;
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now).count();
    printf("Done. Elapsed time: %d us \n", elapsed);
    printf("Visualisation: visualisation.py \n");
    printf("Coordinates of sources: sources_coordinates.csv \n");
    printf("Coordinates of particles (planets): planets_coordinates.csv \n");
    return 0;
}

