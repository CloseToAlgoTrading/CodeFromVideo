//nvcc -lcurand -I/usr/include/thrust/system/cuda -I./ -L/usr/lib/cuda cuda_pi.cu -o cuda_pi

#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <curand.h>
 
#define CHECKCURAND(expression)                         \
  {                                                     \
    curandStatus_t status = (expression);                         \
    if (status != CURAND_STATUS_SUCCESS) {                        \
      std::cerr << "Curand Error on line " << __LINE__<< std::endl;     \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  }
 
// atomicAdd is introduced for compute capability >=6.0
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
      //printf("device arch <=600\n");
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
          unsigned long long int old = *address_as_ull, assumed;
            do {
                    assumed = old;
                        old = atomicCAS(address_as_ull, assumed,
                                                    __double_as_longlong(val + __longlong_as_double(assumed)));
                          } while (assumed != old);
              return __longlong_as_double(old);
}
#endif
 
__global__ void sumPayoffKernel(float *d_s, const unsigned N_PATHS, double *mysum)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;
 
  extern __shared__ double smdata[];
  smdata[tid] = 0.0;
 
  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    smdata[tid] += (double) d_s[i];
  }
 
  for (unsigned s=blockDim.x/2; s>0; s>>=1)
  {
    __syncthreads();
    if (tid < s) smdata[tid] += smdata[tid + s];
  }
 
  if (tid == 0)
  {
    atomicAdd(mysum, smdata[0]);
  }
}
 
__global__ void MonteCarloMdd(
    float *d_s,
    const float S0,
    const float mu,
    const float sigma,
    const long N,
    const long N_PATHS,
    const float * d_normals)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;

  const double drift_0 = (mu - 0.5 * sigma*sigma);
  int index = 0;
 
  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    float S;
    float mdd = 0.0;
    float maxPrev = S0;
    float W = 0.0;
    
    for (int k=1; k<=N; ++k) 
    {
      index = (k-1) + (i) * N;
      W += d_normals[index];
      S = S0*expf((drift_0 * k) + (sigma*W));
      mdd = fmaxf(mdd, (maxPrev - S)/maxPrev);
      maxPrev = fmaxf(maxPrev, S);
    }
    d_s[i] = mdd;
  }  
}
 

int main(int argc, char *argv[]) {
  try {
    // declare variables and constants
    const float S0 = 205.42999267578125;
    const float mu = 0.00035128687077953227;
    const float sigma = 0.00781510977845114;
    float N_PATHS = 10000;
    float N_STEPS = 100;
    if (argc >= 2)  N_PATHS = atoi(argv[1]);
 
    if (argc >= 3)  N_STEPS = atoi(argv[2]);

 
 
    double gpu_sum{0.0};
 
    int devID{0};
    cudaDeviceProp deviceProps;
 
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",  devID, deviceProps.name, deviceProps.major, deviceProps.minor);

    // Generate random numbers on the device
    curandGenerator_t curandGenerator;
    CHECKCURAND(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32));
    CHECKCURAND(curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL)) ;
 
    const size_t N_NORMALS = (size_t)N_STEPS * N_PATHS;
    float *d_normals;
    checkCudaErrors(cudaMalloc(&d_normals, N_NORMALS * sizeof(float)));
    CHECKCURAND(curandGenerateNormal(curandGenerator, d_normals, N_NORMALS, 0.0f, 1.0f));
    cudaDeviceSynchronize();
 
      // before kernel launch, check the max potential blockSize
      int BLOCK_SIZE, GRID_SIZE;
      checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&GRID_SIZE,
                                                         &BLOCK_SIZE,
                                                         MonteCarloMdd,
                                                         0, N_PATHS));
 
      std::cout << "suggested block size " << BLOCK_SIZE
                << " \nsuggested grid size " << GRID_SIZE
                << std::endl;
 
      std::cout << "Used grid size " << GRID_SIZE << std::endl;
 
      // Kernel launch
      auto t1=std::chrono::high_resolution_clock::now();
 
      float *d_s;
      checkCudaErrors(cudaMalloc(&d_s, N_PATHS*sizeof(float)));
       
      MonteCarloMdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, S0, mu, sigma, N_STEPS, N_PATHS, d_normals);
      cudaDeviceSynchronize();
 
      double* mySum;
      checkCudaErrors(cudaMallocManaged(&mySum, sizeof(double)));
      sumPayoffKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(d_s, N_PATHS, mySum);
      cudaDeviceSynchronize();
      
      gpu_sum = mySum[0] / N_PATHS;
      
      auto t2=std::chrono::high_resolution_clock::now();
 
      // clean up
      CHECKCURAND(curandDestroyGenerator( curandGenerator )) ;
      checkCudaErrors(cudaFree(d_s));
      checkCudaErrors(cudaFree(d_normals));
      checkCudaErrors(cudaFree(mySum));
 
      std::cout << "mdd "
              << gpu_sum * 100 << "%"
              << " time "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() * 1e-9
                << " s\n";
  }
 
  catch(std::
        exception& e)
  {
    std::cout<< "exception: " << e.what() << "\n";
  }
} 