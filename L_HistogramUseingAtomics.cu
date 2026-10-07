// Name: Emerson Scott
// Histogram useing atomics in global memory and shared memory.
// nvcc L_HistogramUseingAtomics.cu -o temp

/*
 What to do:
 This code generates a series of random numbers and places them into bins based on size ranges using the CPU.

 Your task:
 - Create a binning scheme that utilizes the GPU.
 - Take advantage of both global and shared memory atomic operations.
 - The function call has already been provided.
 - Set the block size to **twice** the number of multiprocessors on the GPU.
*/

/*
 Purpose:
 To learn how to use atomic operations at both the shared and global memory levels.
 Along the way, you'll also learn a bit about generating random numbers using `srand`,
 which will come in handy when we use `curand` in a later assignment.
*/

/*
 Explain what you did to fix the code:
 1. Fix the block size
 2. Fill out the GPU kernel
 3. Add relevant headers
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 Note: The Max int value is 2,147,483,647
 The length of the sequence of random number that srand generates is 2^32,
 that is 4,294,967,296 which is bigger than the largest int but the max for an unsigned int.
*/

// Defines
#define NUMBER_OF_RANDOM_NUMBERS 2147483 //the program will generate 2,147,483 numbers
#define NUMBER_OF_BINS 10 //the histogram will have 10 bins - bins 0 through 9
#define MAX_RANDOM_NUMBER 100.0f //the largest number is approximately 100

// Global variables
float *RandomNumbersGPU; //pointer to floating-point numbers that will live in GPU memory
int *HistogramGPU; //pointer to an integer array in GPU memory that will contain the final GPU histogram
float *RandomNumbersCPU; //pointer to floating-point numbers in CPU memory - where the random numbers initially live
int *HistogramCPU; //histogram calculated by the CPU
int *HistogramCPUTemp; // Use it to hold the GPU histogram past back so we can compare to CPU histogram.
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//Function prototypes
void cudaErrorCheck(const char *, int);
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
void fillHistogramCPU();
__global__ void fillHistogramGPU(float *, int *); //GPU kernel that can be called by the CPU and runs on the GPU
int main();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line) //function checks whether a CUDA error occurred
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices() //determines how the GPU threads are organized
{
	cudaDeviceProp prop; //holds the information about the GPU
	cudaGetDeviceProperties(&prop, 0); //"give me the properties of GPU number 0
	cudaErrorCheck(__FILE__, __LINE__); //checks whether getting the GPU information caused an error
	
	BlockSize.x = 2 * prop.multiProcessorCount; //sets the block size to twice the number of multiprocessors (assignment says to)
	
	if(prop.maxThreadsDim[0] < BlockSize.x) //checking maximum threads
	{
		printf("\n You are trying to create more threads (%d) than your GPU can support on a block (%d).\n Good Bye\n", BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (NUMBER_OF_RANDOM_NUMBERS - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector - calculates how many blocks are needed
	
	if(prop.maxGridSize[0] < GridSize.x) //checks whether there are more blocks than the GPU allows
	{
		printf("\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n", GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets memory on the GPU and CPU for our use.
void AllocateMemory() //function creates the memory needed
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&RandomNumbersGPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&HistogramGPU, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	RandomNumbersCPU = (float*)malloc(NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	HistogramCPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	HistogramCPUTemp = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	
	//Setting the the histograms to zero.
	cudaMemset(HistogramGPU, 0, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	memset(HistogramCPU, 0, NUMBER_OF_BINS*sizeof(int));
}

//Loading random numbers.
void Innitialize()
{
	time_t t;
	srand((unsigned) time(&t)); //sets the random-number seed
	
	// rand() returns an int in [0, RAND_MAX] "end points included".
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{		
		RandomNumbersCPU[i] = MAX_RANDOM_NUMBER*(float)rand()/RAND_MAX; //creates a random floating-point number - creates a value between 0.0 and 1.0
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(RandomNumbersGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	free(RandomNumbersCPU); 
	free(HistogramCPU);
	free(HistogramCPUTemp);
	//printf("\n Cleanup Done.");
}

void fillHistogramCPU() //calculates the histogram using the CPU
{
	float breakPoint; //stores the upper limit of the current bin
	int k, done; //k keeps track of what bin is being checked and done is when we've found the correct bin
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++) //go through every random number
	{
		breakPoint = stepSize; //start the first bin's upper boundary at these assignments
		k = 0; //start at bin 0
		done = 0; //the correct bin has not been found yet
		while(done == 0) //keep looking until the correct bin is found
		{
			if(RandomNumbersCPU[i] < breakPoint)
			{
				HistogramCPU[k]++; //increases the bin's count by 1
				done = 1; //WE FOUND THE BIN YAYYAYYYAYYAYYAYYYY
			}
			
			if(NUMBER_OF_BINS < k) //make sure k is not too large!
			{
				printf("\n k is too big\n");
				exit(0);
			}
			k++; //move to the next bin
			breakPoint += stepSize; //increase the breakpoint by 10
		}
	}
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void fillHistogramGPU(float *randomNumbers, int *hist)
{
	__shared__ int sharedHistogram[NUMBER_OF_BINS]; //creates an array in shared memory
	
	int id = blockIdx.x * blockDim.x + threadIdx.x; //calculate the thread's ID - gives every GPU thread a unique number
	
	if(threadIdx.x < NUMBER_OF_BINS)
	{
		sharedHistogram[threadIdx.x] = 0; //set the shared histogram to zero
	}
	
	__syncthreads(); //"don't leave me behind thread friends!!!"
	
	if(id < NUMBER_OF_RANDOM_NUMBERS) //make sure the thread is actually processing a number - protects us from going out of bounds
	{
		//calculate the bin size
		float stepSize = MAX_RANDOM_NUMBER / (float)NUMBER_OF_BINS;
		
		int bin = (int)(randomNumbers[id] / stepSize);
		
		if(bin >= NUMBER_OF_BINS - 1) //protect the last bin
		{
			bin = NUMBER_OF_BINS - 1;
		}
		
		atomicAdd(&sharedHistogram[bin], 1); //prevents the race condition to happen - "safely add 1 to this bin, even if other threads are trying to add to it at the same time" - like the fast pass at disney world watching everybody else wait for the same ride...
	}
	
	__syncthreads();
	
	if(threadIdx.x < NUMBER_OF_BINS) //takes each block's histogram and adds it to the final histogram
	{
		atomicAdd(&hist[threadIdx.x], sharedHistogram[threadIdx.x]); //global-memory atomic operation
	}
}

int main()
{
	float time;
	timeval start, end;
	
	long int test = NUMBER_OF_RANDOM_NUMBERS;
	if(2147483647 < test)
	{
		printf("\nThe length of your vector is longer than the largest integer value allowed of 2,147,483,647.\n");
		printf("You should check your code.\n Good Bye\n");
		exit(0);
	}
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	fillHistogramCPU();
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on CPU = %.15f milliseconds\n", (time/1000.0));
	
	gettimeofday(&start, NULL);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(RandomNumbersGPU, RandomNumbersCPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	fillHistogramGPU<<<GridSize,BlockSize>>>(RandomNumbersGPU, HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(HistogramCPUTemp, HistogramGPU, NUMBER_OF_BINS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on GPU = %.15f milliseconds\n", (time/1000.0));
	
	//Check
	for(int i = 0; i < NUMBER_OF_BINS; i++)
	{
		printf("\n Difference in histogram bins %d is %d.", i, abs(HistogramCPUTemp[i] - HistogramCPU[i]));
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	printf("\n\n");
	return(0);
}
