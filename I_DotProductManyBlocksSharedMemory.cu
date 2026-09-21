// Name: Emerson Scott
// Vector Dot product on many block and useing shared memory
// nvcc I_DotProductManyBlocksSharedMemory.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors smaller than the block size.

 Your tasks:
 - Extend the code to launch as many blocks as needed based on a fixed thread count and the vector length.
 - Use **shared memory** within each block to speed up the computation.
 - Pad the input with zeros to fill the last block, if necessary.
 - Perform the final reduction (summing partial results) on the **CPU**.
 - Set the thread count (block size) to 256.
 - Test your code by setting N to different values.
*/

/*
 Purpose:
 To understand that blocks do **not** synchronize with each other during a kernel call.
 In other words, you can't detect when **all blocks** are finished from inside the kernel.
 You can work around this by exiting the kernel, which ensures all blocks have completed.
 Also to learn how to use shared memory to speed up your code.
*/

/*
 Explain what you did to fix the code:
 1. Define the length of the vector and the size of each block
 2. Add necessary headers
 3. Make the PaddedN and NumBlocks variables/equations
 4. Do the multiplying and folding process until each block has one answer in the GPU
 5. Add each block's results together on the CPU
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 500 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
int PaddedN; //the size after adding zeros
int NumBlocks; //tells us how many blocks we need
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256; //Set each block to have 256 values
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	NumBlocks = (N + BlockSize.x - 1) / BlockSize.x; //integer division rounded upward - (500+256-1)/256=755/256-->2
	
	GridSize.x = NumBlocks;
	GridSize.y = 1;
	GridSize.z = 1;
	
	PaddedN = NumBlocks * BlockSize.x;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(PaddedN*sizeof(float)); //use the PaddedN instead of just N - needs the entire padded vector
	B_CPU = (float*)malloc(PaddedN*sizeof(float)); //use the PaddedN instead of just N - needs the entire padded vector
	C_CPU = (float*)malloc(N*sizeof(float)); //only needs one result per block
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,NumBlocks*sizeof(float)); //only needs to store the two answers from A and B
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < PaddedN; i++)
	{		
		if(i < N)
		{
	
			A_CPU[i] = (float)i; //puts the current value of i into A_CPU as a float
			B_CPU[i] = (float)(3*i); //puts 3 times i into position i of B_CPU - 3 is just a random number to easily know what is in element B
		}
		else //makes the padded 0's
		{
			A_CPU[i] = 0.0f;
			B_CPU[i] = 0.0f;
		}
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x; //define id with using multiple blocks
	
	__shared__ float sharedData[256]; //creates an array in shared memory - every block gets its own copy - they do not share the same sharedData
	
	sharedData[threadIdx.x] = a[id] * b[id]; //each thread calculates one multiplication - then the result is stored in shared memory
	
	__syncthreads(); //synchronize the threads
		
	int fold = blockDim.x / 2; 
	
	while(0 < fold) //start folding and reducing
	{
		if(threadIdx.x < fold) //add pairs together
		{
			sharedData[threadIdx.x] = 
				sharedData[threadIdx.x] + 
				sharedData[threadIdx.x + fold];
		}
		
		__syncthreads();
		
		fold = fold / 2;
	}
		if(threadIdx.x == 0)
		{
			c[blockIdx.x] = sharedData[0]; //contains the sum for that block
		}
}

// Checking to see if anything went wrong in the dot product.
bool check(float cpuAnswer, float gpuAnswer, float tolerance)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, PaddedN*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, PaddedN*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, NumBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	DotGPU = C_CPU[0]; // C_GPU was copied into C_CPU.
	
	DotGPU = 0.0f; //sets DotGPU to zero before starting to add things to it
	
	for(int i = 0; i < NumBlocks; i++) //go through every block's result, one at a time
	{
		DotGPU += C_CPU[i];
	}
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


