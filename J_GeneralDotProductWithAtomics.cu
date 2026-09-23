// Name: Emerson Scott
// Robust Vector Dot product 
// nvcc J_GeneralDotProductWithAtomics.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors of any length using shared memory to
 reduce the number of global memory accesses. However, since blocks can’t synchronize
 with each other, the final reduction must be handled on the CPU.

 To simplify the GPU-side logic, we’ll add some “pregame” setup and use atomic adds.

 1. Make sure the number of threads per block is a power of 2. This avoids messy edge
    cases during the reduction step. If it’s not a power of 2, print an error message
    and exit. (Without this, you'd have to check if the reduction is even or not,
    add the last element to the first, adjust the loop, etc.)

 2. Calculate the correct number of blocks needed to process the entire vector.
    Then check device properties to ensure the grid and block sizes are within hardware limits.
    Just because it works on your fancy GPU doesn’t mean it will work on your client’s older one.
    If the block or grid size exceeds the device’s capabilities, report the issue and exit gracefully.

 3. It’s inefficient to check inside your kernel if a thread is working past the end of the vector
    on every iteration. Instead, figure out how many extra elements are needed to fill out the grid,
    and pad the vector with zeros. Zero-padding doesn’t affect the dot product (0 * anything = 0).
    Use `cudaMemset` to explicitly zero out your device memory — don’t rely on "getting lucky"
    like you might have in previous assignments.

 4. In previous assignments, we had to do the final reduction on the CPU because we couldn't sync blocks.
    Now, use **atomic adds** to sum partial results directly on the GPU and avoid CPU post-processing.
    Then, copy the final result back to the CPU using `cudaMemcpy`.

    Note: Atomic operations on floats are only supported on GPUs with compute capability 3.0 or higher.
    Use device properties to check this before running the kernel.
    While you’re at it, if multiple GPUs are available, select the best one based on compute capability.

 5. Add any additional bells and whistles to make your code more robust and user-proof.
    Think of edge cases or bad input your client might provide and handle it cleanly.
*/

/*
 Purpose:
 To learn how to use atomic adds to avoid jumping out of the kernel for block synchronization.
 This is also your opportunity to make the code "foolproof" — handling edge cases gracefully.

 At this point, you should understand all the CUDA basics.
 From now on, we’ll focus on refining that knowledge and adding advanced features.
*/

/*
 Explain what you did to fix the code:
 1. Define the block size to be a power of 2
 2. Add necessary headers
 3. Add the padded vector variable
 4. Choose the GPU with the highest compute capability
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 256 // Threads in a block - changed to a power of 2

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int PaddedN; //this stores the size of the padded GPU vectors

// Function prototypes
void cudaErrorCheck(const char*, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*);
bool check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error; //CUDA data type used to store an error code
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
	int deviceCount;
	
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(deviceCount == 0)
	{
		printf("No CUDA devices were found.\n");
		exit(0);
	}
	 //using -1 as placeholders
	int bestDevice = -1;
	int bestMajor = -1;
	int bestMinor = -1;
	
	for(int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp prop; //CUDA structure containing information about a GPU
		
		cudaGetDeviceProperties(&prop, i); //fills prop with the information for GPU i
		cudaErrorCheck(__FILE__, __LINE__);
		
		printf("\nGPU %d: %s", i, prop.name);
		printf("\nCompute capability: %d.%d\n", prop.major, prop.minor); //contains the compute capability
		
	// Float atomicAdd requires compute capability 3.0+
	if(prop.major < 3)
	{
		printf("GPU %d does not support float atomicAdd.\n", i);
		continue;
	}
	
	//Make sure this GPU can support our block size
	if(BLOCK_SIZE > prop.maxThreadsPerBlock)
	{
		printf("GPU %d cannot support %d threads per block.\n", i, BLOCK_SIZE);
		continue;
	}
	
	//Choose the GPU with the highest compute capability
	if(prop.major > bestMajor || (prop.major == bestMajor && prop.minor > bestMinor)) //chooses the GPU with the highest computer capability - so... pick this GPU if its major version is higher, OR if the major versions are equal and its minor version is higher
	{
		bestDevice = i;
		bestMajor = prop.major;
		bestMinor = prop.minor;
	}
}
if(bestDevice == -1)
{
	printf("\nNo suitable GPU was found.\n");
	exit(0);
}
	cudaSetDevice(bestDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, bestDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	printf("\nUsing GPU %d: %s\n",
		bestDevice, prop.name);

	printf("Compute capability: %d.%d\n",
 	       prop.major, prop.minor);

	
	//Set block size
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Calculate number of blocks
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks. (100000 - 1) / 256 + 1 = 391 blocks
	GridSize.y = 1;
	GridSize.z = 1;
	
	//Check grid size against GPU hardware limit
	if(GridSize.x > prop.maxGridSize[0])
	{
		printf("\nGrid size is too large for this GPU.\n");
		exit(0);
	}
	
	PaddedN = GridSize.x * BLOCK_SIZE; //gives us enough room for every thread in every block --> 391 x 256 = 100096
	
	printf("Original vector size = %d\n", N);
	printf("Block size = %d\n", BLOCK_SIZE);
	printf("Number of blocks = %d\n", GridSize.x);
	printf("Padded vector size = %d\n", PaddedN);
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	if(A_CPU == NULL || B_CPU == NULL || C_CPU == NULL)
	{
		printf("CPU memory allocation failed.\n");
		exit(0);
	}
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,sizeof(float)); //Now only one float because we're using atomicAdd so every block contributes to the same number
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will doting.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	C_CPU[0] = 0.0f;
	
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[0] += a[id] * b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c)
{
	int threadIndex = threadIdx.x; //tells you the thread's position inside its block
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x; //Determines which element of the vector this thread should work on
	__shared__ float c_sh[BLOCK_SIZE]; //shared memory for this block
	
	c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]); //each thread calculates one multiplication
	__syncthreads(); //wait until all threads have filled shared memory
	
	int fold = blockDim.x; //reduce the values inside the block
	while(1 < fold)
	{
		fold = fold/2;
		if(threadIndex < fold)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
	if(threadIndex == 0) //thread 0 has the answer for this block
	{
		atomicAdd(&c[0], c_sh[0]); //add the answer to the global GPU result - this prevents the blocks from modifying c[0] at the same time
	}	// c[0] = sum of every block's answer
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
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
	cudaMemset(A_GPU, 0, PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(C_GPU, 0, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy Memory from GPU to CPU	
	
	// Copy real vector values to GPU
	cudaMemcpy( A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy( B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Run GPU kernel
	dotProductGPU<<<GridSize, BlockSize>>>( A_GPU, B_GPU, C_GPU);
	
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy final GPU answer back to CPU
	cudaMemcpy( &DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
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


