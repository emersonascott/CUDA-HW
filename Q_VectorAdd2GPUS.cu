// Name: Emerson Scott
// Vector addition on two GPUs.
// nvcc Q_VectorAdd2GPUS.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 0:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 1:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

/*
 Purpose:
  To learn how to use more than one GPU.
*/

/*
 Explain what you did to fix the code:
 1. Add pointers for two GPUs instead of one
 2. Split the elements in to the two GPUs "evenly" assigning the first half to GPU0 and the second half to GPU1
 3. Select the GPU to use
 4. Free all of the memory on both GPUs
 
 Basically keep most things the same just do it twice and assign it to its GPU!
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <math.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU0, *B_GPU0, *C_GPU0; //GPU pointers for GPU0
float *A_GPU1, *B_GPU1, *C_GPU1; //GPU pointers for GPU1
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int); 
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float*, float*, float*, int); //kernel takes pointers so the arguments must be pointers
bool check(float*, int, float); 
long elapsedTime(struct timeval, struct timeval);
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

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int i = 0; i < n; i++)
	{ 
		c[i] = a[i] + b[i];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerance)
{
	for(int id = 0; id < n; id++)
	{
		float trueAnswer = 3.0f * id;
		
		if(fabsf(c[id] - trueAnswer) > tolerance)
		{
			printf("\nIncorrect result at index %d\n", id);
			return false;
		}
	}
	return true;
}

// Calculating elapsed time.
long elapsedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elapsed in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	//free GPU0 memory
	cudaSetDevice(0);
	cudaFree(A_GPU0);
	cudaFree(B_GPU0);
	cudaFree(C_GPU0);
	
	//Free GPU1 memory
	cudaSetDevice(1);
	cudaFree(A_GPU1);
	cudaFree(B_GPU1);
	cudaFree(C_GPU1);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	int numberOfGPUs = 0; //check that two GPUs are available
	
	cudaError_t error = cudaGetDeviceCount(&numberOfGPUs);
	
	if(error != cudaSuccess || numberOfGPUs < 2)
	{
		printf("\nError: At least two GPUs are required.\n");
		return 1;
	}
	
	//split the vector between the GPUs
	int N0 = (N + 1) / 2; //GPU0 gets 5,752 elements
	int N1 = N - N0; //GPU1 gets 5,751 elements
	
	printf("\nGPU 0 will process %d elements.", N0);
	printf("\nGPU 1 will process %d elements.\n", N1);
	
	//set up block and grid sizes
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	dim3 GridSize0((N0 - 1) / BlockSize.x + 1, 1, 1);
	dim3 GridSize1((N1 - 1) / BlockSize.x + 1, 1, 1);
	
	//Allocate CPU memory
	A_CPU = (float*)malloc(N * sizeof(float));
	B_CPU = (float*)malloc(N * sizeof(float));
	C_CPU = (float*)malloc(N * sizeof(float));
	
	if(A_CPU == NULL || B_CPU == NULL || C_CPU == NULL)
	{
		printf("\nCPU memory allocation failed.\n");
		free(A_CPU);
		free(B_CPU);
		free(C_CPU);
		return 1;
	}
	
	cudaSetDevice(0); //allocate memory on GPU 0
	
	cudaMalloc((void**)&A_GPU0, N0 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMalloc((void**)&B_GPU0, N0 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMalloc((void**)&C_GPU0, N0 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1); //allocate memory on GPU 1
	
	cudaMalloc((void**)&A_GPU1, N1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMalloc((void**)&B_GPU1, N1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMalloc((void**)&C_GPU1, N1 * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elapsedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0f;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	//copy the first half to GPU 0
	cudaSetDevice(0); //this tells CUDA that the next GPU operation should use GPU0
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU0, A_CPU, N0*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU0, B_CPU, N0*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1); //this tells CUDA that the next GPU operation should use GPU1
	
	cudaMemcpyAsync(A_GPU1, A_CPU + N0, N1*sizeof(float), cudaMemcpyHostToDevice); //starts at element 5752 and puts it on GPU1
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU1, B_CPU + N0, N1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(0); //add the first half on GPU 0
	
	addVectorsGPU<<<GridSize0,BlockSize>>>(A_GPU0, B_GPU0 ,C_GPU0, N0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	
	addVectorsGPU<<<GridSize1,BlockSize>>>(A_GPU1, B_GPU1 ,C_GPU1, N1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(0); //copy GPU 0 results to the first part of C_CPU
	
	cudaMemcpy(C_CPU, C_GPU0, N0 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1); //copy GPU 0 results to the first part of C_CPU
	
	cudaMemcpy(C_CPU + N0, C_GPU1, N1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	
	timeGPU = elapsedTime(start, end);
	
	//check the results
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\nSomething went wrong in the GPU vector addition.\n");
	}
	else
	{
		printf("\nYou added the two vectors correctly on both GPUs!");
		printf("\nThe time it took on the CPU was %ld microseconds", timeCPU);
		printf("\nThe time it took on the two GPUs was %ld microseconds", timeGPU);
	}
	
	//free the memory
	cleanUp();
	
	printf("\n\n");
	
	return 0;
}
