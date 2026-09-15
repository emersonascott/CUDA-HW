// Name: Emerson Scott
// Vector Dot product on 1 block 
// nvcc H_DotProductGPU1Block.cu -o temp
/*
 What to do:
 This code uses the CPU to compute the dot product of two vectors of length N.
 It also includes a skeleton for setting up a GPU dot product, though that part is currently empty.

 Notes:
 - The CPU code is intentionally a bit convoluted to mirror the structure of the GPU version you will write.
 - The program will verify whether your GPU implementation of the dot product is correct.

 Instructions:
 - Leave the block and vector sizes as:
     Block = 1000
     N     = 823
 - Use **folding at the block level** for the addition (reduction) step.
 - Do **not** use shared memory — that will be introduced in the next assignment.
 - Do **not** use bit shifting — keep the code readable for someone without a CS background.
 - Keep it simple enough that you could explain it to a 5th grader!
*/

/*
 Purpose:
 To learn how to get threads to cooperate using __syncthreads().
*/

/*
 Explain what you did to fix the code:
 1. Add needed headers
 2. Change the functions to work on the GPU kernels rather than CPU
 3. Make sure not to use the unneeded threads
 4. multiply a * b
 5. synchronize all of the threads so that it does not keep going without some of them and mess everything up
 6. fold and repeat
 
*/

// Include files
#include <sys/time.h> //for measuring time functions
#include <stdio.h> //for standard input/output functions
#include <stdlib.h> //for malloc, free, exit
#include <cuda_runtime.h> //for CUDA functionality
#include <math.h> //for fabs

// Defines
#define N 823 // Length of the vector - creates a constant called N of 823 elements (0-822)

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers - pointers to floating-point numbers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers - pointers to memory on the GPU
float DotCPU, DotGPU; //these hold the final answers and get compared at the end
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01; //determines how much difference will be allowed between the CPU and GPU answers

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int); //this means that the function runs on the GPU and is called from the CPU
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line) //checks whether something went wrong in CUDA
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line); //tells us the file and line that caused the error
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices() //function determines the layout of our GPU computation
{
	BlockSize.x = 1000; //threads 0 through 822 have vector elements and threads 823 through 999 don't have an element
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float)); //reserves enough CPU memory for 823 floats
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float)); //gives cudaMalloc the address of our GPU pointer so it can put the new GPU memory address there
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float)); //allocates GPU memory for the other arrays
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float)); //allocates GPU memory for the other arrays
	cudaErrorCheck(__FILE__, __LINE__);
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
void dotProductCPU(float *a, float *b, float *C_CPU, int n) //CPU function call
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
__global__ void dotProductGPU(float *a, float *b, float *C_GPU, int n) //GPU function call - runs this function on the GPU
{
	int id = threadIdx.x; //tells us which thread we're currently running because each thread gets a unique number - this creates an integer called id and gives it the thread's ID number
	
	if(id < n)
	{
		C_GPU[id] = a[id] * b[id];
	}
	
	for(int step = 1; step < n; step = step * 2) //doubling the distance between values being added
	{
		__syncthreads(); //synchronize the threads so they cannot keep going without each other
		
		if(id % (2 * step) == 0 && id + step < n) //selects which threads are being used and making fewer and fewer threads participate each round - the last part prevents them from going outside of the vector
		{
			C_GPU[id] += C_GPU[id + step];
		}
	}
}

// Checking to see if anything went wrong in the vector addition.
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
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	DotGPU = C_CPU[0]; // C_GPU was copied into C_CPU.
	
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


