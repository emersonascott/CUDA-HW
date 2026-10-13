// Name: Emerson Scott
// Setting up a stream
// nvcc N_Streams.cu -o temp

/*
 What to do:
 Read about CUDA streams. Look at all the ???s in the code and remove the ???s that need to be removed so the code will run.
*/

/*
 Purpose:
 To learn how to setup and work with CUDA streams.
*/

/*
 Explain what you did to fix the code:
 1. Fill out main with MemCpy's
 2. Understand what each line means
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Defines
#define DATA_CHUNKS (1024*1024) //creates a named constant through text substitution before compilation - DATA_CHUNKS = 1,048,576
#define ENTIRE_DATA_SET (20*DATA_CHUNKS) //the total number of numbers is 20 chunks = 20,971,520
#define MAX_RANDOM_NUMBER 1000 //sets the upper scaling value used when generating random numbers - generates input values from approximately 0 to 1000
#define BLOCK_SIZE 256 //each CUDA block will have 256 threads
//there are 4,096 blocks

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
// ??? Notice that we have to define a stream
cudaStream_t Stream0; //declares a variable representing a CUDA stream
//a stream is like a queue of GPU operations - operations in the same stream execute in order

//Function prototypes
void cudaErrorCheck(const char*, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int ); //three float* arguments represent the input arrays and output array and the final int tells the kernel how many elements it should process

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent); //creates a CUDA event for measuring when the GPU work starts
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent); //creates the second event which marks the end of the times work
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop; //creates a variable named prop that can hold information about the GPU like it's capabilities and memory limits
	int whichDevice; //creates an integer to store the selected GPU's ID
	
	cudaGetDevice(&whichDevice); //asks CUDA which GPU is currently selected and checks for errors
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice); //gets the properties of that GPU and stores them in prop
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1) //checks whether the GPU reports support for overlapping certain data transfers and kernel execution
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	// ??? Notice that we have to create the stream
	cudaStreamCreate(&Stream0); //creates the stream that the program will use to schedule its memory copies and kernel launches
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE; //256 threads
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0) //make sure the chunks divide evenly - remainder needs to equal 0
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE; //4096 blocks
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,DATA_CHUNKS*sizeof(float)); //reserves enough GPU memory for one chunk of array A = 4,194,304 bytes
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,DATA_CHUNKS*sizeof(float)); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//??? Notice that we are using host page locked memory
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault); //reserves CPU memory for the entire A array using CUDA's host-memory allocation function
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData() //loading the input data
{
	time_t t; //creates a variable named t that can store a time value
	srand((unsigned) time(&t)); //sets the starting point, called the seed, for the random-number generator - generally uses the current time
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++) //loop that repeats once for every element in the arrays - starts at 0, keeps looping while the index is less than the total number of elements, increases the index by one each time
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX; //creates a random number and stores it in element i of array A
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A_GPU); //frees the GPU memory allocated for array A
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to free this memory with cudaFreeHost
	cudaFreeHost(A_CPU); //frees the page-locked CPU memory allocated for A
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent); //releases the two CUDA events that were created for timing
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to kill the stream.
	cudaStreamDestroy(Stream0); //destroys the CUDA stream and releases its associated resources
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x; //each thread gets a unique index
	
	if(id < n) //check that the thread's index is less than the number of elements
	{
		c[id] = sin(a[id]) + cos(b[id]); //calculates the result for one element
	}
}

int main()
{
	float timeEvent; //creates a floating-point variable to store the elapsed time in milliseconds
	
	//configure the CUDA device, stream, and events, and then allocate CPU and GPU memory
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS) //the loop repeats the GPU calculations until all the data has been processed - increases i by 1,048,576 after every loop iteration
	{
		cudaMemcpyAsync(A_GPU, A_CPU + i, DATA_CHUNKS * sizeof(float),
				cudaMemcpyHostToDevice, Stream0); //copy array A from the CPU to GPU
		cudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(B_GPU, B_CPU + i, DATA_CHUNKS * sizeof(float),
		                cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);

		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(
		    A_GPU, B_GPU, C_GPU, DATA_CHUNKS); //launches the GPU function that calculates the answers
		cudaErrorCheck(__FILE__, __LINE__);

		cudaMemcpyAsync(C_CPU + i, C_GPU, DATA_CHUNKS * sizeof(float),
		                cudaMemcpyDeviceToHost, Stream0); //copies the calculated results back to CPU memory
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	// ??? Notice that we have to make the CPU wait until the GPU has finished stream0
	cudaStreamSynchronize(Stream0); //tells the CPU to wait until all previously queued work in Stream0 is complete
	
	cudaEventRecord(StopEvent, 0); //records the event used to mark the end of the timed work
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent); //waits until the stop event has actually completed before calculating the elapsed time
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent); //calculates the time between the two CUDA events and stores it in timeEvent in milliseconds
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent); //prints the elapsed time in the terminal
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp(); //frees the allocated memory and destroys the CUDA events and stream
	
	return(0);
}
