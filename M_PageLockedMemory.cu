// Name: Emerson Scott
// Page-locked memory test
// nvcc M_PageLockedMemory.cu -o temp

/*
 What to do:
 Read about **page-locked (pinned) memory**. Fill in the ???s in this code to understand how to
 set up and test page-locked memory on the host.
*/

/*
 Purpose:
 To learn how page-locked (pinned) memory works and how to use it effectively.
*/

/*
 Explain what you did to fix the code:
 1. Allocate page-locked CPU memory
 2. Free CUDA host
 
 Three memory buffers: GPU, pageable CPU, and pinned CPU
 Down: GPU to CPU
 Up: CPU to GPU
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Defines
#define SIZE 2000000 //each array holds space for 2,000,000 floating-point numbers
#define NUMBER_OF_COPIES 1000 //each test repeats its memory copy 1,000 times to provide a longer total time that is easier to compare

//Globals
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
cudaEvent_t StartEvent, StopEvent;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void cleanUp();
void copyPageableMemoryUp();
void copyPageLockedMemoryUp();
void copyPageableMemoryDown();
void copyPageLockedMemoryDown();

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&NumbersOnGPU, SIZE*sizeof(float)); //memory allocation function for device memory - gives CUDA the address of the pointer so it can store the new GPU memory address there - SIZE is 2,000,000 * 4 bytes = 8,000,000
	cudaErrorCheck(__FILE__, __LINE__); //check whether the allocation succeeded

	//Allocate pageable Host (CPU) Memory
	PageableNumbersOnCPU = (float*)malloc(SIZE*sizeof(float)); //malloc requests a block of memory, SIZE * sizeof determines how many bytes are requested, float* converts the returned pointer to a pointer to floats in C++ - the address of the memory is saved in PageableNumbersOnCPU
	//ordinary allocated memory can be paged out by the operating system
	
	//Allocate page locked Host (CPU) Memory
	cudaMallocHost(&PageLockedNumbersOnCPU, SIZE * sizeof(float)); //allocates CPU memory and page-locks it so the operating system will not page it out while it remains pinned so that CUDA has access to host memory for transfers more efficiently and is necessary for certain asynchronous transfer operations
	cudaErrorCheck(__FILE__, __LINE__);
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(NumbersOnGPU); //releases the GPU memory previously allocated by cudaMalloc()
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(PageLockedNumbersOnCPU); //Releases the pinned CPU memory allocated by cudaMallocHost() and removes its pinned status
	cudaErrorCheck(__FILE__, __LINE__);
	
	free(PageableNumbersOnCPU); //releases the ordinary CPU memory allocated with malloc()
	
	//release the resources associated with CUDA events
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

void copyPageableMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++) //start counting at 0 and repeat while less than 1000 increasing by 1
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice); //copy data between memory locations(the destination of where the data is going, the source where the data comes from, the number of bytes to copy, tell CUDA the transfer direction is CPU to GPU)
		cudaErrorCheck(__FILE__, __LINE__); //checks for a CUDA runtime error after the copy
	}
}

void copyPageableMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost); //copy data between memory locations(the destination of where the data is going, the source where the data comes from, the number of bytes to copy, tell CUDA the transfer direction is GPU to CPU)
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageLockedNumbersOnCPU, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryDown() //pinned version
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageLockedNumbersOnCPU, NumbersOnGPU, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

int main()
{
	float timeEvent; //stores the elapsed time reported by CUDA events
	
	setUpCudaDevices(); //creates the start and stop events
	allocateMemory(); //allocates the GPU memory, pageable CPU memory, and pinned CPU memory
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory down = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory down = %3.1f milliseconds", timeEvent);
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
