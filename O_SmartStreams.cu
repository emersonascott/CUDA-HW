// Name: Emerson Scott
// Two streams overlapped smartly 
// nvcc O_SmartStreams.cu -o temp
/*
 What to do:
 Read about CUDA stream cooperation.

This code provides most of the setup needed to create two CUDA streams. 
Complete the implementation by replacing all the ???s.

Once the two streams are working, overlap them in a smart way to improve performance.
*/

/*
 Purpose:
 To learn how to use CUDA streams intelligently.
*/

/*
 Explain what you did to fix the code:
 So basically what is happening is first the events, streams, and dimensions are created. Then, memory is reserved on the GPU and data is loaded into A and B on the CPU. Stream 0 copies chunk 1 to the GPU, calculates is, and the results are copied back. The same thing is happening at the same time with stream 1 but they can overlap with the GPU. After that, both streams are being waited on to finish and then the elapsed time is printed and the CUDA stuff is all released.
 
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) //number of elements processed by one stream = 1,048,576
#define ENTIRE_DATA_SET (20*DATA_CHUNKS) //total number of elements in the entire dataset = 20,971,520
#define MAX_RANDOM_NUMBER 1000 //maximum scale for the random input values = 1,000
#define BLOCK_SIZE 256 //number of threads in each GPU block = 256

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers - A and B are the input numbers and C is the output results calculated on the GPU
float *A0_GPU, *B0_GPU, *C0_GPU, *A1_GPU, *B1_GPU, *C1_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
cudaStream_t Stream0, Stream1;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices(); //prepares the GPU and streams
void allocateMemory(); //reserves memory for the arrays
void loadData(); //fills the input arrays
void cleanUp(); //releases memory and destroys CUDA resources
__global__ void trigAdditionGPU(float *, float *, float *, int ); //accepts three pointers to arrays and an integer specifying how many elements to process

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
	cudaEventCreate(&StartEvent); //creates the start event
	cudaErrorCheck(__FILE__, __LINE__); //check for errors
	cudaEventCreate(&StopEvent); //creates the stop event
	cudaErrorCheck(__FILE__, __LINE__); //check for errors
	
	cudaDeviceProp prop; //creates a variable to store information about the GPU
	int whichDevice; //creates an integer to store the selected GPU's device number
	
	cudaGetDevice(&whichDevice); //finds the currently selected GPU and save its number in "whichDevice"
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice); //loads information about that GPU into "prop"
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1) //checks the older CUDA device property that indicates support for overlapping certain data transfers and kernel execution
	{
		printf("\n GPU will not handle overlaps so no speedup from streams"); //if the property is not 1 then the program will print this warning and exit
		printf("\n Good bye.");
		exit(0);
	}
	
	//creates CUDA streams
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__); 
	cudaStreamCreate(&Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE; //256 threads
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0) //checks that the remainder is 0
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE; 
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory() //begins the function that reserves the memory your program needs
{	
	//Allocate Device (GPU) Memory
	//reserves enough GPU memory for each array in each stream
	cudaMalloc(&A0_GPU,DATA_CHUNKS*sizeof(float)); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Allocate page locked Host (CPU) Memory
	//pinned memory is important because CUDA can perform asynchronous transfers between CPU and GPU memory more effectively
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++) //runs once for every element in the dataset
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	//releases GPU memory previously allocated with cudaMalloc()
	cudaFree(A0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(A1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	//releases pinned CPU memory allocated with cudaHostAlloc()
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//destroys the two CUDA events used to measure execution time
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//destroys both CUDA streams and releases their associated resources
	cudaStreamDestroy(Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) //make sure the thread's index is within the array bounds
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent; //declares a variable that will store the elapsed GPU time in milliseconds
	
	setUpCudaDevices(); //creates the events and streams and configures the launch dimensions
	allocateMemory(); //allocates GPU and CPU memory
	loadData(); //fills the CPU input arrays with random values
	
	cudaEventRecord(StartEvent, 0); //records the start event
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS*2) //i starts at 0, the loop continues until i reaches the total dataset size, i increases by 2,097,152 each iteration
	{
		//	******************************************
		cudaMemcpyAsync(A0_GPU, A_CPU + i, //destination address on the GPU, starting address in the CPU array
				DATA_CHUNKS * sizeof(float), //number of bytes to copy
				cudaMemcpyHostToDevice, Stream0); //direction of the copy (CPU to GPU), the stream that will perform the operation
		cudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(B0_GPU, B_CPU + i,
				DATA_CHUNKS * sizeof(float),
				cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(
			A0_GPU, B0_GPU, C0_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy the results back
		cudaMemcpyAsync(C_CPU + i, C0_GPU,
				DATA_CHUNKS * sizeof(float),
				cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy A to the GPU
		cudaMemcpyAsync(A1_GPU, A_CPU + i + DATA_CHUNKS,
				DATA_CHUNKS * sizeof(float),
				cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy B to the GPU
		cudaMemcpyAsync(B1_GPU, B_CPU + i + DATA_CHUNKS,
				DATA_CHUNKS * sizeof(float),
				cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream1>>>(
			A1_GPU, B1_GPU, C1_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy the results back
		cudaMemcpyAsync(C_CPU + i + DATA_CHUNKS, C1_GPU,
				DATA_CHUNKS * sizeof(float),
				cudaMemcpyDeviceToHost, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		//	******************************************
	}
	
	// Make the CPU wait until the streams have finished before it continues.
	cudaStreamSynchronize(Stream0);
	cudaStreamSynchronize(Stream1);
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
