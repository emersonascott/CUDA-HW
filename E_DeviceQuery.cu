// Name: Emerson Scott
// Device query
// nvcc E_DeviceQuery.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 1. Add headers
 2. Add new functions to learn more about my GPU
 
*/

// Include files
#include <stdio.h>
#include <stdlib.h> //use functions like "exit"
#include <cuda_runtime.h> //for all of the CUDA funtions

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop; //this creates a structure that will hold information about the GPU

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i); //fills structure with information about GPUi
		cudaErrorCheck(__FILE__, __LINE__); //checks whether the previous CUDA command caused an error
		printf("\n");
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); //prints the GPU's name
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); //prints the CUDA compute capability
		printf("Clock rate: %d\n", prop.clockRate); //Shows the GPU's clock frequency (typically in kHz
		printf("Device copy overlap: "); //checks whether the GPU supports overlapping certain memory transfers with kernel execution
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n"); //checks whether or not the GPU has a timeout for kernel execution
		else printf("Disabled\n");
		printf("\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem); //tells how much global memory the GPU has
		printf("Total constant Mem: %ld\n", prop.totalConstMem); //tells how much constant memory the GPU has
		printf("Max mem pitch: %ld\n", prop.memPitch); //gives the maximum memory pitch supported by the device - pitch is related to how CUDA organizes memory for certain 2D/3D memory allocations
		printf("Texture Alignment: %ld\n", prop.textureAlignment); //tells the required memory alignment for texture references - this is for things like image processing and data with spatial locality
		printf("\n");
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //prints the number of Streaming Multiprocessors (SMs) - an SM is essentially one of the GPU's major processing units
		printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock); //change the per "mp" to "block" to correctly label what we are looking for and printing - this tells how much shared memory a CUDA block can use
		printf("Registers per block: %d\n", prop.regsPerBlock); //change the per "mp" to "block" to correctly label what we are looking for and printing - shows how many registers are available per CUDA block
		printf("Threads in warp: %d\n", prop.warpSize); //tells how many threads make up a warp - a warp is a group of threads that the GPU schedules/executes together
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); //tells the maximum number of threads that can be places in one CUDA block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //tells the maximum dimensions of a block
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //tells the maximum dimensions of a grid - the grid contains all the blocks launched by the kernel
		printf("\n");
		
		//added lines ahead
		printf(" ---CUDA Capability Information for device %d ---\n", i);
		printf("Concurrent kernels: "); //finds out if the GPU can have multiple kernels running simultaneously (a kernel is a function that runs on the GPU)
			if (prop.concurrentKernels)
				printf("Support\n");
			else
				printf("Not supported\n");
		printf("ECC enabled: ");
			if (prop.ECCEnabled) //Sees if the GPU has error-correcting code memory enabled
				printf("Yes\n");
			else
				printf("No\n");
		printf("Integrated GPU: "); //This tells us whether the GPU shares resources with the CPU/system (meaning integrated), or if it has a separate graphics processor
			if (prop.integrated)
				printf("Yes\n");
			else
				printf("No\n");
		printf("Managed memory: "); //This tells us whether the GPU supports CUDA managed memory. Managed memory makes it easier for CUDA programs to work with memory that can be accessed by both the CPU and GPU
			if (prop.managedMemory)
				printf("Supported\n");
			else
				printf("Not supported\n");
		printf("Can map host memory: "); //This tells us whether the GPU can map CPU host memory into the GPU's address space. With mapped host memory, CUDA can allow the GPU to access certain CPU memory directly
			if (prop.canMapHostMemory)
				printf("Yes\n");
			else
				printf("No\n");
		printf("Unified addressing: "); //Unified Virtual Addressing (UVA) allows the CPU and GPU to use a unified virtual address space. CUDA can use a common addressing system to help manage memory between the CPU and GPU
			if (prop.unifiedAddressing)
				printf("Supported\n");
			else
				printf("Not supported\n");
		
	}	
	return(0);
}

