// Name: Emerson Scott
// GPU random walk. 
// nvcc 'P_GPURandomWalk .cu' -o temp
/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:
- generate random directions on the GPU
- run 20 GPU threads where each one performs one random walk
- share the results between the GPU and CPU without manually copying them
- print the final position of all 20 walks
ORDER OF OPERATIONS:
1. allocate unified memory
2. create the seed
3. launch the GPU kernel
4. wait for the GPU to finish
5. print all 20 final positions
6. free the memory

 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Defines
#define NUMBER_OF_WALKS 20
#define NUMBER_OF_RANDOM_STEPS 10000

__global__ void randomWalk(int *finalX, int *finalY, unsigned long long seed)
//function running on the GPU and is called from the CPU(stores the final horizontal position of every walk, stores the final vertical position of every walk, the starting value used to initialize the random-number generators)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; //calculating which walk a GPU thread is responsible for
	
	if(id < NUMBER_OF_WALKS)
	{
		curandState state; //creates a state object that tracks the random-number generator's progress
		curand_init(seed + id, 0, 0, &state); //initialize the generator(give each walk a different seed, the zeros specify the sequence number and starting offset, tell cuRAND where to store the generator's state)
		
		//every walk begins at (0,0)
		int positionX = 0;
		int positionY = 0;
		
		for(int i = 0; i < NUMBER_OF_RANDOM_STEPS; i++) //repeat the loop 10,000 time for each walk
		{
			if(curand(&state) % 2 == 0)
				positionX--; //if the remainder is 0, the walk moves left by one
			else
				positionX++; //if the remainder is not 0, the walk moves right by one
				
			if(curand(&state) % 2 == 0)
				positionY--; //if the remainder is 0, the walk moves down by one
			else
				positionY++; //if the remainder is not 0, the walk moves up by one
		}
		
		//each thread saves its final coordinates into its own position in the arrays
		finalX[id] = positionX;
		finalY[id] = positionY;
	}
}

// Function prototypes
int main(int, char**);

int main(int argc, char** argv)
{
	int *finalX;
	int *finalY;
	
	//allocate memory that both the CPU and GPU can access(set the pointer to the allocated memory, 20 * 4 = 80 bytes)
	cudaMallocManaged(&finalX, NUMBER_OF_WALKS * sizeof(int));
	cudaMallocManaged(&finalY, NUMBER_OF_WALKS * sizeof(int));
	
	unsigned long long seed = (unsigned long long)time(NULL); //create a seed based on the current time
	
	randomWalk<<<1, NUMBER_OF_WALKS>>>(finalX, finalY, seed);
	//<<<number of blocks, threads per block>>>
	
	cudaDeviceSynchronize(); //wait for all GPU calculations to finish
	
	printf("\n Final position of the 20 random walks:\n");
	
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("Walk %2d: (%d, %d)\n", i + 1, finalX[i], finalY[i]);
	}
	
	cudaFree(finalX);
	cudaFree(finalY);
	
	return 0;
}
