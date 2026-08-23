// Name: Emerson Scott
// Vector addition on the CPU, with timer and error checking
// To compile: nvcc A_VectorAddCPU.cu -o temp
/*
 What to do:
 1. Understand every line of the code and be able to explain it in class.
 2. I have intentionally broken the code in several places — find and fix them.
 3. Compile, run, and experiment with the code.
 4. Also explore the Pointerstest.cu code to understand how pointers work.
*/

/*
 Purpose:
 To fully understand how vector addition works on the CPU, so you can compare
 it to GPU-based vector addition in the next assignment.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h> //Add header because we use "malloc", "free", and "abs"
#include <math.h> //Add header because we use "fabs"

// Defines
#define N 1000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; 
float Tolerance = 0.01; //Change from "0.0" to "0.01" because "percentError < Tolerance" so it needs to be greater than 0 because "0 cannot be less than 0"

// Function prototypes
void allocateMemory();
void initialize();
void addVectorsCPU(float*, float*, float*, int);
bool check(float*, int, float); //Add "float" to match the parameters of line 68
long elapsedTime(struct timeval, struct timeval);
void cleanUp();

//Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));//B_CPU needs memory allocated
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id]; //Change from multiplying to adding because we are using "addVectors"
	}
}

//Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerance)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; //Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = fabs((myAnswer - trueAnswer)/trueAnswer)*100.0; //Change "abs" to "fabs" because that is the function we should use for a floating-point value
	
	if(percentError < tolerance) //Need lowercase because we passed the global function and are pulling it from the "check" 
	{
		return(true); //Change "totally" to "true" because that is what we use
	}
	else 
	{
		return(false);
	}
}

// Calculating elapsed time.
long elapsedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elapsed in microseconds
	return endTime - startTime; //Add "- startTime" to calculate elapsed time and not just return the end time
}

//Cleaning up memory after we are finished.
void cleanUp() //Change to lowercase because that is what is defined
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
}

int main()
{
	timeval start, end;
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	initialize();

	// Starting the timer.	
	gettimeofday(&start, NULL);

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elapsedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	cleanUp(); //Change from "CleanUp" to "cleanUp" because we need to match whatever is defined
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}

