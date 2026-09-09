// Name: Emerson Scott
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 1. Add CUDA headers
 2. Add the __device__ to the prototype and function
 3. Add the __global__, point to the pixel array, define the dimensions of Julia, assign a color to each pixel
 4. Move the pixel calculations from the CPU to the GPU. The GPU kernel handles each pixel so the while loops were removed for that. There is still a while loop needed inside the "escapeOrNotColor" to perform the Julia calculation
 5. Calculate the total number of pixels (1,048,576)
 6. Calculate the X and Y coordinates
 7. Allocate CPU and GPU memory
 8. Check for CUDA errors
 9. Copy the GPU result back to the CPU
 10. Free the memory
 
*/

// Include files
#include <stdio.h> //needed for printf
#include <GL/glut.h> //includes GLUT, which is used to create the OpenGL window
#include <stdlib.h> //needed for malloc, free, exit
#include <math.h> //needed for sqrt
#include <cuda_runtime.h> //needed for all the CUDA functions

// Defines
#define MAXMAG 10.0 // Creates a constant - If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape. - prevents the program from getting stuck calculating a point forever
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C
//together they represent the complex number: C = -0.824 - 0.1711i

// Global variables that specify the size of the OpenGL window
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

//defines the portion of Julia that is being displayed - creates a 4x4 region
float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int); //takes a string and an integer and returns nothing
__device__ float escapeOrNotColor(float, float); //add the __device__ prototype to become the GPU version - this means that the function runs on the GPU and can be called by GPU code

__global__ void juliaGPU(float *pixels, int width, int height, //pointer to the pixel array (every pixel has 3 values: red, green, blue), the width and height tells the GPU what they are (1024)
			 float xmin, float xmax, //area of the Julia set being displayed (-2,2)
			 float ymin, float ymax) //area of the Julia set being displayed (-2,2)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //every CUDA thread gets its own IDs - this formula turns the block/thread IDs into one global index
	
	int totalPixels = width * height; //number of pixels = 1024*1024 = 1,048,576 pixels
	
	if(index < totalPixels) //makes sure it doesn't go past the array so a CUDA thread doesn't try to access a pixel that doesn't exist
	{
		int pixelX = index % width; //% operator gives the remainder
		int pixelY = index / width; //integer division gives the row
		
		float stepSizeX = (xmax - xmin) / (float)width; //tells how far apart the points are in a new GPU kernel code
		float stepSizeY = (ymax - ymin) / (float)height; //tells how far apart the points are in a new GPU kernel code
		
		float x = xmin + pixelX * stepSizeX; //converts the pixel position horizontally into a coordinate
		float y = ymin + pixelY * stepSizeY; //converts the pixel position vertically into a coordinate
		
		float value = escapeOrNotColor(x, y); //calculation now happens on the GPU - this calls the device function
		
		pixels[index * 3] = 0.0; //store the red value
		//store the green and blue values
		pixels[index * 3 + 1] = value * (128.0 / 255.0);
		pixels[index * 3 + 2] = value * (128.0 / 255.0);
	}
}

void cudaErrorCheck(const char *file, int line) //this function checks whether CUDA reported an error
{
	cudaError_t  error; //creates a CUDA error variable
	error = cudaGetLastError(); //gets the most recent CUDA error

	if(error != cudaSuccess) //checks whether an error occured (cudaSuccess means everything is okay)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line); //prints information about the error
		exit(0); //stops the program if there is a CUDA error
	}
}

__device__ float escapeOrNotColor (float x, float y) //add to the function to match the GPU prototype - this runs the function on the GPU
{
	float mag,tempX; //magnitude, temporary copy of X
	int count; //number of iterations
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y); //calculates the magnitude
	while (mag < maxMag && count < maxCount) //keeps calculating the equation while the point hasn't escaped and it hasn't reached 200 iterations
	{	
		tempX = x; //We will be changing the x but we need its old value to find y. - this saves the old X value
		x = x*x - y*y + A; //calculates the new X value
		y = (2.0 * tempX * y) + B; //calculates the new Y value using the old X stored in tempX
		mag = sqrt(x*x + y*y); //recalculates the magnitude after the new X and Y values
		count++; //adds one to the iteration counter
	}
	if(count < maxCount) 
	{
		return(0.0); //escaped
	}
	else
	{
		return(1.0); //did not escape
	}
}

void display(void) //this is the OpenGL function that creates and displays the image
{ 
	float *pixels; //allocating CPU memory 
	float *pixelsGPU; //allocating GPU memory
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float)); //allocate CPU memory
	
	cudaMalloc((void**)&pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float)); //allocate GPU memory

	int totalPixels = WindowWidth * WindowHeight;

	int blockSize = 256;
	int gridSize = (totalPixels + blockSize - 1) / blockSize; //equation we figured out the other day
	
	juliaGPU<<<gridSize, blockSize>>>(pixelsGPU, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax); //launches the GPU kernel
	//gridSize = number of blocks = 4096
	//blockSize = threads per block = 256
	
	//check for CUDA errors
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize(); //waits for GPU to finish
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy(pixels, pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost); //this copies the result back to the CPU (GPU -> CPU)(pixelsGPU -> pixels)
	
	cudaErrorCheck(__FILE__, __LINE__);
	
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); //draws pixels using OpenGL
	
	glFlush();
	
	cudaFree(pixelsGPU); //prevents the GPU memory from being leaked every time the display runs
	
	free(pixels); //frees the CPU memory
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

