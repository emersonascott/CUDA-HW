// Name: Emerson Scott
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:
 1. Add headers
 2. Change the max iterations and area
 3. Add GPU functions instead of CPU (using things like __global__ and __device__)
 
 
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define MAXITERATIONS 400 // If you have not escaped after this many attempts, we assume you are not going to escape.

// Global variables
unsigned int WindowWidth = 1200;
unsigned int WindowHeight = 1200;

float XMin = -1.0;
float XMax =  1.0;
float YMin = -1.0;
float YMax =  1.0;

// Function prototypes
void cudaErrorCheck(const char*, int); //function called cudaErrorCheck that takes a string and an integer and doesn't return anything
__device__ float3 newtonColor(float, float); //declares a CUDA device function, takes two float values (x and y), float3 is a CUDA type containing three floating-point numbers (red, green, and blue)
__global__ void colorPixels(float*, float, float, float, float, unsigned int, unsigned int); //declares a CUDA kernel
void display(void); //declares the OpenGL display function - GLUT will call this function when it needs to display the image

void cudaErrorCheck(const char *file, int line) //defines the function declared earlier - tells where the error happened
{
	cudaError_t  error; //creates a variable called error
	error = cudaGetLastError(); 

	if(error != cudaSuccess) //checks whether an error occurred
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line); //print error message
		exit(0);
	}
}

__device__ float3 newtonColor(float x, float y) 
{
	float tolerance = 0.0001;
	int maxIterations = MAXITERATIONS;
	int count = 0;
	
	//Newton's method
	for(count = 0; count < maxIterations; count++) //start at 0, continue while count is less than 400, and increase count by 1 each time
	{
		//z = x + yi, this is a complex number
		
		//z^2
		float x2 = x*x - y*y;
		float y2 = 2.0*x*y;
		
		//z^3
		float x3 = x2*x - y2*y;
		float y3 = x2*y + y2*x;
		
		//f(z) = z^3 - 1
		float fx = x3 - 1.0;
		float fy = y3;
		
		//Check if f(z) is close enough to zero
		if(sqrt(fx*fx + fy*fy) < tolerance)
		{
			break;
		}
		
		//f'(z) = 3z^2
		float dfx = 3.0*x2;
		float dfy = 3.0*y2;
		
		//|f'(z)|^2
		float denominator = dfx*dfx + dfy*dfy;
		
		//Prevent division by zero
		if(denominator < 0.0000001)
		{
			break;
		}
		
		//f'(z) / f'(z)	
		float quotientX = (fx*dfx + fy*dfy) / denominator;
		float quotientY = (fy*dfx - fx*dfy) / denominator;
		
		//Newton's method
		//z = z - f(z)/f'(z)
		x = x - quotientX;
		y = y - quotientY;
		
		//Check how close we are to a root
  	        float checkX2 = x*x - y*y;
  	      	float checkY2 = 2.0*x*y;
	
        	float checkX3 = checkX2*x - checkY2*y;
        	float checkY3 = checkX2*y + checkY2*x;
	
        	float checkFx = checkX3 - 1.0;
        	float checkFy = checkY3;

        if(sqrt(checkFx*checkFx + checkFy*checkFy) < tolerance)
        {
            break; //immediately exits the for loop
        }
    }

    // ------------------------------------------------
    // Find which of the three roots we reached
    // ------------------------------------------------
		
	//The three roots of z^3 - 1
	
	//Root 1: approximately (1,0)
	float distance1 = sqrt((x-1.0)*(x-1.0) + y*y);
	
	//Root 2: approximately (-0.5, 0.866025)
	float distance2 = sqrt((x+0.5)*(x+0.5) +
			       (y-0.866025)*(y-0.866025));
			       
	//Root 3: approximately (-0.5, -0.866)
	float distance3 = sqrt((x+0.5)*(x+0.5) +
			       (y+0.866025)*(y+0.866025));
			       
	//Determine which root is closest
	int root;
	
	if(distance1 < distance2 && distance1 < distance3)
	{
		root = 0;
	}
	else if(distance2 < distance1 && distance2 < distance3)
	{
		root = 1;
	}
	else
	{
		root = 2;
	}
			       
	// ------------------------------------------------
	// Create depth based on Newton iterations
	// ------------------------------------------------

	float t = (float)count / (float)maxIterations;

	// Darker areas have a "deeper" appearance.
	// Faster convergence = brighter.
	float depth = 1.0 - t;

	// Keep the darkest areas from becoming completely black.
	float brightness = 0.20 + 0.80 * depth; //creates a brightness value
	
	// ------------------------------------------------
	// Make the boundary glow
	// ------------------------------------------------

	float glow = 1.0 - depth;

	// Make the glow much stronger.
	glow = glow * glow;

	// Add some extra brightness around the boundary.
	brightness += 0.70 * glow;

	if(brightness > 1.0)
	    brightness = 1.0;
	    
        // ------------------------------------------------
	// Create changing colors
	// ------------------------------------------------

	float angle = 6.2831853 * (4.0 * t + root / 3.0);

	float red =
	    0.5 + 0.5 * cosf(angle);

	float green =
	    0.5 + 0.5 * cosf(angle - 2.0943951);

	float blue =
	    0.5 + 0.5 * cosf(angle + 2.0943951);
	    
	// ------------------------------------------------
	// Apply depth and glow to the colors
	// ------------------------------------------------

	red *= brightness;
	green *= brightness;
	blue *= brightness;
	
	// Add a little extra glow to the boundary.
	red += 0.20 * glow;
	green += 0.20 * glow;
	blue += 0.20 * glow;
	
	// Keep RGB values between 0 and 1.
	if(red > 1.0) red = 1.0;
	if(green > 1.0) green = 1.0;
	if(blue > 1.0) blue = 1.0;

	if(red < 0.0) red = 0.0;
	if(green < 0.0) green = 0.0;
	if(blue < 0.0) blue = 0.0;

	return make_float3(red, green, blue);
}

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, unsigned int windowWidth, unsigned int windowHeight) //main GPU function, every CUDA thread handles one pixel
{
	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(pixelX >= windowWidth || pixelY >= windowHeight)
		return;
		
	int id = 3 * (pixelY * windowWidth + pixelX);
	
	float x = xMin + dx * pixelX;
	float y = yMin + dy * pixelY;
	
	float3 color = newtonColor(x, y);
	
	pixels[id]   = color.x;
	pixels[id+1] = color.y;
	pixels[id+2] = color.z;
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float)); //allocates enough CPU memory for every pixel
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__); //checks whether the previous CUDA operation worked
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	blockSize.x = 16; //WindowWidth;
	blockSize.y = 16;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = (WindowWidth + blockSize.x - 1) / blockSize.x;
	gridSize.y = (WindowHeight + blockSize.y - 1) / blockSize.y;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
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


