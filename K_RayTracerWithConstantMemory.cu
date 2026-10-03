// Name: Emerson Scott
// Ray tracing
// nvcc K_RayTracerWithConstantMemory.cu -o temp -lglut -lGL -lm

/*
 What to do:
 This program creates a random set of spheres and uses ray tracing to render an image of them 
 to be displayed on the screen. In the scene, positive X is to the right, positive Y is up, and 
 positive Z comes out of the screen toward the viewer.

 All the spheres are located within a 2x2x2 cube, and you observe them through a 2x2 viewing window.
 
 Your mission, should you choose to accept it:
 1. The spheres created on the CPU do not change, so transfer them to the GPU and store them in constant memory.
 2. Use CUDA events to time your code execution.
*/

/*
 Purpose:
 To learn how to use constant memory and CUDA events.
*/

/*
 Explain what you did to fix the code:
 1. Store the array in constant memory on the GPU
 2. Fix the prototypes
 3. Take away the sphere pointer since it is now stored in the constant memory
 
*/

// Include files
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Defines
#define WINDOWWIDTH 1024 //1024 pixels wide
#define WINDOWHEIGHT 1024 //1024 pixels tall
//total of 1,048,576 pixels
#define XMIN -1.0f
#define XMAX 1.0f
#define YMIN -1.0f
#define YMAX 1.0f
#define ZMIN -1.0f
#define ZMAX 1.0f
#define NUMSPHERES 100 //we have 100 spheres
#define MAXRADIUS 0.2 // The biggest radius a sphere can have.

// Local structures
struct sphereStruct //creates a custom data type called sphereStruct - basically creates a type of variable called a sphere
{
	float r,b,g; // storing sphere color
	float radius; // storing sphere radius
	float x,y,z; // storing sphere center
};

// Globals variables
static int Window; //stores the ID of the OpenGL window
//creates variables containing the window's width and height
unsigned int WindowWidth = WINDOWWIDTH;
unsigned int WindowHeight = WINDOWHEIGHT;
dim3 BlockSize, GridSize;
float *PixelsCPU, *PixelsGPU; //pointers to memory containing the image pixels
sphereStruct *SpheresCPU; //creates a pointer to the sphere date in CPU memory
__constant__ sphereStruct SpheresGPU[NUMSPHERES]; //constant memory for the spheres on the GPU
//constant memory is useful here because the spheres don't change while the kernel is running, and many GPU threads can read the same sphere information
float Zoom = 1.0f;

// Function prototypes
void cudaErrorCheck(const char *, int); //using precognition to tell predict that a cudaErrorCheck function will be called later - much like Bran Stark can in Game of Thrones
void Display();
void idle();
void KeyPressed(unsigned char , int , int);
__device__ float hit(float , float , float *, sphereStruct); //change the prototype to match the function
__global__ void makeSphersBitMap(float *, float); //take away the sphere pointer because it is now stored in constant memory
void makeRandomSpheres();
void makeBitMap();
void paintScreen();
void setup();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line) //defines the error-checking function
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//this function gets called bu GLUT when the window needs to be displayed
void display()
{
	makeBitMap(); //this is what created the image
}

void KeyPressed(unsigned char key, int x, int y) //this function runs whenever the user presses a keyboard key
{	

	if(key == '+')
	{
	    Zoom *= 1.2f;
	    glutPostRedisplay();
	}
	
	if(key == '-')
	{
	    Zoom /= 1.2f;
	    glutPostRedisplay();
	}

	if(key == 'q')
	{
		glutDestroyWindow(Window); //opens the OpenGL window
		
		// Free host memory.
		free(PixelsCPU); //frees CPU memory used by the pixels
		free(SpheresCPU); //frees the CPU memory used by the spheres
	
		// Free divice memory.
		cudaFree(PixelsGPU); //frees GPU memory used by the pixels
		cudaErrorCheck(__FILE__, __LINE__); //checks whether the previous CUDA operation had an error
		
		printf("\nGET OUT\n");
		exit(0);
	}
}

__device__ float hit(float pixelx, float pixely, float *dimingValue, sphereStruct sphere) //determines whether a ray going through a pixel hits a particular sphere
{
	float dx = pixelx - sphere.x;  //Distance from ray to sphere center in x direction
	float dy = pixely - sphere.y;  //Distance from ray to sphere center in y direction
	float r2 = sphere.radius*sphere.radius;
	if(dx*dx + dy*dy < r2) // if the ray hits the sphere, then we need to find distance
	{
		float dz = sqrtf(r2 - dx*dx - dy*dy); // Distance from ray to edge of sphere?
		*dimingValue = dz/sphere.radius; // n is value between 0 and 1 used for darkening points near edge.
		return dz + sphere.z; //  Return the distance to be scaled by
	}
	return (ZMIN- 1.0); //If the ray doesn't hit anything return a number 1 unit behind the box.
}

__global__ void makeSphersBitMap(float *pixels, float zoom) //fix to match new prototype
{
	float halfWidth = (XMAX - XMIN) / (2.0f * zoom);
        float halfHeight = (YMAX - YMIN) / (2.0f * zoom);

 	float currentXMin = -halfWidth;
 	float currentYMin = -halfHeight;

	float stepSizeX = (2.0f * halfWidth) / ((float)WINDOWWIDTH - 1);
  	float stepSizeY = (2.0f * halfHeight) / ((float)WINDOWHEIGHT - 1);
		
	float pixelx = currentXMin + threadIdx.x * stepSizeX;
	float pixely = currentYMin + blockIdx.x * stepSizeY;
	
	// Finding this pixels location in memory
	int id = 3*(threadIdx.x + blockIdx.x*blockDim.x); //fints the pixel's location memory - uses 3 because of R G B
	
	//initialize rgb values for each pixel to zero (black)
	float pixelr = 0.0f;
	float pixelg = 0.0f;
	float pixelb = 0.0f;
	float hitValue; //stores how far away a sphere is
	float dimingValue; //stores the brightness/dimming amount
	float maxHit = ZMIN -1.0f; // Initializing it to be 1 unit behind the box.
	for(int i = 0; i < NUMSPHERES; i++) //loops through all 100 spheres
	{
		hitValue = hit(pixelx, pixely, &dimingValue, SpheresGPU[i]); //change to it using SpheresGPU
		// do we hit any spheres? If so, how close are we to the center? (i.e. n)
		if(maxHit < hitValue) //find the closest sphere and use it's colord
		{
			// Setting the RGB value of the sphere but also diming it as it gets close to the side of the sphere.
			pixelr = SpheresGPU[i].r * dimingValue; 	
			pixelg = SpheresGPU[i].g * dimingValue;	
			pixelb = SpheresGPU[i].b * dimingValue; 	
			maxHit = hitValue; // reset maxHit value to be the current closest sphere
		}
	}
	
	pixels[id] = pixelr; //stores red
	pixels[id+1] = pixelg; //stores green
	pixels[id+2] = pixelb; //stores blue
}

void makeRandomSpheres() //function creates all 100 random spheres
{	
	float rangeX = XMAX - XMIN;
	float rangeY = YMAX - YMIN;
	float rangeZ = ZMAX - ZMIN;
	
	for(int i = 0; i < NUMSPHERES; i++)
	{
		SpheresCPU[i].x = (rangeX*(float)rand()/RAND_MAX) + XMIN;
		SpheresCPU[i].y = (rangeY*(float)rand()/RAND_MAX) + YMIN;
		SpheresCPU[i].z = (rangeZ*(float)rand()/RAND_MAX) + ZMIN;
		SpheresCPU[i].r = (float)rand()/RAND_MAX;
		SpheresCPU[i].g = (float)rand()/RAND_MAX;
		SpheresCPU[i].b = (float)rand()/RAND_MAX;
		SpheresCPU[i].radius = MAXRADIUS*(float)rand()/RAND_MAX;
	}
}	

void makeBitMap()
{	
	cudaEvent_t start, stop;
	float milliseconds = 0;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	cudaMemcpyToSymbol(SpheresGPU, SpheresCPU, NUMSPHERES*sizeof(sphereStruct)); //copy spheres to constant memory
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventRecord(start);
	
	makeSphersBitMap<<<GridSize, BlockSize>>>(PixelsGPU, Zoom);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&milliseconds, start, stop); //calculates the time between the start and stop and stores it in milliseconds
	
	printf("GPU kernel time: %f ms\n", milliseconds);
	
	//delets the CUDA events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaMemcpy(PixelsCPU, PixelsGPU, WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	paintScreen(); //displays the image
}

void paintScreen() //function for displaying the image
{
	//Putting pixels on the screen.
	glDrawPixels(WINDOWWIDTH, WINDOWHEIGHT, GL_RGB, GL_FLOAT, PixelsCPU); 
	glFlush();
}

void setup()
{
	//Allocating memory for the scene that will be displayed to the screen.
	//We need the 3 because each pixel has a red, green, and blue value.
	PixelsCPU = (float *)malloc(WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float)); //allocates CPU memory for the pixels
	cudaMalloc(&PixelsGPU,WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float)); //allocated the same amount of memory on the GPU
	cudaErrorCheck(__FILE__, __LINE__); //checks whether the GPU memory allocation worked
	
	//Allocating memory for the spheres that will create the scene.
	//This is what you will be changing out for constant memory.
	SpheresCPU = (sphereStruct*)malloc(NUMSPHERES*sizeof(sphereStruct)); //allocates CPU memory for the 100 spheres
	
	//Threads in a block
	if(WINDOWWIDTH > 1024) //To keep the code simple we make sure the scene width fits in a block.
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	BlockSize.x = WINDOWWIDTH; //sets the number of threads in the X direction
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Blocks in a grid
	GridSize.x = WINDOWHEIGHT; //creates 1024 blocks - each block represents one row of pixels
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Seeding the random number generator.
	time_t t; //creates a variable for storing the current time
	srand((unsigned) time(&t)); //uses the current time as the seed for the random-number generator
}

int main(int argc, char** argv)
{ 
	setup();
	makeRandomSpheres(); //creates the 100 random spheres
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WINDOWWIDTH, WINDOWHEIGHT); //creates a 1024 x 1024 window
	Window = glutCreateWindow("Random Spheres");
	glutKeyboardFunc(KeyPressed);
   	glutDisplayFunc(display);
   	glutMainLoop();
}

