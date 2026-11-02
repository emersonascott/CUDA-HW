// Name: Emerson Scott
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

/*
 Explain what you did to fix the code:
 1. define the number of spheres
 2. change the 1's and 2's to i's and j's because we are no longer working with just 2 bodies, but N bodies so we need to create lots of for loops
 3. replace the individual parameters to arrays
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define NUMBER_OF_SPHERES 10 //define the number of spheres
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0 //how long the simulation loop runs according to its simulation-time variable
#define DT        0.0001 //size of each simulation time step
#define GRAVITY 0.1 //strength of the gravity used in the force calculation
#define DIAMETER 1.0 //diameter of each sphere
#define SPHERE_PUSH_BACK_STRENGTH 50.0 //strength of the collision push-back calculation
#define PUSH_BACK_REDUCTION 0.1 //reduces push-back force in one collision situation
#define DAMP 0.01 //reduces velocity over time
#define DRAW 100 //draw a new frame after every 100 simulation steps
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0 //range for random starting velocities

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
//boundaries of the box are from -3 to +3 along each axis
//const means the value won't change while the program is running

//position of each sphere
float px[NUMBER_OF_SPHERES];
float py[NUMBER_OF_SPHERES];
float pz[NUMBER_OF_SPHERES];

//velocity of each sphere
float vx[NUMBER_OF_SPHERES];
float vy[NUMBER_OF_SPHERES];
float vz[NUMBER_OF_SPHERES];

//force of each sphere
float fx[NUMBER_OF_SPHERES];
float fy[NUMBER_OF_SPHERES];
float fz[NUMBER_OF_SPHERES];

//mass of each sphere
float mass[NUMBER_OF_SPHERES];

// Function prototypes
void set_initial_conditions(); //sets the starting positions and velocities
void Drawwirebox(); //draws the box
void draw_picture(); //draws all the spheres
void keep_in_box(); //keeps the spheres inside the boundaries
void get_forces(); //calculates the forces on the spheres
void move_bodies(float); //moves the spheres using the time value
void nbody(); //runs the simulation
void Display(void); //function called by the graphics system to display the scene
void reshape(int, int); //responds when the window changes size
int main(int, char**); //program's starting function

void set_initial_conditions()
{ 
	srand((unsigned) time(NULL)); //seed the random-number generator using the current time
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		//choosing random starting position
		px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	
		//choosing random starting velocity
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
		//assigns the mass as 1 for each sphere
		mass[i] = 1.0;
	
		//starting forces are zero (they get calculated later)
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3f(
			(i % 3 == 0) ? 1.0f : 0.0f,
			(i % 3 == 1) ? 1.0f : 0.5f,
			(i % 3 == 2) ? 1.0f : 0.5f
			);
			//ternary operator - if the first part is true, use the first option, if not, use the second option
	
	glPushMatrix(); //saves the current transformation state
	glTranslatef(px[i], py[i], pz[i]); //moves the drawing position to i's coordinates
	glutSolidSphere(radius,20,20); //draws a solid sphere
	glPopMatrix(); //restores the previous transformation state so the next sphere can be drawn at its own position
	}
	
	glutSwapBuffers(); //displays the completed frame
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	//halfBoxLength calculates the maximum distance a sphere's center can be from the middle of the box
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
	if(px[i] > halfBoxLength)
	{
		px[i] = 2.0*halfBoxLength - px[i];
		vx[i] = -vx[i];
	}
	else if(px[i] < -halfBoxLength)
	{
		px[i] = -2.0*halfBoxLength - px[i];
		vx[i] = -vx[i];
	}
	
	if(py[i] > halfBoxLength)
	{
		py[i] = 2.0*halfBoxLength - py[i];
		vy[i] = -vy[i];
	}
	else if(py[i] < -halfBoxLength)
	{
		py[i] = -2.0*halfBoxLength - py[i];
		vy[i] = -vy[i];
	}
			
	if(pz[i] > halfBoxLength)
	{
		pz[i] = 2.0*halfBoxLength - pz[i];
		vz[i] = - vz[i];
	}
	else if(pz[i] < -halfBoxLength)
	{
		pz[i] = -2.0*halfBoxLength - pz[i];
		vz[i] = - vz[i];
	}
	}
}

void get_forces()
{
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i + 1; j < NUMBER_OF_SPHERES; j++) //starts at the next sphere so the same sphere is not compared with itself or calculated the same pair twice
		{
			float dx = px[j] - px[i];
			float dy = py[j] - py[i];
			float dz = pz[j] - pz[i];
			
			float r2 = dx*dx + dy*dy + dz*dz;
			
			if(r2 < 0.000001f) //makes sure the spheres are not in the same spot
				continue;
				
			float r = sqrt(r2);
			
			float forceMag =  mass[i]*mass[j]*GRAVITY/r2;
			
			if (r < DIAMETER)
			{
				float dvx = vx[j] - vx[i];
				float dvy = vy[j] - vy[i];
				float dvz = vz[j] - vz[i];
				float inout = dx*dvx + dy*dvy + dz*dvz;
				
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

		float forceX = forceMag*dx/r;
		float forceY = forceMag*dy/r;
		float forceZ = forceMag*dz/r;
		
		fx[i] += forceX;
		fy[i] += forceY;
		fz[i] += forceZ;
	
		fx[j] -= forceX;
		fy[j] -= forceY;
		fz[j] -= forceZ;	
		}
	}
}

void move_bodies(float time)
{
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		float factor = (time == 0.0) ? 0.5 : 1.0;

		//basically F = ma --> a = F/m
		vx[i] += factor*DT*(fx[i] - DAMP*vx[i])/mass[i];
		vy[i] += factor*DT*(fy[i] - DAMP*vy[i])/mass[i];
		vz[i] += factor*DT*(fz[i] - DAMP*vz[i])/mass[i];
		
		//updates the positions using the new velocities
		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initial_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); //sets the camera's viewpoint
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0); //sets up a perspective viewing volume, farther away objects are smaller and vice versa

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N-Body Gravity Simulation");
	printf("N-Body Gravity Simulation\n");
	printf("Number of spheres: %d\n", NUMBER_OF_SPHERES);
	printf("Press Ctrl+C in the terminal to stop.\n");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


