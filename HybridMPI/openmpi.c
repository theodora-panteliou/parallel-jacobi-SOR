/************************************************************

 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define SRC(XX,YY) u_old[(YY)*(pcolumns)+(XX)]
#define DST(XX,YY) u[(YY)*(pcolumns)+(XX)]

#define u(XX,YY) u[(XX)*(pcolumns)+(YY)]
#define u_old(XX,YY) u_old[(XX)*(pcolumns)+(YY)]



/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
static inline double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(XX)*maxYCount+(YY)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
	return error;
}


int main(int argc, char **argv)
{
	int ndims = 2,ierr,my_cart_rank;
	int dims[ndims],coord[ndims];
	int reorder = 0,nrows,ncols,periods[ndims],me2d;
	int down,up,left,right;
	double absoluteError,finerror;

	int prows,pcolumns;
 	int n, m, mits,me,p,mp,mi;
	double alpha, tol, relax;
	double maxAcceptableError;
	double error = 0.0;
	double *u, *u_old, *tmp;
	int allocCount;
	int iterationCount, maxIterationCount;
    double t1, t2;


	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&prov);
	MPI_Comm comm2D;
	MPI_Datatype column_type, row_type;
	MPI_Request send_up,send_down,send_right,send_left,receive_up,receive_down,receive_right,receive_left;
	MPI_Status stat;

	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	if(me == 0){
		printf("\n # procs = %d \n",p);
	}
	if(me == 0){
	//    printf("Input n,m - grid dimension in x,y direction:\n");
		scanf("%d,%d", &n, &m);
		printf("%d,%d\n", n, m);
	//    printf("Input alpha - Helmholtz constant:\n");
		scanf("%lf", &alpha);
	//    printf("Input relax - successive over-relaxation parameter:\n");
		scanf("%lf", &relax);
	//    printf("Input tol - error tolerance for the iterrative solver:\n");
		scanf("%lf", &tol);
	//    printf("Input mits - maximum solver iterations:\n");
		scanf("%d", &mits);
		//printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&mits, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	dims[0]=0;
	dims[1]=0;

	MPI_Dims_create(p, 2, dims);
	
	periods[0] = 0;
	periods[1] = 0;

	prows = n/dims[0] + 2;
	pcolumns = m/dims[1] + 2;

	allocCount = (prows)*(pcolumns);
	MPI_Comm old = MPI_COMM_WORLD;
	MPI_Cart_create(old, ndims, dims, periods, reorder, &comm2D);

	/* column type */
	MPI_Type_vector(prows-2, 1, pcolumns, MPI_DOUBLE,&column_type);
	MPI_Type_commit(&column_type);

	/* row type */
	MPI_Type_contiguous(pcolumns-2, MPI_DOUBLE, &row_type);
	MPI_Type_commit(&row_type);

    // Those two calls also zero the boundary elements
    u = 	(double*)calloc(allocCount, sizeof(double));
    u_old = (double*)calloc(allocCount, sizeof(double));

    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n",prows,pcolumns);
        exit(1);
    }
    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);
	
	int coords[2];
	MPI_Cart_coords(comm2D, me, 2, coords);
	double xStart, yStart; /* local starts (each axis is divided equally to proccesses) */
	
	/* every process has pcolumns-2 real columns */
	xStart = xLeft + deltaX*(coords[0]*(prows-2));
	yStart = yBottom + deltaY*(coords[1]*(pcolumns-2));

    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;
    iterationCount = 0;
    error = HUGE_VAL;

	int x, y;
    double fX, fY;
    double loop_error, total_error;
    double updateVal;
	
	/* topology: rows as columns and columns as rows 
	ex: 0 2
		1 3
	*/
	MPI_Cart_shift(comm2D, 0, 1,  &up, &down);
	MPI_Cart_shift(comm2D, 1, 1, &left, &right);

    clock_t start , diff;
	MPI_Barrier(comm2D);
	if(me == 0){
		start = clock();
		t1 = MPI_Wtime();
	}
	
    MPI_Pcontrol(1);
	#pragma omp parallel private(iterationCount)
	{
    for (iterationCount = 0; iterationCount < maxIterationCount; iterationCount++)
    {
		loop_error = 0.0; /* jacobi error */
		#pragma omp barrier
		#pragma omp master
		{
		/* first prepare receives so that they are ready for sends */
		/* u_old is the array that holds the solution after the most recent buffer swap and the src at one_iteration_jacobi function */
		MPI_Irecv(&(u_old(0,1)),1,row_type,up,1,comm2D,&receive_up);//up
		MPI_Irecv(&(u_old(prows-1,1)),1,row_type,down,0,comm2D,&receive_down);//down

		MPI_Irecv(&(u_old(1,pcolumns-1)),1,column_type,right,3,comm2D,&receive_right);//right
		MPI_Irecv(&(u_old(1, 0)),1,column_type,left,2,comm2D,&receive_left);//left

		MPI_Isend(&(u_old(1,1)),1,row_type,up,0,comm2D,&send_up);//up
		MPI_Isend(&(u_old(prows-2,1)),1,row_type,down,1,comm2D,&send_down);//down

		MPI_Isend(&(u_old(1, pcolumns-2)),1,column_type,right,2,comm2D,&send_right);//right
		MPI_Isend(&(u_old(1,1)),1,column_type,left,3,comm2D,&send_left);//left

		}
	    //INSIDE
		#pragma omp for reduction(+:loop_error) private(fX, fY, updateVal, x, y) collapse(2) schedule(static)
		for (int y = 2; y < (prows-2); y++)
		{
			for (int x = 2; x < (pcolumns-2); x++)
			{
				fY = deltaX+xStart + deltaY*(y-2);
				fX = deltaY+yStart + (x-2)*deltaX;
				updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc -  ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
				DST(x,y) = SRC(x,y) - relax*updateVal;	
				loop_error += updateVal*updateVal;
			}
		}

		#pragma omp master
		{
        //left
        MPI_Wait(&receive_left,&stat);
		fX =  yStart;
		}
		#pragma omp barrier
		#pragma omp for reduction(+:loop_error) private(fY, updateVal, y) schedule(static)
    	for (int y = 2; y < (prows-2); y++)
    	{	
			fY= deltaX+xStart + (y-2)*deltaY;
            updateVal = ((SRC(0,y) + SRC(2,y))*cx + (SRC(1,y-1) + SRC(1,y+1))*cy + SRC(1,y)*cc -  ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
		 	DST(1,y) = SRC(1,y) - relax*updateVal;	
		 	loop_error += updateVal*updateVal;
		}

		#pragma omp master
		{
		//right
		MPI_Wait(&receive_right,&stat);
		fX= (deltaY*(pcolumns-3))+yStart;
		}
		#pragma omp barrier
		#pragma omp for reduction(+:loop_error) private(fY, updateVal, y) schedule(static)
    	for (int y = 2; y < prows-2; y++)
    	{
			fY = deltaX+xStart + (y-2)*deltaY;
           	updateVal = ((SRC(pcolumns-3,y) + SRC(pcolumns-1,y))*cx + (SRC(pcolumns-2,y-1) + SRC(pcolumns-2,y+1))*cy + SRC(pcolumns-2,y)*cc -  ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
			DST(pcolumns-2,y) = SRC(pcolumns-2,y) - relax*updateVal;	
			loop_error += updateVal*updateVal;
		}

		//up
		#pragma omp master
		{
		MPI_Wait(&receive_up,&stat);
		fY =xStart;
		}
		#pragma omp barrier
		#pragma omp for reduction(+:loop_error) private(fX, updateVal, x) schedule(static)
		for (int x = 1; x < (pcolumns-1); x++)
		{
			fX = yStart + (x-1)*deltaX;
			updateVal = ((SRC(x-1,1) + SRC(x+1,1))*cx + (SRC(x,0) + SRC(x,2))*cy + SRC(x,1)*cc - ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
			DST(x,1) = SRC(x,1) - relax*updateVal;	
			loop_error += updateVal*updateVal;
		}

		//down
		#pragma omp master
		{
        MPI_Wait(&receive_down,&stat);	
		fY =  deltaX*(prows-3)+xStart;
		}
		#pragma omp barrier
		#pragma omp for reduction(+:loop_error) private(fX, updateVal, x) schedule(static)
		for (int x =  1; x < pcolumns-1; x++)
		{
			fX = yStart + (x-1)*deltaX;
			updateVal = ((SRC(x-1,prows-2) + SRC(x+1,prows-2))*cx + (SRC(x,prows-3) + SRC(x,prows-1))*cy + SRC(x,prows-2)*cc -  ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
			DST(x,prows-2) = SRC(x,prows-2) - relax*updateVal;	
			loop_error += updateVal*updateVal;
		}
	
		#pragma omp master
		{
		/* swap buffers */
		tmp = u_old;
        u_old = u;
        u = tmp;

		/* wait sends */
		MPI_Wait(&send_up, &stat);
		MPI_Wait(&send_down, &stat);
		MPI_Wait(&send_right, &stat);
		MPI_Wait(&send_left, &stat);

		total_error = 0.0; /* sum */
		MPI_Allreduce(&loop_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, comm2D);
		error = sqrt(total_error)/(m*n); /* total error for the entire solution at current iterations */
		}
		#pragma omp barrier
		if (error <= maxAcceptableError){
			break;
		}
		
    }
	}
    MPI_Pcontrol(0);
	MPI_Barrier(comm2D);
	t2 = MPI_Wtime();//process 0
	
	if(me == 0){
		diff = clock() - start;
		printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 );
	}
	// u_old holds the solution after the most recent buffers swap
	absoluteError = checkSolution(xStart, yStart, prows, pcolumns, u_old, deltaX, deltaY, alpha);

	MPI_Allreduce(&absoluteError,&finerror,1,MPI_DOUBLE,MPI_SUM,comm2D);
	finerror = sqrt(finerror)/(m*n);
	if(me==0){
		double sec = diff/(double) CLOCKS_PER_SEC;
		printf("Time %lf \n",sec);
		printf("Residual %g\n",error);
		printf("The sum error of the iterative solution is is %g\n", finerror);
	}
	MPI_Finalize();
	return 0;
}