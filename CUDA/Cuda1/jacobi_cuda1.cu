#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include "timestamp.h"

#define N 32 // 2-4-8-16-32 max 32
#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dest[(YY)*maxXCount+(XX)]


__global__ void one_jacobi_itteration(int maxXCount, int maxYCount,double*dest,double *src,double deltaX, double deltaY,double cc,double cx,double cy,double relax,double alpha,double *loop_error_c)
{
    __shared__ double  temp[N*N]; 
    int index = blockDim.x * threadIdx.y + threadIdx.x;
    temp[index]=0.0;
    //Calculate the column index of the Pd element, denote by x
     int x = threadIdx.x + blockIdx.x * blockDim.x+1; 
    //Calculate the row index of the Pd element, denote by y
    int y = threadIdx.y + blockIdx.y * blockDim.y+1; 
    
    if (x < maxXCount-1 && y < maxYCount -1) {
  	    double fY =  -1.0 + (y-1)*deltaY;
       	double fX = -1.0 + (x-1)*deltaX;
        double updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - ((fX*fX-1.0)*(alpha*(1.0-fY*fY) + 2.0) - 2.0*(1.0 - fY*fY)))/cc;
		DST(x,y) = SRC(x,y) - relax*updateVal;	
		temp[index] = updateVal*updateVal;
        
        // Synchronize (ensure all the data is available) 
        __syncthreads();
        
        //apply stencil

        for(int offset=N*N/2;offset>=1;offset=offset/2)
        {
            if(index<offset)
            {
                temp[index]=temp[index]+temp[index+offset];
            }
        __syncthreads();
        }
        if(threadIdx.x + threadIdx.y == 0) {
            atomicAdd(loop_error_c, temp[0]);
        }
    }
}

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
    int n, m, mits;
    double alpha, tol, relax;
    double error;
    double *u, *u_old, *loop_error, *tmp,*u1;
    int allocCount;
    int iterationCount;
    timestamp t1;
    double t2;

//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    allocCount = (n+2)*(m+2);
    
    // Those three calls also zero the boundary elements
    u1 = 	(double*)calloc(allocCount, sizeof(double));

    cudaMalloc((void**)&u_old, allocCount * sizeof(double));
    cudaMalloc((void**)&u, allocCount * sizeof(double));
    cudaMalloc((void**)&loop_error, 1 * sizeof(double));


    cudaMemset(u, 0, allocCount * sizeof(double));
    cudaMemset(u_old, 0, allocCount * sizeof(double));
    cudaMemset(loop_error, 0, sizeof(double));

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);
	
    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;
    
    iterationCount = 0;
    error = HUGE_VAL;


    dim3 dimBl(N, N);
    dim3 dimGr(FRACTION_CEILING(n+2, N), FRACTION_CEILING(m+2, N));

    clock_t start , diff;
    start = clock();
    t1 = getTimestamp();
    

    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < mits && error > tol)
    {
        cudaMemset(loop_error, 0, sizeof(double));
        
        one_jacobi_itteration<<<dimGr,dimBl>>>((m+2),(n+2),u,u_old,deltaX,deltaY,cc,cx,cy,relax,alpha,loop_error);
                
        cudaMemcpy(&error,loop_error,sizeof(double),cudaMemcpyDeviceToHost);

        iterationCount++;
        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;
        error = sqrt(error)/(n*m);
    }

	t2 = getElapsedtime(t1) / 1000.0;

    // u1 holds the solution after the most recent buffers swap
	cudaMemcpy(u1,u_old,sizeof(double)*allocCount,cudaMemcpyDeviceToHost);
	
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount,t2);
	
    
	diff = clock() - start;
	double sec = diff/(double) CLOCKS_PER_SEC;
	printf("Clock Time %lf \n",sec);
	printf("Residual %g\n",error);
	printf("The sum error of the iterative solution is is %g\n", sqrt(checkSolution(-1.0,-1.0, n+2, m+2, u1, deltaX, deltaY, alpha))/(m*n));
    
    return 0;
}
