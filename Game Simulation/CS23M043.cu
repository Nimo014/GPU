#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here
__device__ long long int k = 1;

__device__ bool isColinear(int x1, int y1, int x2, int y2, int x_c, int y_c)
{   
    long long int a1 = x1;
    long long int b1 = y1;
    long long int a2 = x2;
    long long int b2 = y2;
    long long int a3 = x_c;
    long long int b3 = y_c;
    return (a1 - a2) * (b2 - b3) == (a2 - a3) * (b1 - b2);
}

__device__ int isInSameDir(int x1, int y1, int x2, int y2)
{
    if (x1 >= x2)
    {
        return (y1 > y2) ? 1 : 2;
    }
    else
    {
        return (y1 > y2) ? 3 : 4;
    }
}

__device__ long long int euclideanDistance(int x1, int y1, int x2, int y2)
{
    long long int dx = x2 - x1;
    long long int dy = y2 - y1;
    return dx * dx + dy * dy;
}

__device__ int is_same_dir(int xsrc,int ysrc, int xtarget,int ytarget, int xtank, int ytank)
{
    return (xsrc != xtarget) ? (((xsrc > xtarget && xsrc < xtank)||(xsrc > xtank && xsrc < xtarget)) ? 0 : 1) : (((ysrc > ytarget && ysrc < ytank)||(ysrc > ytank && ysrc < ytarget)) ? 0 : 1);
}

// Kernel function for the main computation
__global__ void Round(int T, int* d_xcoord, int* d_ycoord, int* dscore, int* d_hp, int* d_destroyed) {
    __shared__ int tank;
    __shared__ int target;
    __shared__ int trueTarget;
    __shared__ long long int minDist;
    __shared__ int lock;
    //__shared__ int old;
    
    
    int tid = threadIdx.x;
    if(tid == 0){
      tank = blockIdx.x;
      target = (tank + k) % T;
      trueTarget = -1;
      minDist = LLONG_MAX;
      lock = 0;
      //printf("Orginal Tank %d Fire in the dir of Tank %d \n", tank, target);
    
    }
    
    __syncthreads();
    // Shift vector to idx as origin
    if(tid != tank && d_destroyed[tank] == 0){

       bool check1 = isColinear(d_xcoord[tank], d_ycoord[tank], d_xcoord[target], d_ycoord[target], d_xcoord[tid], d_ycoord[tid]);
       //bool check2 = isInSameDir(d_xcoord[tank], d_ycoord[tank], d_xcoord[target], d_ycoord[target]) == isInSameDir(d_xcoord[tank], d_ycoord[tank], d_xcoord[tid], d_ycoord[tid]);
       bool check2 = is_same_dir(d_xcoord[tank], d_ycoord[tank], d_xcoord[target], d_ycoord[target], d_xcoord[tid], d_ycoord[tid]);
       
       //printf("Checking Tank %d Fire in the dir of Tank %d \n", tank, target);
       if (check1 && check2 && d_destroyed[tid] != 1 ){
            //long int dist = euclideanDistance(d_xcoord[tank], d_ycoord[tank], d_xcoord[tid], d_ycoord[tid]);
                        
            //printf("Confirm Collinear Tank %d Fire in the dir of Tank %d Dist %ld \n", tank, tid, dist);

            for(int i = 0 ; i < 32; i++){
              if(tid % 32 == i){
                while(atomicCAS(&lock, 0, 1) != 0){}
                long long int dist = euclideanDistance(d_xcoord[tank], d_ycoord[tank], d_xcoord[tid], d_ycoord[tid]);
                if(minDist > dist)
                {
                  minDist = dist;
                  trueTarget = tid;
                }
                atomicExch(&lock, 0);
              }
            }
            
              /*
                 do
                  {
                    old = atomicCAS(&lock, 0, 1);
                    if (old == 0) {
                        long long int dist = euclideanDistance(d_xcoord[tank], d_ycoord[tank], d_xcoord[tid], d_ycoord[tid]);
                        if(minDist > dist)
                        {
                          minDist = dist;
                          trueTarget = tid;
                        }
                      lock = 0; // unlock
                    }
                  } while (old != 0);
              */  
            
        }
    }

    __syncthreads();
    if(tid == 0){
      // Check if the target tank is not destroyed

      if (trueTarget != -1 ) {
          //printf("Firing Collinear Tank %d Fire in the dir of Tank %d \n", tank, trueTarget);
        
          atomicSub(&d_hp[trueTarget], 1); // Decrement HP of the target tank
          atomicAdd(&dscore[tank], 1); // Increment score of the firing tank
      }
    }

}



__global__ void declaredDead(int T, int* dead, int* d_xcoord, int* d_ycoord, int* dscore, int* d_hp, int* d_destroyed){
  int tid = threadIdx.x;
  
   if(tid == 0){
      k++;
      if ((k % T) == 0){
        k++;
      }  
  //     for (int i = 0; i < T; i++) {
  //         printf("Final Result Tank %d: x-%d y-%d s-%d h-%d \n", i, d_xcoord[i], d_ycoord[i], dscore[i], d_hp[i]);
  //     }
  
  }

  if( tid < T){
    // New Tank found dead
    if(d_hp[tid] <= 0 && d_destroyed[tid] == 0){
      d_destroyed[tid] = 1;
      atomicAdd(dead, 1);
    }
  }

}

__global__ void assignValue(int* d_hp, int* dscore, int* d_destroyed, int T, int H) {
    int tid = threadIdx.x;
    d_hp[tid] = H;
    dscore[tid] = 0;
    d_destroyed[tid] = 0;
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    // Memory allocation on GPU
    int* d_xcoord, *d_ycoord, *dscore, *d_hp, *d_destroyed;
    cudaMalloc(&d_xcoord, T * sizeof(int));
    cudaMalloc(&d_ycoord, T * sizeof(int));
    cudaMalloc(&dscore, T * sizeof(int));
    cudaMalloc(&d_hp, T * sizeof(int));
    cudaMalloc(&d_destroyed, T * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    assignValue<<<1,T>>>(d_hp, dscore, d_destroyed, T, H);
    
    int *dead;
    cudaHostAlloc(&dead, sizeof(int), 0);
    *dead = 0;

    // Computation On GPU
    while(*dead < T - 1){
      Round<<<T, T>>>(T, d_xcoord, d_ycoord, dscore, d_hp, d_destroyed);
      declaredDead<<<1,T>>>(T, dead, d_xcoord, d_ycoord, dscore, d_hp, d_destroyed);
      cudaDeviceSynchronize();
      
    }
    cudaMemcpy(score, dscore, T * sizeof(int), cudaMemcpyDeviceToHost);


    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaFree(d_xcoord);
    cudaFree(d_ycoord);
    cudaFree(dscore);
    cudaFree(d_hp);
    cudaFree(d_destroyed);
    cudaFreeHost(dead);
    cudaDeviceSynchronize();
    return 0;
}