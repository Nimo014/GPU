/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


// KERNELs

// Initialize Global Scene
__global__ void initializeScene(int *dArray1, int val, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
      dArray1[tid] = val;    
    }
}


// Initialize Update X & Y array
__global__ void initializeZero(int *dArray1, int *dArray2, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
      dArray1[tid] = 0;
      dArray2[tid] = 0;     
    }
}

//Get translations for a particular mesh id
__global__ void computeUpdates(int * dTranslations, int* dUpdateX, int* dUpdateY,int n){
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if(tid < n){

    int meshIdx = dTranslations[tid * 3];
    int dir = dTranslations[tid * 3 + 1];
    int amount = dTranslations[tid * 3 + 2];
    int updateX = 0, updateY = 0;
    if(dir == 0){
        updateX = -amount;
    }
    else if(dir == 1){
        updateX = amount;
    }
    else if(dir == 2){
        updateY = -amount;
    }
    else{
      updateY = amount;
    }

    
    // Handle Data Race
    atomicAdd(&dUpdateX[meshIdx], updateX);
    atomicAdd(&dUpdateY[meshIdx], updateY);
    
  }
}

// computeCumlativeUpdate by following transitivity properties
__global__ void computeCumlativeUpdate(int start, int end, int* dCsr, int parent, int* dUpdateX, int* dUpdateY) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (start <= tid && tid < end) {
			int node = dCsr[tid];
			dUpdateX[node] += dUpdateX[parent];
      dUpdateY[node] += dUpdateY[parent];
    }
}

// Display Resulant Matrix
__global__ void fillScene(int meshX, int meshY, int frameSizeX, int frameSizeY, int opacity, int meshIdx, int* mesh, int* globalPositionX, int* globalPositionY, int* dUpdateX, int* dUpdateY, int *GlobalScene, int* GlobalSceneOpacity){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / meshY;
    int col = tid % meshY;


    if( row >= 0 && row < meshX && col >= 0 && col < meshY){
      int posX = globalPositionX[meshIdx] + dUpdateX[meshIdx];
      int posY = globalPositionY[meshIdx] + dUpdateY[meshIdx];

      int globalX = row + posX;
      int globalY = col + posY;

      if (globalX >= 0 && globalX < frameSizeX && globalY >= 0 && globalY < frameSizeY)
      {

          if (GlobalSceneOpacity[globalX * frameSizeY + globalY] < opacity)
          {
              GlobalScene[globalX * frameSizeY + globalY] = mesh[row * meshY + col];
              GlobalSceneOpacity[globalX * frameSizeY + globalY] = opacity;
          }
      }
    }

}

int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

  // Intialize Final PNG To Zero and opacity to int min
	
	int *dCsr, *dGlobalScene, *dGlobalSceneOpacity;
	cudaMalloc(&dCsr, sizeof(int) * V);
	cudaMalloc(&dGlobalSceneOpacity, sizeof(int) * frameSizeX * frameSizeY);
	cudaMalloc(&dGlobalScene, sizeof(int) * frameSizeX * frameSizeY);


	cudaMemcpy(dCsr, hCsr, sizeof(int) * V, cudaMemcpyHostToDevice);
	int n = frameSizeX * frameSizeY;
  int threads = 1024;
  int blocks = ceil(float(n)/threads);
  initializeScene<<<blocks, threads>>>(dGlobalSceneOpacity, INT_MIN, n);
  initializeScene<<<blocks, threads>>>(dGlobalScene, 0, n);

	/*
	printf("Current global positions\n");
	for (int i = 0; i < V; ++i) {
			printf("\n Vertex %d : (%d, %d) ", i, hGlobalCoordinatesX[i], hGlobalCoordinatesY[i]);
	}*/

	// Get resultant movement of particular mesh
  int *dGlobalCoordinatesX, * dGlobalCoordinatesY, *dUpdateX, *dUpdateY, *dTranslations;
  cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
	cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);
	cudaMalloc(&dUpdateX, sizeof(int) * V);
  cudaMalloc(&dUpdateY, sizeof(int) * V);
  cudaMalloc(&dTranslations, sizeof(int) * numTranslations * 3);

  cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
  initializeZero<<<ceil(float(V/1024)), 1024>>>(dUpdateX, dUpdateY, V);
  for (int i = 0; i < numTranslations; ++i) {
      cudaMemcpy(dTranslations + i * 3, translations[i].data(), sizeof(int) * 3, cudaMemcpyHostToDevice);
  }
  // adding each meshIdx translations on GPU
  computeUpdates<<<ceil(float(numTranslations)/1024), 1024>>>(dTranslations, dUpdateX, dUpdateY, numTranslations);
  cudaFree(dTranslations); // no need of translation

  // Run BFS one time to get resultant updates for the mesh and its children on GPU
	std::queue<int> q;
  q.push(0);
  while (!q.empty()) {
        int node = q.front();
        q.pop();
				int start = hOffset[node];
				int end = hOffset[node + 1];
        for (int i = start; i < end; ++i) {
            q.push(hCsr[i]);
        }

				// update dUpdateX and dUpdateY resultant on GPU
				if(end - start> 0)
				{
					// Get result updates
          int threads = 1024;
          int Blocks = ceil(float(V) / threads);
					computeCumlativeUpdate<<<Blocks, threads>>>(start, end, dCsr, node, dUpdateX, dUpdateY);
				}
		}
  cudaFree(dCsr); // no need of csr


  int* dMesh;
  cudaMalloc(&dMesh, sizeof(int) * 10000);

	for (int i = 0; i < V; ++i) {
			int meshX = hFrameSizeX[i];
			int meshY = hFrameSizeY[i];
			int opacity = hOpacity[i];
      cudaMemcpy(dMesh, hMesh[i], sizeof(int) * meshX * meshY, cudaMemcpyHostToDevice);
			int threads = 1024;
			int blocks = ceil(float(meshX*meshY)/ 1024);
			fillScene<<<blocks, threads>>>(meshX, meshY, frameSizeX, frameSizeY, opacity, i, dMesh, dGlobalCoordinatesX, dGlobalCoordinatesY, dUpdateX, dUpdateY, dGlobalScene, dGlobalSceneOpacity);
  }
  cudaFree(dMesh);

	cudaMemcpy(hFinalPng, dGlobalScene, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
  cudaFree(dGlobalScene);
  cudaFree(dGlobalSceneOpacity);
  cudaFree(dGlobalCoordinatesX);
  cudaFree(dGlobalCoordinatesY);
  cudaFree(dUpdateX);
  cudaFree(dUpdateY);
  // Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;
  cudaFree(hFinalPng);
}
