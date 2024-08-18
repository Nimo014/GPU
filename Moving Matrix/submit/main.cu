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


// kernel
// Root
__global__ void updateRoot(int node, int updateX, int updateY, int* globalPositionX, int * globalPositionY){
			globalPositionX[node] += updateX;
      globalPositionY[node] += updateY;
}
// Child
__global__ void bfsLevel(int start, int end, int* dCsr, int updateX, int updateY, int* globalPositionX, int* globalPositionY) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (start <= tid && tid < end) {
			int node = dCsr[tid];
			globalPositionX[node] += updateX;
      globalPositionY[node] += updateY;
    }
}

// Display Resulant Matrix
__global__ void fillScene(int meshX, int meshY, int frameSizeX, int frameSizeY, int opacity, int meshIdx, int* mesh, int* globalPositionX, int* globalPositionY, int *GlobalScene, int* GlobalSceneOpacity){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / meshY;
    int col = tid % meshY;

    
    if( row >= 0 && row < meshX && col >= 0 && col < meshY){
      int posX = globalPositionX[meshIdx];
      int posY = globalPositionY[meshIdx];

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
	int* GlobalSceneOpacity = (int*)malloc(sizeof(int) * frameSizeX * frameSizeY);
	for (int i = 0; i < frameSizeX; ++i) {
        for (int j = 0; j < frameSizeY; ++j) {
            hFinalPng[i * frameSizeY + j] = 0;
						GlobalSceneOpacity[i * frameSizeY + j] = INT_MIN;
        }
    }

	int *dGlobalCoordinatesX, * dGlobalCoordinatesY, *dCsr, *dGlobalScene, *dGlobalSceneOpacity;
	cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
	cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);
	cudaMalloc(&dCsr, sizeof(int) * V);
	cudaMalloc(&dGlobalSceneOpacity, sizeof(int) * frameSizeX * frameSizeY);
	cudaMalloc(&dGlobalScene, sizeof(int) * frameSizeX * frameSizeY);

	cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dCsr, hCsr, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(&dGlobalSceneOpacity, GlobalSceneOpacity, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(&dGlobalScene, hFinalPng, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);



	printf("Current global positions\n");
	for (int i = 0; i < V; ++i) {
			printf("\n Vertex %d : (%d, %d) ", i, hGlobalCoordinatesX[i], hGlobalCoordinatesY[i]);
	}
	// update Childrens
	for (auto translation : translations) {
		int meshIdx = translation[0];
		int dir = translation[1];
		int amount = translation[2];

		int updateX = 0, updateY = 0;
		switch (dir) {
				case 0: updateX = -amount; break;
				case 1: updateX = amount; break;
				case 2: updateY = -amount; break;
				case 3: updateY = amount; break;
				default: break;
		}


		std::queue<int> q;
    q.push(meshIdx);
		//updateRoot
		updateRoot<<<1,1>>>(meshIdx, updateX, updateY, dGlobalCoordinatesX, dGlobalCoordinatesY);
    while (!q.empty()) {
        int node = q.front();
        q.pop();
				int start = hOffset[node];
				int end = hOffset[node + 1];
        for (int i = start; i < end; ++i) {
            q.push(hCsr[i]);
        }

				// update Child
				if(end - start> 0)
				{
					// Update Children Parallel
          int threads = 1024;
          int Blocks = ceil(float(V) / threads);
					bfsLevel<<<Blocks, threads>>>(start, end, dCsr, updateX, updateY, dGlobalCoordinatesX, dGlobalCoordinatesY);
				}
		}

	}
	// Print Mesh to Scene
	for (int v = 0; v < V; v++) {
			int meshX = hFrameSizeX[v];
			int meshY = hFrameSizeY[v];
			int posX = hGlobalCoordinatesX[v];
			int posY = hGlobalCoordinatesY[v];
			int opacity = hOpacity[v];
			int* mesh = hMesh[v];
			for (int i = 0; i < meshX; ++i) {
					for (int j = 0; j < meshY; ++j) {
							int globalX = posX + i;
							int globalY = posY + j;


							if (globalX >= 0 && globalX < frameSizeX && globalY >= 0 && globalY < frameSizeY) {
									if (GlobalSceneOpacity[globalX * frameSizeY + globalY] < opacity) {
											hFinalPng[globalX * frameSizeY + globalY] = mesh[i * meshY + j];
											GlobalSceneOpacity[globalX * frameSizeY + globalY] = opacity;
									}

							}
					}
			}
	}
  /*
	for (int i = 0; i < frameSizeX; ++i) {
        for (int j = 0; j < frameSizeY; ++j) {
            printf("%d ",hFinalPng[i * frameSizeY + j]);
        }
				printf("\n");
  }
  */
	for (int i = 0; i < V; ++i) {
			int meshX = hFrameSizeX[i];
			int meshY = hFrameSizeY[i];
			int opacity = hOpacity[i];
			int* dMesh;
      cudaMalloc(&dMesh, sizeof(int) * meshX * meshY);
      cudaMemcpy(dMesh, hMesh[i], sizeof(int) * meshX * meshY, cudaMemcpyHostToDevice);
			int threads = 1024;
			int blocks = ceil(float(meshX*meshY)/ 1024);
			fillScene<<<blocks, threads>>>(meshX, meshY, frameSizeX, frameSizeY, opacity, i, dMesh, dGlobalCoordinatesX, dGlobalCoordinatesY, dGlobalScene, dGlobalSceneOpacity);
      cudaDeviceSynchronize();
  }
	cudaMemcpy(hFinalPng, dGlobalScene, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
  
  // Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
