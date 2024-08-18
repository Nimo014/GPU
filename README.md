# GPU
Parallel Programming Project


## Project 1: Moving Matrices (Scene Graph)

### Problem Statement:
Scene Graphs are widely used in the domains of Computer Graphics and Computer Vision. They offer a convenient way of chaining translations/transformations from one component of a scene to multiple other components.

This project is an exercise in implementing a simplified Scene Graph with a focus on maximizing parallelism. The task is to generate a scene of given dimensions using multiple grayscale images (meshes) with their initial positions and a multi-set of translations. The relationships between the images are defined by an edge list, ensuring that translations propagate through connected components.

### Key Features:
- **Graph Structure:** The relationships are represented as a tree rooted at 0, where an edge from image `a` to image `b` means that both images translate together. Translations are transitive.
- **Handling Overlaps:** When images overlap, the more opaque matrix is visible in the overlapping regions.
- **Parallelism:** The implementation leverages parallelism to optimize the translation of images and their transformations across the scene.

### Input:
- Grayscale images as matrices.
- Initial positions as (x, y) coordinates.
- Edge list defining relationships between images.

### Output:
- A scene matrix representing the final positions of all images after applying translations.

### Usage:
Provide detailed instructions on how to run the project, including dependencies and example commands.

## Project 2: Game Simulation (Tank Battle)

### Problem Statement:
Simulate a game between tanks on a battlefield grid of size M x N. The goal is to calculate the score of each tank after multiple rounds, adhering to specific game rules.

### Key Features:
- **Grid-based Simulation:** The battlefield is represented as a grid, with tanks placed randomly on integer coordinates.
- **Health Points (HP):** Each tank has HP, and a tank is destroyed when its HP drops to 0 or below.
- **Turn-based Combat:** In each round, tanks that are still active fire at other tanks based on a specific pattern.
- **Score Calculation:** Tanks gain points by successfully hitting other tanks, with various scenarios affecting hit detection.

### Rules:
1. Tanks fire in the direction of the next tank according to a cyclic pattern.
2. If a round is a multiple of T (number of tanks), no shots are fired.
3. Hits reduce the target tank's HP and increase the shooter's score.
4. Complex hit detection logic ensures realistic interactions between tanks on the grid.

### Input:
- Grid dimensions (M x N).
- Number of tanks (T) and their initial coordinates.
- Initial HP of each tank.

### Output:
- Final scores of all tanks after the game ends.

### Usage:
Execute `nvcc main.cu` and provide `input.txt` as input using the command: `./a.out < input.txt`.
