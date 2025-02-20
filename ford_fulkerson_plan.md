# Ford-Fulkerson Implementation Plan

## Overview
Implement the Ford-Fulkerson algorithm to find maximum matching in a bipartite graph representation of the grid pairing problem.

## 1. Graph Construction

### Bipartite Graph Structure
- Left vertices: Cells with even (i+j)
- Right vertices: Cells with odd (i+j)
- Edges: Between valid pairs based on:
  * Adjacency rules (cells must be adjacent)
  * Color compatibility rules
  * No black cells allowed

### Required Methods
```python
def build_bipartite_graph(self):
    """Constructs bipartite graph representation"""
    # 1. Separate cells into even/odd sets
    # 2. Create adjacency lists
    # 3. Initialize residual graph
```

## 2. Ford-Fulkerson Algorithm

### Core Components

#### Find Augmenting Path
```python
def find_augmenting_path(self, source, sink, parent):
    """
    Uses BFS to find an augmenting path from source to sink
    Returns True if path exists, False otherwise
    Updates parent array with the path
    """
```

#### Augment Path
```python
def augment_path(self, source, sink, parent):
    """
    Augments the path found by BFS
    Updates residual graph
    Returns the flow sent along the path
    """
```

#### Main Algorithm
```python
def run(self):
    """
    Main solver method:
    1. Build bipartite graph
    2. While augmenting path exists:
       - Find path using BFS
       - Augment path
    3. Convert max flow to matching
    4. Update self.pairs and self.cells
    """
```

## 3. Solution Conversion

### Converting Flow to Pairs
- Track which edges in residual graph have flow
- Convert edges with flow to cell pairs
- Update solver's pairs and cells lists

## Implementation Steps

1. Create SolverFordFulkerson class inheriting from Solver
2. Implement graph construction methods
3. Implement core Ford-Fulkerson algorithm
4. Add solution conversion logic
5. Add testing and validation
6. Optimize performance

## Complexity Analysis

- Time Complexity: O(VE) where:
  * V = number of vertices (cells in grid)
  * E = number of edges (valid pairs)
- Space Complexity: O(V + E) for graph storage

## Testing Strategy

1. Unit Tests:
   - Graph construction
   - Path finding
   - Flow augmentation
   - Solution validation

2. Integration Tests:
   - Full algorithm on sample grids
   - Edge cases (no valid pairs, all pairs valid)
   - Performance on larger grids