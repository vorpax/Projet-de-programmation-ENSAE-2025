"""
Unit tests for the SolverHungarian class against known best scores.

This module specifically tests the Hungarian algorithm implementation
against the best scores recorded in best-score.txt for all grid sizes.
"""

import sys
import unittest
import re
import time

sys.path.append("code/")
from grid import Grid
from solver import SolverHungarian


class TestHungarianBestScores(unittest.TestCase):
    """
    Test the Hungarian solver against known best scores for all grid sizes.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the best scores from the best-score.txt file once for all tests.
        """
        cls.best_scores = {}
        
        # Parse the best-score.txt file
        with open("tests/best-score.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_grid = None
        for line in lines:
            line = line.strip()
            if line == "---":
                continue
            elif line.endswith(".in"):
                current_grid = line
            elif current_grid and line.isdigit():
                cls.best_scores[current_grid] = int(line)
        
        print(f"Loaded {len(cls.best_scores)} grid best scores")
    
    def _test_grid_with_hungarian(self, grid_name, tolerance=0.25):
        """
        Test a specific grid with the Hungarian solver.
        
        Parameters:
        -----------
        grid_name: str
            The name of the grid file to test
        tolerance: float
            The acceptable percentage deviation from the best score.
            Positive tolerance allows scores above best_score.
            Negative scores are always acceptable (better than expected).
            
        Returns:
        --------
        dict
            Test results including time, score, best score, and deviation
        """
        if grid_name not in self.best_scores:
            self.fail(f"No best score found for {grid_name}")
            
        best_score = self.best_scores[grid_name]
        
        try:
            # Load the grid
            grid = Grid.grid_from_file(f"input/{grid_name}", read_values=True)
            grid.cell_init()
            
            # Run the solver with timing
            start_time = time.time()
            solver = SolverHungarian(grid)
            matching_pairs = solver.run()
            end_time = time.time()
            
            # Calculate results
            execution_time = end_time - start_time
            actual_score = solver.score()
            
            # Calculate deviation from best score
            if best_score == 0:
                # Avoid division by zero
                deviation_percentage = 0 if actual_score == 0 else float('inf')
            else:
                deviation_percentage = ((actual_score - best_score) / best_score) * 100
            
            # Determine quality rating
            if actual_score <= best_score:
                quality = "Better than optimal" if actual_score < best_score else "Optimal"
            else:
                quality = "Suboptimal" if deviation_percentage <= tolerance * 100 else "Poor"
            
            print(f"Grid {grid_name} ({grid.n}x{grid.m}): Score={actual_score}, "
                  f"Best={best_score}, Time={execution_time:.3f}s, "
                  f"Deviation={deviation_percentage:.2f}%, "
                  f"Quality={quality}, Pairs={len(matching_pairs)}")
            
            # Check if within tolerance:
            # - Scores above best_score should be within positive tolerance
            # - Scores below best_score are always acceptable (that's better than expected!)
            if actual_score < best_score:
                is_within_tolerance = True  # Better than optimal is good!
            else:
                # Calculate tolerance range for scores above best
                max_acceptable = best_score * (1 + tolerance)
                is_within_tolerance = actual_score <= max_acceptable
                
            self.assertTrue(is_within_tolerance, 
                          f"Score {actual_score} exceeds best score {best_score} "
                          f"by more than {tolerance*100}%")
            
            return {
                "grid": grid_name,
                "grid_size": f"{grid.n}x{grid.m}",
                "time": execution_time,
                "score": actual_score,
                "best_score": best_score,
                "deviation": deviation_percentage,
                "pairs": len(matching_pairs),
                "quality": quality
            }
            
        except Exception as e:
            self.fail(f"Error testing grid {grid_name}: {str(e)}")
    
    def test_hungarian_small_grids(self):
        """
        Test the Hungarian solver on small grid files.
        """
        # Find all small grid files in the best scores dictionary
        small_grids = [name for name in self.best_scores.keys() 
                      if re.match(r'grid0[0-9]\.in', name)]
        
        if not small_grids:
            self.skipTest("No small grids found in best-score.txt")
            
        print(f"\nTesting Hungarian solver on {len(small_grids)} small grids...")
        
        results = []
        for grid_name in sorted(small_grids):
            result = self._test_grid_with_hungarian(grid_name)
            results.append(result)
            
        # Summary
        avg_deviation = sum(r["deviation"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\nSmall grids summary: Average deviation={avg_deviation:.2f}%, "
              f"Average time={avg_time:.3f}s")
    
    def test_hungarian_medium_grids(self):
        """
        Test the Hungarian solver on medium grid files.
        """
        # Find all medium grid files in the best scores dictionary
        medium_grids = [name for name in self.best_scores.keys() 
                       if re.match(r'grid1[0-9]\.in', name)]
        
        if not medium_grids:
            self.skipTest("No medium grids found in best-score.txt")
            
        print(f"\nTesting Hungarian solver on {len(medium_grids)} medium grids...")
        
        results = []
        for grid_name in sorted(medium_grids):
            result = self._test_grid_with_hungarian(grid_name)
            results.append(result)
            
        # Summary
        avg_deviation = sum(r["deviation"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\nMedium grids summary: Average deviation={avg_deviation:.2f}%, "
              f"Average time={avg_time:.3f}s")
    
    def test_hungarian_large_grids(self):
        """
        Test the Hungarian solver on large grid files.
        """
        # Find all large grid files in the best scores dictionary
        large_grids = [name for name in self.best_scores.keys() 
                      if re.match(r'grid2[0-9]\.in', name)]
        
        if not large_grids:
            self.skipTest("No large grids found in best-score.txt")
            
        print(f"\nTesting Hungarian solver on {len(large_grids)} large grids...")
        
        # Use higher tolerance for large grids since they're more complex
        tolerance = 0.1
        
        results = []
        for grid_name in sorted(large_grids):
            result = self._test_grid_with_hungarian(grid_name, tolerance=tolerance)
            results.append(result)
            
        # Summary
        avg_deviation = sum(r["deviation"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\nLarge grids summary: Average deviation={avg_deviation:.2f}%, "
              f"Average time={avg_time:.3f}s")
    
    def test_all_grid_statistics(self):
        """
        Run SolverHungarian on all grids and collect statistics.
        
        This test doesn't assert but collects useful statistics about the performance
        of the Hungarian solver across all grid sizes.
        """
        all_grids = sorted(self.best_scores.keys())
        
        if not all_grids:
            self.skipTest("No grids found in best-score.txt")
            
        print(f"\nCollecting statistics for Hungarian solver on all {len(all_grids)} grids...")
        
        # Statistics to collect
        total_time = 0
        total_deviation = 0
        optimal_count = 0  # Number of grids where Hungarian found the optimal solution
        near_optimal_count = 0  # Within 5% of optimal
        suboptimal_count = 0  # More than 5% from optimal
        
        # Test each grid
        for grid_name in all_grids:
            best_score = self.best_scores[grid_name]
            
            try:
                # Load and run
                grid = Grid.grid_from_file(f"input/{grid_name}", read_values=True)
                grid.cell_init()
                
                start_time = time.time()
                solver = SolverHungarian(grid)
                solver.run()
                execution_time = time.time() - start_time
                
                score = solver.score()
                
                # Calculate statistics
                total_time += execution_time
                
                if best_score == 0:
                    deviation = 0 if score == 0 else float('inf')
                else:
                    deviation = ((score - best_score) / best_score) * 100
                
                total_deviation += deviation
                
                # Categorize result
                if score == best_score:
                    optimal_count += 1
                elif deviation <= 5:
                    near_optimal_count += 1
                else:
                    suboptimal_count += 1
                    
            except Exception as e:
                print(f"Error processing grid {grid_name}: {str(e)}")
        
        # Print statistics
        avg_time = total_time / len(all_grids)
        avg_deviation = total_deviation / len(all_grids)
        
        print("\n" + "=" * 60)
        print("Hungarian Solver Performance Statistics")
        print("=" * 60)
        print(f"Total grids tested: {len(all_grids)}")
        print(f"Average execution time: {avg_time:.3f}s")
        print(f"Average score deviation: {avg_deviation:.2f}%")
        print(f"Optimal solutions: {optimal_count} ({optimal_count/len(all_grids)*100:.1f}%)")
        print(f"Near-optimal solutions (within 5%): {near_optimal_count} "
              f"({near_optimal_count/len(all_grids)*100:.1f}%)")
        print(f"Suboptimal solutions (>5% deviation): {suboptimal_count} "
              f"({suboptimal_count/len(all_grids)*100:.1f}%)")
        

if __name__ == "__main__":
    unittest.main()