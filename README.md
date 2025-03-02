# Projet de Programmation ENSAE 2025

Houard Alexandre (vorpax) et Octave Hedarchet (maupassaant)

## Implémentation

### Structure
- `Grid`: Représentation de la grille avec couleurs et valeurs
- `Cell`: Représentation des cellules individuelles
- `Solver`: Interface pour les algorithmes de résolution

### Algorithmes
- `SolverEmpty`: Solution vide (pour tests)
- `SolverGreedy`: Sélection gloutonne par coût minimal
- `SolverFulkerson`: Implémentation de Ford-Fulkerson pour couplage biparti maximum

### Fonctionnalités clés
- Chargement de grilles depuis fichiers
- Visualisation graphique
- Matrice de compatibilité des couleurs
- Calcul optimisé du couplage maximisant le score

## Utilisation
```python
# Exemple d'utilisation
grid = Grid.grid_from_file("input/grid21.in", read_values=True)
solver = SolverFulkerson(grid)
matching_pairs = solver.run()
print(f"Score final: {solver.score()}")
```
