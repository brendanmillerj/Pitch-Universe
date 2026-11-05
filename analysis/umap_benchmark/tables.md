# Benchmark tables

Pitcher-clustered bootstrap, 1000 draws, 95% percentile intervals in brackets. Weighted = rows pooled across pitch types. Self-free = a pitcher's other seasons are excluded from both the true and the on-map neighbour sets.

## A1. Neighbour preservation by picture (all types, weighted)

| view | dims | recall@10 self-free % [CI] | unweighted % | vs t-SNE 2D (pts) | vs t-SNE 3D (pts) | all-rows % | recall@30 % | trust | continuity | dist corr | K_max | Q_NX(10) | Q_NX(50) | Q_NX(200) | R_NX AUC | island kNN acc | island ARI | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Movement (ride x HB) | 2 | 10 [9, 10] | 11 | -34 [-34, -33] | -38 [-39, -38] | 11 | 19 | 0.78 | 0.91 | 0.70 | 230 | 0.11 | 0.25 | 0.49 | 0.21 | 0.45 | 0.18 | 0 |
| Savant 5 (PCA 2) | 2 | 10 [10, 11] | 12 | -33 [-34, -32] | -38 [-39, -37] | 12 | 20 | 0.78 | 0.91 | 0.71 | 224 | 0.12 | 0.26 | 0.49 | 0.22 | 0.45 | 0.17 | 0 |
| PCA 2 | 2 | 14 [13, 14] | 16 | -29 [-30, -29] | -34 [-35, -34] | 15 | 26 | 0.84 | 0.95 | 0.83 | 206.50 | 0.15 | 0.33 | 0.58 | 0.28 | 0.57 | 0.24 | 0 |
| PCA 3 | 3 | 25 [25, 26] | 28 | -18 [-19, -17] | -23 [-24, -22] | 27 | 39 | 0.91 | 0.97 | 0.90 | 168 | 0.27 | 0.46 | 0.69 | 0.40 | 0.71 | 0.30 | 0 |
| Isomap 2D | 2 | 14 [14, 15] | 17 | -29 [-29, -28] | -34 [-34, -33] | 16 | 26 | 0.84 | 0.95 | 0.81 | 209.50 | 0.16 | 0.33 | 0.58 | 0.29 | 0.60 | 0.27 | 0.17 |
| TriMap 2D | 2 | 28 [28, 29] | 31 | -15 [-15, -14] | -20 [-20, -19] | 31 | 36 | 0.91 | 0.95 | 0.72 | 140.50 | 0.31 | 0.39 | 0.56 | 0.34 | 0.77 | 0.31 | 10.39 |
| PHATE 2D | 2 | 18 [17, 18] | 20 | -25 [-26, -25] | -30 [-31, -30] | 19 | 28 | 0.85 | 0.94 | 0.69 | 186.50 | 0.19 | 0.33 | 0.54 | 0.27 | 0.66 | 0.30 | 3.61 |
| PaCMAP 2D (10) | 2 | 30 [30, 31] | 33 | -13 [-13, -12] | -18 [-18, -17] | 33 | 38 | 0.91 | 0.94 | 0.65 | 85 | 0.33 | 0.40 | 0.55 | 0.33 | 0.78 | 0.30 | 0.46 |
| PaCMAP 2D (15) | 2 | 26 [26, 27] | 29 | -17 [-17, -16] | -22 [-22, -21] | 28 | 37 | 0.91 | 0.94 | 0.67 | 124 | 0.28 | 0.40 | 0.56 | 0.32 | 0.78 | 0.32 | 0.25 |
| LocalMAP 2D (15) | 2 | 26 [25, 26] | 27 | -17 [-18, -17] | -22 [-23, -22] | 28 | 36 | 0.91 | 0.93 | 0.59 | 117 | 0.28 | 0.40 | 0.55 | 0.30 | 0.79 | 0.35 | 0.28 |
| UMAP 2D (15, 0.3) | 2 | 30 [30, 31] | 33 | -13 [-13, -12] | -18 [-18, -17] | 34 | 38 | 0.90 | 0.95 | 0.67 | 120 | 0.34 | 0.40 | 0.55 | 0.35 | 0.79 | 0.35 | 10.64 |
| UMAP 2D (15, 0.3), PCA init | 2 | 31 [30, 31] | 33 | -13 [-13, -12] | -17 [-18, -17] | 34 | 38 | 0.91 | 0.95 | 0.67 | 129 | 0.34 | 0.40 | 0.55 | 0.35 | 0.80 | 0.33 | 1.75 |
| UMAP 2D (10, 0.1) | 2 | 39 [39, 40] | 41 | -4 [-4, -3] | -9 [-9, -8] | 44 | 41 | 0.94 | 0.94 | 0.64 | 7 | 0.44 | 0.41 | 0.53 | 0.40 | 0.84 | 0.32 | 1.55 |
| UMAP 2D (5, 0.0) | 2 | 40 [40, 40] | 41 | -3 [-4, -3] | -8 [-8, -8] | 45 | 38 | 0.95 | 0.91 | 0.58 | 3.50 | 0.45 | 0.38 | 0.49 | 0.42 | 0.81 | 0.26 | 1.20 |
| t-SNE 2D | 2 | 43 [43, 44] | 44 | 0 [0, 0] | -5 [-5, -4] | 48 | 42 | 0.96 | 0.91 | 0.62 | 1 | 0.48 | 0.41 | 0.52 | 0.50 | 0.85 | 0.34 | 10.44 |
| t-SNE 2D, exag 4 | 2 | 19 [19, 20] | 21 | -24 [-25, -23] | -29 [-29, -28] | 22 | 30 | 0.86 | 0.95 | 0.72 | 180 | 0.22 | 0.35 | 0.56 | 0.31 | 0.67 | 0.31 | 2.17 |
| t-SNE 2D, uniform k=15 | 2 | 44 [43, 44] | 44 | 0 [-0, 1] | -5 [-5, -4] | 46 | 45 | 0.96 | 0.92 | 0.58 | 15 | 0.46 | 0.44 | 0.51 | 0.38 | 0.90 | 0.38 | 1.66 |
| PHATE 3D | 3 | 22 [22, 23] | 24 | -21 [-22, -20] | -26 [-26, -25] | 24 | 32 | 0.88 | 0.95 | 0.70 | 152 | 0.24 | 0.37 | 0.55 | 0.30 | 0.75 | 0.31 | 3.16 |
| PaCMAP 3D (15) | 3 | 36 [35, 36] | 38 | -7 [-8, -7] | -12 [-13, -12] | 38 | 44 | 0.95 | 0.96 | 0.73 | 83.50 | 0.38 | 0.46 | 0.60 | 0.39 | 0.85 | 0.37 | 0.30 |
| densMAP 3D (15, 0.3) | 3 | 37 [36, 37] | 39 | -7 [-7, -6] | -12 [-12, -11] | 40 | 44 | 0.93 | 0.97 | 0.88 | 141 | 0.40 | 0.47 | 0.65 | 0.46 | 0.82 | 0.35 | 4.03 |
| UMAP 3D (15, 0.3) | 3 | 41 [40, 42] | 43 | -2 [-3, -2] | -7 [-8, -7] | 44 | 45 | 0.95 | 0.97 | 0.73 | 11 | 0.44 | 0.47 | 0.61 | 0.45 | 0.88 | 0.39 | 1.78 |
| UMAP 3D (10, 0.1) | 3 | 44 [44, 45] | 45 | 1 [1, 2] | -4 [-4, -3] | 49 | 44 | 0.96 | 0.95 | 0.69 | 4 | 0.49 | 0.45 | 0.58 | 0.46 | 0.87 | 0.32 | 1.56 |
| t-SNE 3D | 3 | 48 [48, 49] | 49 | 5 [4, 5] | 0 [0, 0] | 53 | 47 | 0.98 | 0.94 | 0.70 | 1 | 0.53 | 0.47 | 0.58 | 0.56 | 0.86 | 0.38 | 3.58 |
| Random 2D | 2 | 1 [1, 1] | 1 | -42 [-43, -42] | -47 [-48, -47] | 1 | 3 | 0.50 | 0.50 | -0.01 | 95.50 | 0.01 | 0.05 | 0.20 | -0.00 | 0.14 | 0.00 | 0 |
| Full space | 15 | 100 [100, 100] | 100 | 57 [56, 57] | 52 [51, 52] | 100 | 100 | 1 | 1 | 1 |  |  |  |  |  | 0.92 | 1 | 0 |
| Savant 5 space (5-D) | 5 | 36 [35, 36] | 38 | -8 [-8, -7] | -12 [-13, -12] | 40 | 47 | 0.95 | 0.98 | 0.86 | 106.50 | 0.40 | 0.52 | 0.68 | 0.48 | 0.74 | 0.24 | 0 |

Chance for recall@10: 1.3%. Nearest true neighbour is the same pitcher's other season for 53% of rows (type mean); self rows are 10% of all-rows 10-NN sets, which is why the self-free score is the headline.

## A2. recall@10 self-free (%) by pitch type

| type | n | Movement | PCA 2 | PaCMAP 2D | UMAP 2D | UMAP 2D 10/0.1 | t-SNE 2D | PaCMAP 3D | UMAP 3D | UMAP 3D 10/0.1 | t-SNE 3D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CH | 1021 | 9 | 13 | 26 | 30 | 39 | 43 | 35 | 41 | 44 | 49 |
| CU | 789 | 12 | 15 | 33 | 35 | 43 | 45 | 41 | 45 | 47 | 50 |
| FC | 722 | 12 | 18 | 28 | 34 | 40 | 43 | 39 | 42 | 44 | 49 |
| FF | 1850 | 6 | 11 | 21 | 26 | 35 | 41 | 31 | 37 | 42 | 45 |
| FS | 277 | 20 | 28 | 41 | 44 | 48 | 49 | 46 | 53 | 51 | 55 |
| SI | 1235 | 9 | 13 | 23 | 27 | 38 | 42 | 33 | 39 | 44 | 47 |
| SL | 1253 | 9 | 12 | 27 | 30 | 39 | 44 | 37 | 41 | 45 | 49 |
| ST | 742 | 12 | 17 | 31 | 35 | 43 | 44 | 39 | 45 | 46 | 49 |

## A3. A linear picture: recall@10 self-free (%) by number of principal components

| index | 2 | 3 | 4 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| recall@10 % | 14 | 25 | 38 | 51 | 76 | 99 | 100 |

## A4. Truth sensitivity: recall@10 self-free (%) under other definitions of 'truly nearest'

Default: the 15 identity measures, block-weighted. The old atlas space is the 27 measures used before 2026-11-04 (adds location, scatter and the six near-duplicate columns); the maps are still fit on the identity space.

| truth | Movement | PCA 2 | PaCMAP 2D | UMAP 2D 10/0.1 | UMAP 2D | UMAP 3D | t-SNE 2D |
|---|---|---|---|---|---|---|---|
| block-weighted (default) | 10 | 14 | 26 | 39 | 30 | 41 | 43 |
| cosine on block-weighted | 8 | 10 | 24 | 37 | 28 | 38 | 40 |
| old atlas space (27: + location, scatter, duplicates) | 7 | 8 | 13 | 16 | 14 | 17 | 18 |
| plain z-score (no block weights) | 10 | 14 | 26 | 39 | 30 | 41 | 43 |

## A5. Neighbourhood mixing (10-NN), same-pitcher-other-season recovery, sample-size drift

| view | same hand % | chance % | same season % | chance season % | other-season top-10 % | n-drift Spearman |
|---|---|---|---|---|---|---|
| Full space | 69.21 | 60.83 | 30.46 | 33.40 | 88.89 | -0.01 |
| Movement | 61.99 | 60.83 | 32.70 | 33.40 | 21.85 | -0.00 |
| PCA 2 | 62.66 | 60.83 | 32.49 | 33.40 | 27.81 | -0.00 |
| PaCMAP 2D | 65.15 | 60.83 | 31.71 | 33.40 | 49.62 | 0.01 |
| UMAP 2D | 65.95 | 60.83 | 31.02 | 33.40 | 61.26 | 0.02 |
| UMAP 2D 10/0.1 | 67.09 | 60.83 | 30.53 | 33.40 | 72.55 | 0.03 |
| t-SNE 2D | 67.58 | 60.83 | 30.56 | 33.40 | 79.55 | 0.01 |
| PaCMAP 3D | 66.41 | 60.83 | 31.31 | 33.40 | 59.64 | 0.01 |
| UMAP 3D | 67.18 | 60.83 | 30.77 | 33.40 | 72.16 | 0.03 |
| UMAP 3D 10/0.1 | 67.68 | 60.83 | 30.59 | 33.40 | 76.82 | 0.04 |
| t-SNE 3D | 67.86 | 60.83 | 30.48 | 33.40 | 81.03 | 0.01 |

## A6. Same pitcher, other season, within 10 nearest (%) by pitch-count quartile of the query row

| index | n quartile 1 | n quartile 2 | n quartile 3 | n quartile 4 |
|---|---|---|---|---|
| Full space | 85 | 88 | 89 | 93 |
| Movement | 17 | 19 | 22 | 27 |
| PCA 2 | 23 | 25 | 28 | 32 |
| PaCMAP 2D | 41 | 45 | 50 | 58 |
| PaCMAP 3D | 51 | 56 | 61 | 66 |
| UMAP 2D 10/0.1 | 64 | 70 | 74 | 79 |
| UMAP 2D | 53 | 59 | 62 | 68 |
| UMAP 3D 10/0.1 | 69 | 74 | 77 | 83 |
| UMAP 3D | 65 | 68 | 74 | 78 |
| t-SNE 2D | 73 | 77 | 80 | 86 |
| t-SNE 3D | 75 | 79 | 81 | 87 |

## B1. Refresh-style held-out test: second half placed on a map of first halves

Own first half among the 10 nearest embedded first halves (%, CI); top-1; median rank. Chance for top-10 is 1.7%.

| view | placement | top-10 % [CI] | top-1 % | median rank | pairs |
|---|---|---|---|---|---|
| Full space | as is | 96 [96, 97] | 72.65 | 1 | 4293 |
| Movement | as is | 31 [29, 32] | 4.80 | 24 | 4293 |
| PCA 2 | as is | 41 [39, 43] | 8.39 | 15 | 4293 |
| UMAP 2D | transform | 59 [57, 61] | 12.23 | 7 | 4293 |
| UMAP 2D | IDW 15, other pitchers | 28 [26, 29] | 2.89 | 24 | 4293 |
| UMAP 2D | IDW 15, all rows | 43 [42, 45] | 6.64 | 14 | 4293 |
| UMAP 2D 10/0.1 | transform | 73 [71, 75] | 19.52 | 4 | 4293 |
| UMAP 2D 10/0.1 | IDW 15, other pitchers | 24 [23, 26] | 2.75 | 28 | 4293 |
| UMAP 2D 10/0.1 | IDW 15, all rows | 40 [38, 41] | 5.57 | 15 | 4293 |
| UMAP 3D | transform | 75 [73, 77] | 23.95 | 4 | 4293 |
| UMAP 3D | IDW 15, other pitchers | 50 [48, 51] | 6.92 | 11 | 4293 |
| UMAP 3D | IDW 15, all rows | 67 [65, 68] | 16.14 | 5 | 4293 |
| UMAP 3D 10/0.1 | transform | 81 [79, 82] | 28.84 | 3 | 4293 |
| UMAP 3D 10/0.1 | IDW 15, other pitchers | 41 [39, 43] | 5.17 | 15 | 4293 |
| UMAP 3D 10/0.1 | IDW 15, all rows | 60 [58, 62] | 11.18 | 7 | 4293 |
| t-SNE 2D | transform | 83 [81, 84] | 62.54 | 1 | 4293 |
| t-SNE 2D | IDW 15, other pitchers | 20 [19, 22] | 2.31 | 40 | 4293 |
| t-SNE 2D | IDW 15, all rows | 33 [32, 35] | 4.87 | 22 | 4293 |
| t-SNE 2D ex4 | transform | 37 [36, 39] | 4.59 | 15 | 4293 |
| t-SNE 2D ex4 | IDW 15, other pitchers | 21 [20, 23] | 1.86 | 28 | 4293 |
| t-SNE 2D ex4 | IDW 15, all rows | 38 [37, 40] | 3.84 | 16 | 4293 |
| t-SNE 3D | transform | 88 [87, 89] | 69.51 | 1 | 4293 |
| t-SNE 3D | IDW 15, other pitchers | 38 [37, 40] | 5.31 | 17 | 4293 |
| t-SNE 3D | IDW 15, all rows | 58 [56, 60] | 12.42 | 7 | 4293 |
| PaCMAP 2D | transform | 44 [42, 45] | 5.10 | 13 | 4293 |
| PaCMAP 2D | IDW 15, other pitchers | 25 [24, 27] | 2.59 | 26 | 4293 |
| PaCMAP 2D | IDW 15, all rows | 40 [38, 42] | 4.98 | 15 | 4293 |
| PaCMAP 3D | transform | 60 [58, 61] | 11.27 | 7 | 4293 |
| PaCMAP 3D | IDW 15, other pitchers | 44 [42, 45] | 5.96 | 13 | 4293 |
| PaCMAP 3D | IDW 15, all rows | 60 [59, 62] | 12.65 | 7 | 4293 |

## B2. Stranger-style held-out test (pitcher folds, both halves placed on a map fit without that pitcher)

| view | placement | top-10 % [CI] | fold range % | median rank |
|---|---|---|---|---|
| Full space | as is | 96 [96, 97] | 96-97 | 1 |
| Movement | as is | 31 [29, 32] | 28-33 | 24 |
| PCA 2 | as is | 41 [39, 43] | 39-42 | 16 |
| UMAP 2D | transform | 39 [37, 41] | 38-41 | 18 |
| UMAP 2D | IDW 15 | 42 [40, 44] | 41-44 | 15 |
| UMAP 3D | transform | 57 [55, 58] | 54-60 | 7 |
| UMAP 3D | IDW 15 | 64 [62, 66] | 62-66 | 5 |
| t-SNE 2D | transform | 50 [49, 52] | 49-53 | 10 |
| t-SNE 2D | IDW 15 | 42 [40, 43] | 41-43 | 15 |
| t-SNE 3D | transform | 54 [53, 56] | 51-59 | 6 |
| t-SNE 3D | IDW 15 | 62 [60, 64] | 60-64 | 6 |
| PaCMAP 3D | transform | 61 [59, 63] | 59-62 | 6 |
| PaCMAP 3D | IDW 15 | 62 [60, 63] | 59-63 | 6 |
| UMAP 2D 10/0.1 | transform | 45 [43, 47] | 44-46 | 14 |
| UMAP 2D 10/0.1 | IDW 15 | 41 [40, 43] | 40-43 | 15 |
| UMAP 3D 10/0.1 | transform | 54 [52, 56] | 51-56 | 9 |
| UMAP 3D 10/0.1 | IDW 15 | 62 [60, 64] | 60-63 | 5 |
| PaCMAP 2D | transform | 47 [45, 49] | 45-48 | 12 |
| PaCMAP 2D | IDW 15 | 42 [40, 44] | 39-43 | 15 |

## B3. Refresh-style top-10 (%) by pitch-count quartile (second-half pitch count), transform placement

| index | n quartile 1 | n quartile 2 | n quartile 3 | n quartile 4 |
|---|---|---|---|---|
| Full space | 96 | 95 | 96 | 98 |
| Movement | 29 | 28 | 32 | 34 |
| PCA 2 | 38 | 40 | 43 | 41 |
| PaCMAP 2D | 40 | 43 | 47 | 44 |
| PaCMAP 3D | 55 | 59 | 62 | 62 |
| UMAP 2D 10/0.1 | 70 | 69 | 76 | 78 |
| UMAP 2D | 58 | 55 | 62 | 60 |
| UMAP 3D 10/0.1 | 78 | 78 | 82 | 86 |
| UMAP 3D | 72 | 72 | 78 | 79 |
| t-SNE 2D | 82 | 80 | 83 | 85 |
| t-SNE 2D ex4 | 37 | 37 | 38 | 37 |
| t-SNE 3D | 87 | 86 | 88 | 91 |

## C1. Outcome signal in neighbourhoods: Spearman between a pitch's outcome and the mean of its 30 nearest other-pitcher dots

| view | whiff_pct | csw_pct | xwoba_contact | rv100_raw | rv100 |
|---|---|---|---|---|---|
| Half-to-half reliability r (pitchers with 40+ pitches in both halves) | 0.46 | 0.42 | 0.25 | 0.10 | 0.15 |
| Ceiling for a season outcome, sqrt(2r / (1 + r)) (Spearman-Brown) | 0.80 | 0.77 | 0.64 | 0.42 | 0.52 |
| Full space | 0.23 [0.21, 0.26] | 0.18 [0.16, 0.21] | 0.11 [0.09, 0.13] | 0.07 [0.04, 0.09] | 0.08 [0.06, 0.10] |
| Savant-5 (5-D) | 0.20 [0.17, 0.23] | 0.13 [0.11, 0.16] | 0.10 [0.07, 0.12] | 0.08 [0.05, 0.10] | 0.09 [0.07, 0.11] |
| Movement | 0.08 [0.06, 0.11] | 0.08 [0.06, 0.11] | 0.03 [0.01, 0.05] | -0.01 [-0.03, 0.01] | -0.01 [-0.03, 0.01] |
| PCA 2 | 0.09 [0.07, 0.12] | 0.10 [0.07, 0.12] | 0.06 [0.04, 0.09] | 0.02 [-0.01, 0.04] | 0.03 [0.01, 0.06] |
| t-SNE 2D | 0.15 [0.13, 0.18] | 0.13 [0.11, 0.16] | 0.10 [0.07, 0.12] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.08] |
| t-SNE 3D | 0.17 [0.14, 0.19] | 0.14 [0.11, 0.16] | 0.10 [0.07, 0.12] | 0.06 [0.04, 0.09] | 0.07 [0.05, 0.10] |
| PaCMAP 2D | 0.15 [0.12, 0.18] | 0.15 [0.12, 0.17] | 0.07 [0.05, 0.10] | 0.02 [0.00, 0.05] | 0.05 [0.02, 0.07] |
| PaCMAP 3D | 0.18 [0.15, 0.20] | 0.16 [0.13, 0.18] | 0.09 [0.07, 0.12] | 0.05 [0.02, 0.07] | 0.06 [0.04, 0.08] |
| PHATE 3D | 0.14 [0.12, 0.17] | 0.11 [0.09, 0.13] | 0.06 [0.04, 0.09] | 0.01 [-0.01, 0.03] | 0.02 [-0.00, 0.04] |
| UMAP 2D | 0.14 [0.12, 0.17] | 0.12 [0.10, 0.15] | 0.07 [0.05, 0.10] | 0.03 [0.01, 0.06] | 0.05 [0.03, 0.08] |
| UMAP 2D 10/0.1 | 0.18 [0.15, 0.20] | 0.13 [0.11, 0.16] | 0.09 [0.07, 0.12] | 0.04 [0.02, 0.06] | 0.06 [0.04, 0.08] |
| UMAP 3D | 0.17 [0.15, 0.20] | 0.16 [0.14, 0.18] | 0.10 [0.07, 0.12] | 0.04 [0.02, 0.07] | 0.06 [0.03, 0.08] |
| UMAP 3D 10/0.1 | 0.17 [0.14, 0.19] | 0.14 [0.11, 0.16] | 0.09 [0.06, 0.11] | 0.04 [0.02, 0.07] | 0.06 [0.03, 0.08] |
| Random | -0.02 [-0.04, 0.00] | 0.00 [-0.02, 0.02] | -0.00 [-0.02, 0.02] | -0.01 [-0.03, 0.01] | -0.02 [-0.04, 0.01] |

## C2. Whiff-rate Spearman by neighbourhood size k

| view | k=10 | k=30 | k=100 |
|---|---|---|---|
| Full space | 0.21 | 0.23 | 0.26 |
| Movement | 0.08 | 0.08 | 0.09 |
| PCA 2 | 0.08 | 0.09 | 0.11 |
| t-SNE 2D | 0.14 | 0.15 | 0.13 |
| UMAP 2D | 0.13 | 0.14 | 0.13 |
| UMAP 3D | 0.15 | 0.17 | 0.17 |
| PaCMAP 3D | 0.15 | 0.18 | 0.17 |

## C3. Location added back (identity + location, zone rate) and the old 27-measure space: k = 30

| view | whiff_pct | csw_pct | xwoba_contact | rv100_raw |
|---|---|---|---|---|
| Full space | 0.23 | 0.18 | 0.11 | 0.07 |
| Full space, + location | 0.39 | 0.32 | 0.16 | 0.11 |
| Full space, old 27 measures | 0.39 | 0.32 | 0.16 | 0.08 |
| t-SNE 3D | 0.17 | 0.14 | 0.10 | 0.06 |
| t-SNE 3D, + location | 0.32 | 0.25 | 0.13 | 0.05 |

## C4. R² of the 30-neighbour mean as a predictor (weighted)

| view | whiff_pct | csw_pct | xwoba_contact | rv100_raw | rv100 |
|---|---|---|---|---|---|
| Full space | 0.05 | 0.02 | -0.00 | -0.02 | -0.01 |
| Movement | -0.00 | -0.01 | -0.03 | -0.04 | -0.04 |
| PCA 2 | -0.00 | -0.01 | -0.02 | -0.03 | -0.03 |
| t-SNE 2D | 0.02 | -0.00 | -0.01 | -0.03 | -0.02 |
| UMAP 2D | 0.02 | -0.01 | -0.02 | -0.03 | -0.02 |
| UMAP 3D | 0.03 | 0.02 | -0.01 | -0.02 | -0.01 |
| PaCMAP 3D | 0.03 | 0.01 | -0.01 | -0.02 | -0.01 |
| Random | -0.04 | -0.04 | -0.03 | -0.04 | -0.04 |

## D1. Seed stability (5 seeds): neighbourhood overlap between runs, Procrustes disparity, island share, recall range

| method | dims | overlap@10 | overlap@50 | Procrustes | island share | recall@10 min-max % |
|---|---|---|---|---|---|---|
| PaCMAP (15) | 2 | 0.47 | 0.69 | 0.08 | 0.71 | 26-27 |
| PaCMAP (15) | 3 | 0.70 | 0.83 | 0.02 | 0.79 | 35-36 |
| UMAP (10, 0.1) | 2 | 0.70 | 0.79 | 0.03 | 0.77 | 39-40 |
| UMAP (10, 0.1) | 3 | 0.82 | 0.87 | 0.02 | 0.81 | 44-45 |
| UMAP (15, 0.3) | 2 | 0.61 | 0.79 | 0.02 | 0.72 | 30-31 |
| UMAP (15, 0.3) | 3 | 0.81 | 0.89 | 0.01 | 0.80 | 41-41 |
| t-SNE (exag 4) | 2 | 0.99 | 1.00 | 0.00 | 0.60 | 19-19 |
| t-SNE (exag 4) | 3 | 1.00 | 1.00 | 0.00 | 0.72 | 28-29 |
| t-SNE (perp 30) | 2 | 0.59 | 0.51 | 0.37 | 0.77 | 43-44 |
| t-SNE (perp 30) | 3 | 0.77 | 0.70 | 0.24 | 0.80 | 47-48 |

## D2. Refresh: layout fit on first-half profiles, refit on full-season profiles (common rows compared)

Reference (not a bound): the full space's own neighbourhoods are 77% (k = 10) and 84% (k = 50) the same between the two profile tables (71% at k = 50 in the old 27-measure space); a map above it is suppressing real change. Single seeded fit per type; weighted over types.

| method | dims | start | overlap@10 | overlap@50 | Procrustes | island share | recall@10 % after refit |
|---|---|---|---|---|---|---|---|
| PaCMAP (15) | 2 | cold | 0.32 | 0.55 | 0.18 | 0.69 | 26.36 |
| PaCMAP (15) | 2 | warm (500 epochs) | 0.33 | 0.55 | 0.19 | 0.70 | 26.45 |
| PaCMAP (15) | 3 | cold | 0.46 | 0.64 | 0.11 | 0.78 | 35.70 |
| PaCMAP (15) | 3 | warm (500 epochs) | 0.46 | 0.64 | 0.11 | 0.78 | 35.77 |
| UMAP (10, 0.1) | 2 | aligned reg 0.01 | 0.43 | 0.56 | 0.09 | 0.74 | 36.87 |
| UMAP (10, 0.1) | 2 | aligned reg 0.1 | 0.54 | 0.74 | 0.02 | 0.72 | 35.06 |
| UMAP (10, 0.1) | 2 | aligned reg 1.0 | 0.79 | 0.91 | 0.00 | 0.71 | 34.39 |
| UMAP (10, 0.1) | 2 | cold | 0.42 | 0.50 | 0.17 | 0.76 | 39.21 |
| UMAP (10, 0.1) | 2 | warm 100 epochs | 0.40 | 0.49 | 0.16 | 0.74 | 37.00 |
| UMAP (10, 0.1) | 2 | warm 100 epochs, lr 0.5 | 0.41 | 0.51 | 0.13 | 0.75 | 38.14 |
| UMAP (10, 0.1) | 2 | warm 30 epochs | 0.36 | 0.50 | 0.12 | 0.70 | 32.17 |
| UMAP (10, 0.1) | 2 | warm 500 epochs | 0.42 | 0.50 | 0.17 | 0.76 | 39.24 |
| UMAP (10, 0.1) | 3 | aligned reg 0.01 | 0.52 | 0.61 | 0.08 | 0.79 | 43.25 |
| UMAP (10, 0.1) | 3 | aligned reg 0.1 | 0.61 | 0.75 | 0.03 | 0.78 | 42.17 |
| UMAP (10, 0.1) | 3 | aligned reg 1.0 | 0.82 | 0.91 | 0.00 | 0.76 | 40.39 |
| UMAP (10, 0.1) | 3 | cold | 0.51 | 0.58 | 0.13 | 0.79 | 44.24 |
| UMAP (10, 0.1) | 3 | warm 100 epochs | 0.50 | 0.58 | 0.12 | 0.79 | 42.77 |
| UMAP (10, 0.1) | 3 | warm 100 epochs, lr 0.5 | 0.51 | 0.58 | 0.11 | 0.79 | 43.68 |
| UMAP (10, 0.1) | 3 | warm 30 epochs | 0.47 | 0.57 | 0.11 | 0.77 | 39.61 |
| UMAP (10, 0.1) | 3 | warm 500 epochs | 0.52 | 0.58 | 0.13 | 0.80 | 44.31 |
| UMAP (15, 0.3) | 2 | aligned reg 0.01 | 0.40 | 0.58 | 0.07 | 0.74 | 34.32 |
| UMAP (15, 0.3) | 2 | aligned reg 0.1 | 0.52 | 0.73 | 0.02 | 0.72 | 32.24 |
| UMAP (15, 0.3) | 2 | aligned reg 1.0 | 0.78 | 0.90 | 0.00 | 0.70 | 31.57 |
| UMAP (15, 0.3) | 2 | cold | 0.36 | 0.55 | 0.10 | 0.71 | 30.41 |
| UMAP (15, 0.3) | 2 | warm 100 epochs | 0.35 | 0.55 | 0.10 | 0.70 | 28.28 |
| UMAP (15, 0.3) | 2 | warm 100 epochs, lr 0.5 | 0.36 | 0.56 | 0.08 | 0.71 | 29.55 |
| UMAP (15, 0.3) | 2 | warm 30 epochs | 0.30 | 0.52 | 0.09 | 0.67 | 25.34 |
| UMAP (15, 0.3) | 2 | warm 500 epochs | 0.37 | 0.55 | 0.10 | 0.71 | 30.25 |
| UMAP (15, 0.3) | 3 | aligned reg 0.01 | 0.52 | 0.65 | 0.06 | 0.81 | 41.71 |
| UMAP (15, 0.3) | 3 | aligned reg 0.1 | 0.61 | 0.76 | 0.03 | 0.80 | 40.66 |
| UMAP (15, 0.3) | 3 | aligned reg 1.0 | 0.82 | 0.91 | 0.00 | 0.77 | 39.08 |
| UMAP (15, 0.3) | 3 | cold | 0.51 | 0.65 | 0.07 | 0.79 | 40.93 |
| UMAP (15, 0.3) | 3 | warm 100 epochs | 0.49 | 0.65 | 0.07 | 0.79 | 39.43 |
| UMAP (15, 0.3) | 3 | warm 100 epochs, lr 0.5 | 0.50 | 0.65 | 0.07 | 0.79 | 40.21 |
| UMAP (15, 0.3) | 3 | warm 30 epochs | 0.46 | 0.62 | 0.07 | 0.77 | 36.75 |
| UMAP (15, 0.3) | 3 | warm 500 epochs | 0.51 | 0.65 | 0.07 | 0.79 | 40.89 |
| t-SNE (exag 4) | 2 | cold | 0.45 | 0.70 | 0.03 | 0.59 | 19.20 |
| t-SNE (exag 4) | 2 | warm (500 epochs) | 0.45 | 0.70 | 0.03 | 0.59 | 19.19 |
| t-SNE (exag 4) | 3 | cold | 0.53 | 0.72 | 0.03 | 0.72 | 28.48 |
| t-SNE (exag 4) | 3 | warm (500 epochs) | 0.53 | 0.72 | 0.03 | 0.72 | 28.47 |
| t-SNE (perp 30) | 2 | cold | 0.45 | 0.43 | 0.38 | 0.76 | 43.21 |
| t-SNE (perp 30) | 2 | warm 100 iters | 0.64 | 0.77 | 0.02 | 0.73 | 39.93 |
| t-SNE (perp 30) | 2 | warm 250 iters | 0.60 | 0.70 | 0.04 | 0.73 | 41.34 |
| t-SNE (perp 30) | 2 | warm 500 iters | 0.58 | 0.67 | 0.05 | 0.74 | 42.16 |
| t-SNE (perp 30) | 3 | cold | 0.54 | 0.55 | 0.32 | 0.79 | 48.06 |
| t-SNE (perp 30) | 3 | warm 100 iters | 0.63 | 0.72 | 0.04 | 0.78 | 46.16 |
| t-SNE (perp 30) | 3 | warm 250 iters | 0.60 | 0.66 | 0.07 | 0.79 | 47.41 |
| t-SNE (perp 30) | 3 | warm 500 iters | 0.59 | 0.63 | 0.09 | 0.79 | 47.65 |

## E1. UMAP 2D grid: recall@10 self-free % / distance corr / continuity / island share (rows n_neighbors, columns min_dist)

recall@10 %:

| n_neighbors | 0.0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| 5 | 40 | 40 | 37 | 32 |
| 10 | 41 | 39 | 34 | 29 |
| 15 | 39 | 37 | 30 | 25 |
| 30 | 34 | 31 | 25 | 22 |
| 50 | 30 | 28 | 23 | 20 |

distance corr:

| n_neighbors | 0.0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| 5 | 0.58 | 0.58 | 0.61 | 0.64 |
| 10 | 0.62 | 0.64 | 0.66 | 0.68 |
| 15 | 0.63 | 0.64 | 0.67 | 0.69 |
| 30 | 0.64 | 0.66 | 0.68 | 0.69 |
| 50 | 0.65 | 0.67 | 0.69 | 0.70 |

continuity:

| n_neighbors | 0.0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| 5 | 0.91 | 0.91 | 0.92 | 0.93 |
| 10 | 0.93 | 0.94 | 0.94 | 0.95 |
| 15 | 0.94 | 0.94 | 0.95 | 0.95 |
| 30 | 0.94 | 0.94 | 0.95 | 0.95 |
| 50 | 0.94 | 0.94 | 0.95 | 0.95 |

island share:

| n_neighbors | 0.0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| 5 | 0.74 | 0.74 | 0.72 | 0.67 |
| 10 | 0.79 | 0.77 | 0.73 | 0.68 |
| 15 | 0.80 | 0.78 | 0.72 | 0.68 |
| 30 | 0.77 | 0.74 | 0.69 | 0.65 |
| 50 | 0.75 | 0.71 | 0.66 | 0.63 |

## E1. UMAP 3D grid: recall@10 self-free % / distance corr / continuity / island share (rows n_neighbors, columns min_dist)

recall@10 %:

| n_neighbors | 0.0 | 0.1 | 0.3 |
|---|---|---|---|
| 5 | 41 | 42 | 43 |
| 10 | 44 | 44 | 43 |
| 15 | 43 | 43 | 41 |
| 30 | 40 | 39 | 37 |

distance corr:

| n_neighbors | 0.0 | 0.1 | 0.3 |
|---|---|---|---|
| 5 | 0.62 | 0.64 | 0.67 |
| 10 | 0.66 | 0.69 | 0.72 |
| 15 | 0.68 | 0.70 | 0.73 |
| 30 | 0.69 | 0.71 | 0.73 |

continuity:

| n_neighbors | 0.0 | 0.1 | 0.3 |
|---|---|---|---|
| 5 | 0.92 | 0.93 | 0.94 |
| 10 | 0.95 | 0.95 | 0.97 |
| 15 | 0.95 | 0.96 | 0.97 |
| 30 | 0.96 | 0.96 | 0.97 |

island share:

| n_neighbors | 0.0 | 0.1 | 0.3 |
|---|---|---|---|
| 5 | 0.75 | 0.75 | 0.76 |
| 10 | 0.81 | 0.81 | 0.79 |
| 15 | 0.83 | 0.82 | 0.80 |
| 30 | 0.83 | 0.82 | 0.79 |

## E3. PaCMAP 2D grid: recall@10 self-free % / distance corr / continuity / island share (rows n_neighbors, columns MN_ratio)

recall@10 %:

| n_neighbors | 0.5 | 1.0 |
|---|---|---|
| 10 | 30 | 30 |
| 15 | 26 | 26 |
| 30 | 21 | 21 |

distance corr:

| n_neighbors | 0.5 | 1.0 |
|---|---|---|
| 10 | 0.65 | 0.64 |
| 15 | 0.67 | 0.67 |
| 30 | 0.70 | 0.70 |

continuity:

| n_neighbors | 0.5 | 1.0 |
|---|---|---|
| 10 | 0.94 | 0.94 |
| 15 | 0.94 | 0.94 |
| 30 | 0.94 | 0.94 |

island share:

| n_neighbors | 0.5 | 1.0 |
|---|---|---|
| 10 | 0.70 | 0.70 |
| 15 | 0.70 | 0.70 |
| 30 | 0.66 | 0.66 |

## E4. t-SNE 2D grid: recall@10 self-free % / distance corr / continuity / island share (rows perplexity, columns exaggeration)

recall@10 %:

| perplexity | 1.0 | 2.0 | 4.0 |
|---|---|---|---|
| 10 | 43 | 44 | 26 |
| 30 | 43 | 27 | 19 |
| 50 | 43 | 22 | 18 |

distance corr:

| perplexity | 1.0 | 2.0 | 4.0 |
|---|---|---|---|
| 10 | 0.66 | 0.64 | 0.71 |
| 30 | 0.62 | 0.70 | 0.72 |
| 50 | 0.65 | 0.73 | 0.71 |

continuity:

| perplexity | 1.0 | 2.0 | 4.0 |
|---|---|---|---|
| 10 | 0.93 | 0.93 | 0.95 |
| 30 | 0.91 | 0.94 | 0.95 |
| 50 | 0.92 | 0.95 | 0.95 |

island share:

| perplexity | 1.0 | 2.0 | 4.0 |
|---|---|---|---|
| 10 | 0.75 | 0.78 | 0.66 |
| 30 | 0.77 | 0.71 | 0.60 |
| 50 | 0.77 | 0.64 | 0.58 |

## E3b. PaCMAP 3D grid

| n_neighbors | b | recall@10 % | dist corr | continuity | island share |
|---|---|---|---|---|---|
| 10 | 0.50 | 39.01 | 0.71 | 0.96 | 0.78 |
| 15 | 0.50 | 35.72 | 0.73 | 0.96 | 0.79 |
| 30 | 0.50 | 30.01 | 0.75 | 0.96 | 0.76 |

## E4b. t-SNE 3D grid (columns: exaggeration)

| perplexity | b | recall@10 % | dist corr | continuity | island share |
|---|---|---|---|---|---|
| 10 | 1 | 45.16 | 0.67 | 0.93 | 0.76 |
| 10 | 2 | 44.83 | 0.68 | 0.94 | 0.78 |
| 30 | 1 | 48.07 | 0.70 | 0.94 | 0.80 |
| 30 | 2 | 38.57 | 0.74 | 0.96 | 0.80 |
| 50 | 1 | 47.77 | 0.73 | 0.95 | 0.80 |
| 50 | 2 | 33.84 | 0.76 | 0.97 | 0.77 |

## E5. Output dimensions

| method | dims | recall@10 % | trust | continuity | dist corr |
|---|---|---|---|---|---|
| UMAP (10, 0.1) dims | 2 | 39.23 | 0.94 | 0.94 | 0.64 |
| UMAP (10, 0.1) dims | 3 | 44.25 | 0.96 | 0.95 | 0.69 |
| UMAP (10, 0.1) dims | 4 | 46.63 | 0.97 | 0.96 | 0.71 |
| UMAP (10, 0.1) dims | 5 | 47.21 | 0.97 | 0.96 | 0.71 |
| UMAP (15, 0.3) dims | 2 | 30.43 | 0.90 | 0.95 | 0.67 |
| UMAP (15, 0.3) dims | 3 | 40.94 | 0.95 | 0.97 | 0.73 |
| UMAP (15, 0.3) dims | 4 | 47.26 | 0.97 | 0.98 | 0.75 |
| UMAP (15, 0.3) dims | 5 | 50.26 | 0.98 | 0.98 | 0.76 |

## F. Feature pairs with |r| >= 0.8 within a pitch type (count of types, mean |r|)

| a | b | types | mean_abs_r |
|---|---|---|---|
| pfx_z | ddz_delta_az | 4 | 0.94 |
| pfx_x | ddz_delta_ax | 7 | 0.88 |
| spin_sin | spin_cos | 3 | 0.87 |
| vra_above_avg | release_pos_z | 6 | 0.87 |
| spin_cos | arm_angle | 2 | 0.86 |
| vra_above_avg | arm_angle | 2 | 0.86 |
| spin_cos | pfx_z | 2 | 0.85 |
| pfx_z | vra_above_avg | 1 | 0.85 |
| spin_sin | pfx_x | 1 | 0.84 |
| spin_cos | vra_above_avg | 1 | 0.84 |
| spin_sin | arm_angle | 1 | 0.83 |
| arm_angle | release_pos_z | 3 | 0.82 |
| vaa_above_avg | release_pos_z | 2 | 0.81 |

## G1. The All-pitches map (global scaling, 7,889 dots)

| view | dims | recall@10 self-free % [CI] | type purity % | type kNN acc % | trust | continuity | dist corr | fit s |
|---|---|---|---|---|---|---|---|---|
| Movement | 2 | 4 [4, 5] | 70.75 | 77.13 | 0.81 | 0.96 | 0.58 | 0 |
| PCA 2 | 2 | 5 [4, 5] | 36.34 | 45.23 | 0.84 | 0.96 | 0.80 | 0 |
| t-SNE 2D | 2 | 45 [44, 45] | 90.53 | 93.47 | 0.99 | 0.98 | 0.54 | 39.36 |
| t-SNE 2D ex4 | 2 | 17 [17, 18] | 84.14 | 88.07 | 0.95 | 0.99 | 0.50 | 32.46 |
| t-SNE 3D | 3 | 47 [47, 48] | 90.89 | 93.17 | 1.00 | 0.99 | 0.50 | 62.77 |
| UMAP 2D | 2 | 20 [19, 20] | 87.58 | 91.29 | 0.95 | 0.99 | 0.48 | 27.92 |
| UMAP 2D 10/0.1 | 2 | 27 [26, 28] | 88.65 | 91.86 | 0.96 | 0.99 | 0.46 | 10.19 |
| UMAP 3D | 3 | 34 [34, 35] | 88.65 | 92.36 | 0.98 | 0.99 | 0.50 | 11.12 |
| UMAP 3D 10/0.1 | 3 | 38 [38, 39] | 89.51 | 92.71 | 0.99 | 0.99 | 0.43 | 9.15 |
| PaCMAP 2D | 2 | 16 [16, 17] | 83.09 | 87.36 | 0.94 | 0.99 | 0.47 | 2.50 |
| PaCMAP 3D | 3 | 29 [28, 29] | 87.63 | 90.81 | 0.98 | 0.99 | 0.46 | 3.66 |
| Random | 2 | 0 [0, 0] | 14.85 | 17.23 | 0.50 | 0.50 | -0.00 | 0 |
| Full space | 15 | 100 [100, 100] | 90.70 | 95.22 | 1 | 1 | 1 | 0 |

## G2. All-pitches map, refresh-style held-out placement (own first half among 10 nearest, %)

| view | placement | top-10 % [CI] | median rank |
|---|---|---|---|
| Full space | as is | 95 [94, 96] | 1 |
| Movement | as is | 22 [21, 23] | 40 |
| PCA 2 | as is | 19 [18, 21] | 48 |
| UMAP 2D | transform | 44 [42, 46] | 13 |
| UMAP 2D | IDW 15, other pitchers | 24 [22, 25] | 29 |
| UMAP 3D | transform | 68 [66, 70] | 5 |
| UMAP 3D | IDW 15, other pitchers | 46 [45, 48] | 12 |
| UMAP 2D 10/0.1 | transform | 57 [55, 59] | 8 |
| UMAP 2D 10/0.1 | IDW 15, other pitchers | 21 [20, 22] | 34 |
| t-SNE 2D | transform | 85 [84, 87] | 1 |
| t-SNE 2D | IDW 15, other pitchers | 21 [20, 23] | 37 |
| t-SNE 3D | transform | 87 [85, 88] | 1 |
| t-SNE 3D | IDW 15, other pitchers | 36 [34, 38] | 20 |

## G3. All-pitches map, refresh stability (cold vs warm start)

| method | dims | test | n | overlap10 | overlap50 | procrustes | type_purity |
|---|---|---|---|---|---|---|---|
| UMAP 2D (15, 0.3) | 2 | refresh cold | 6282 | 0.29 | 0.55 | 0.03 | 0.88 |
| UMAP 2D (15, 0.3) | 2 | refresh warm | 6282 | 0.29 | 0.55 | 0.01 | 0.87 |
| UMAP 3D (15, 0.3) | 3 | refresh cold | 6282 | 0.47 | 0.65 | 0.05 | 0.88 |
| UMAP 3D (15, 0.3) | 3 | refresh warm | 6282 | 0.48 | 0.66 | 0.02 | 0.88 |
| UMAP 2D (10, 0.1) | 2 | refresh cold | 6282 | 0.30 | 0.48 | 0.02 | 0.89 |
| UMAP 2D (10, 0.1) | 2 | refresh warm | 6282 | 0.31 | 0.51 | 0.01 | 0.89 |
| t-SNE 2D | 2 | refresh cold | 6282 | 0.51 | 0.50 | 0.04 | 0.90 |
| t-SNE 2D | 2 | refresh warm | 6282 | 0.61 | 0.69 | 0.01 | 0.90 |
| t-SNE 3D | 3 | refresh cold | 6282 | 0.56 | 0.55 | 0.11 | 0.91 |
| t-SNE 3D | 3 | refresh warm | 6282 | 0.59 | 0.62 | 0.02 | 0.91 |

## Software

| index | version |
|---|---|
| umap-learn | 0.5.12 |
| openTSNE | 1.0.4 |
| pacmap | 0.9.1 |
| phate | 2.0.0 |
| trimap | 1.2.0 |
| scikit-learn | 1.9.0 |
| numpy | 2.3.1 |

## L1. Same pitch, different location: Spearman between his gap to his 30 nearest lookalikes in location and in results

Signed, within pitch type. Height: + higher. Whiff and CSW in points. Identity space = the atlas's new definition of similar; old atlas space = with location and scatter in it. 95% pitcher-clustered bootstrap intervals for the identity space.

| type | n | height~whiff, identity | height~whiff, old | height~whiff, t-SNE 3D | height~whiff, movement | identity CI | zone~CSW, identity | inside~whiff, identity | release scatter~whiff, identity | whiff pts per inch higher | SD of height gap, in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CH | 1021 | -0.33 | -0.15 | -0.34 | -0.31 | [-0.39, -0.27] | 0.27 | 0.02 | -0.06 | -1.42 | 2.35 |
| CU | 789 | -0.41 | -0.16 | -0.41 | -0.43 | [-0.47, -0.35] | 0.35 | 0.04 | -0.04 | -1.45 | 2.89 |
| FC | 722 | -0.11 | -0.05 | -0.13 | -0.09 | [-0.20, -0.02] | 0.30 | -0.10 | -0.04 | -0.39 | 3.25 |
| FF | 1850 | 0.33 | 0.16 | 0.31 | 0.27 | [0.29, 0.38] | 0.36 | -0.07 | -0 | 0.95 | 2.26 |
| FS | 277 | -0.32 | -0.15 | -0.34 | -0.37 | [-0.44, -0.18] | 0.21 | 0.14 | 0.14 | -1.17 | 2.61 |
| SI | 1235 | -0.07 | 0.03 | -0.08 | -0.05 | [-0.14, -0.01] | 0.39 | -0 | -0 | -0.18 | 2.28 |
| SL | 1253 | -0.33 | -0.11 | -0.32 | -0.30 | [-0.39, -0.27] | 0.17 | -0.07 | 0.02 | -1.13 | 2.85 |
| ST | 742 | -0.41 | -0.17 | -0.40 | -0.41 | [-0.47, -0.34] | 0.29 | -0.05 | -0 | -1.53 | 2.47 |
