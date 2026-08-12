# PhenoGS
Code for the paper Pheno-GS: Phenoscape-scale Geodesic Sinkhorn.

> [!NOTE]  
>This repository is still in development.

### Installation
using Git, by first cloning the repository and running:
```bash
pip install -e .
```

### Minimal Example
You can reproduce this example in the following notebook


We build a graph between 5 Gaussian distributions and compute the pairwise distance between signals on that graph.
```python
import numpy as np
import pandas as pd
import phenoGS_bms as bms

SOURCE = 'file'
t = 10.0
k = 5
knn = 10
eps = 1e-8

rng = np.random.default_rng(42)

marker_cols = [f'marker_{i}' for i in range(10)]
frames = []

for c in range(5):
    mean = rng.normal(loc=0.0, scale=2.0, size=10)
    X = rng.normal(loc=mean, scale=1.0, size=(100, 10))
    X = X - X.min() + 0.01
    df_c = pd.DataFrame(X, columns=marker_cols)
    df_c[SOURCE] = f'condition_{c}'
    frames.append(df_c)

df = pd.concat(frames, ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

D_df = bms.run_pipeline(
    df=df,
    source_col=SOURCE,
    t=t, k=k, knn=knn, eps=eps,
    column_by_column_x2=True,
    output_path=None,
)

print(D_df)

bms.plot_matrix(D_df.values, title='Synthetic data test')
```
Note that it is also possible to provide a graph instance directly to the `GeoSinkhorn` class with `GeoSinkhorn(tau=1.0, order=10, method="cheb", graph=graph)`. The `graph` must have a Laplacian attribute `graph.L`. We suggest using a sparse Laplacian (e.g. in COO format) for better performance.

### How to Cite

If you find this code useful in your research, please cite the following paper (expand for BibTeX):
<details>
<summary>
Wilkinson, A., Tape, C. J., & Krishnaswamy, S. (2026). Pheno-GS: Phenoscape-scale Geodesic Sinkhorn. In 2026 IEEE International Workshop on Machine Learning for Signal Processing (MLSP).
</summary>

```bibtex
@inproceedings{wilkinson2026phenoGS,
  title={Pheno-GS: Phenoscape-scale Geodesic Sinkhorn},
  author={Wilkinson, Alistair and Tape, Christopher J and Krishnaswamy, Smita},
  booktitle={2026 IEEE International Workshop on Machine Learning for Signal Processing (MLSP},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
```
