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


We build a graph between two Gaussian distributions and compute the distance between two signals on that graph.
```python
import numpy as np

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
