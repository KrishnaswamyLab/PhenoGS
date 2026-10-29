import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.special import ive
import scipy.sparse as sps
from sklearn.neighbors import kneighbors_graph
from joblib import Parallel, delayed
import graphtools
import matplotlib.pyplot as plt


def expm_multiply(L, X, phi, coeff, K):
    """Chebyshev-polynomial approximation of exp(-t*L) @ X."""
    inv_phi = 1.0 / phi
    two_inv_phi = 2.0 / phi

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = inv_phi * (L @ X) - T0
    Y = Y + coeff[1] * T1

    for j in range(2, K + 1):
        T2 = two_inv_phi * (L @ T1) - 2 * T1 - T0
        Y = Y + coeff[j] * T2
        T0 = T1
        T1 = T2

    return Y


def uniform_scale_factor(coeff, K):
    c = coeff[:K + 1]
    s = 0.5 * c[0] - c[1::2].sum() + c[2::2].sum()
    return s


def compute_chebychev_coeff_all(phi, tau, K):
    return 2 * ive(np.arange(0, K + 1), -tau * phi)


def _build_M_v_col(j, rows_per_cond, data_per_cond, X, a0, n_rows):
    v = np.zeros(n_rows, dtype=np.float64)
    rows_j = rows_per_cond[j]
    if rows_j.size:
        v[rows_j] = (data_per_cond[j] / X[rows_j]) * a0
    return v


def _process_column(j, L, M_v_col, phi, coeff, k_cheb, rows_per_cond,
                     data_per_cond, pair_idx, n_cols, eps):
    X_2_col = expm_multiply(L, M_v_col, phi, coeff, k_cheb)
    results = []
    for i in range(j + 1, n_cols):
        rows_i = rows_per_cond[i]
        data_i = data_per_cond[i]
        if rows_i.size == 0:
            continue
        ratio = data_i / (X_2_col[rows_i] + eps)
        ratio_clipped = np.clip(ratio, eps, 1e32)
        contribution = (np.log(ratio_clipped + eps) * data_i).sum()
        results.append((pair_idx[i, j], contribution))
    return results



def load_and_filter_data(csv_path, markers, source_col='file'):
    """
    Load the CSV, restrict to the marker columns, and compute the
    grouping information keyed by `source_col`.

    Note: `markers` must include `source_col` (e.g. 'file'), since the
    grouping column is needed after the DataFrame is restricted to `markers`.
    This helper is a convenience for loading data from disk; it is NOT called
    by `run_pipeline`, which expects an already-loaded DataFrame.
    """
    df = pd.read_csv(csv_path)
    df = df[markers]
    df = df.reset_index(drop=True)

    print(f"Final marker set: {len([m for m in markers if m != source_col])} features")
    print("\nTotal CPUs available:", os.cpu_count())

    grouped_indices = df.groupby(source_col).indices
    condition_keys = sorted(grouped_indices.keys())
    print(f'N: {len(condition_keys)}')

    group_indices = {key: grouped_indices[key] for key in condition_keys}
    return df, condition_keys, group_indices



def build_graph_laplacian(df, source_col='file', knn=10):
    """Build the kNN graph (excluding the source column) and its normalized Laplacian."""
    data = df.drop(columns=[source_col])
    graph = graphtools.Graph(data, use_pygsp=True, knn=knn)
    graph.compute_laplacian("combinatorial")
    L = graph.L
    return L, data


def compute_spectral_params(L, k, t):
    """Compute the top eigenvalue-derived phi and Chebyshev coefficients."""
    phi = eigsh(L, k=k, return_eigenvectors=False)[0] / 2
    coeff = compute_chebychev_coeff_all(phi, t, k)
    return phi, coeff



def build_group_index_arrays(condition_keys, group_indices):
    """
    Build per-condition row index arrays, per-cell scale values, and
    per-column normalizing scale factors.

    Each condition is represented as a uniform distribution (mass 1/n_c on
    each of its n_c cells). The row indices come directly from the groupby,
    so the rows of a condition do NOT need to be contiguous in the DataFrame.
    """
    condition_ncells = {key: len(idx) for key, idx in group_indices.items()}

    print('Building group index/scale arrays ...')

    col_scales = np.array(
        [1.0 / condition_ncells[key] if condition_ncells[key] > 0 else 0.0
         for key in condition_keys],
        dtype=np.float64,
    )

    rows_per_cond = []
    data_per_cond = []
    for j, key in enumerate(condition_keys):
        rows_j = np.asarray(group_indices[key], dtype=np.int64)
        if rows_j.size == 0:
            rows_per_cond.append(np.array([], dtype=np.int64))
            data_per_cond.append(np.array([], dtype=np.float64))
            continue
        rows_per_cond.append(rows_j)
        data_per_cond.append(np.full(rows_j.size, col_scales[j], dtype=np.float64))

    return condition_ncells, col_scales, rows_per_cond, data_per_cond



def compute_X(coeff, k, n_rows, eps):
    """Compute X = exp(-tL) @ a using the uniform-vector shortcut."""
    a = np.ones(n_rows, dtype=np.float64) / n_rows
    print('Computing X = exp(-tL) @ a ...')
    s = uniform_scale_factor(coeff, k)
    X = s * a + eps
    return a, X


def compute_sum_M_v(n_cols, rows_per_cond, data_per_cond, X, eps):
    print('Computing sum_M_v ...')
    sum_M_v = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        rows_j = rows_per_cond[j]
        data_j = data_per_cond[j]
        if rows_j.size == 0:
            continue
        ratio = data_j / X[rows_j]
        ratio_clipped = np.clip(ratio, eps, 1e32)
        sum_M_v[j] = (np.log(ratio_clipped + eps) * data_j).sum()
    return sum_M_v


def compute_pair_indices(n_cols):
    n_pairs = n_cols * (n_cols - 1) // 2
    pair_idx = np.full((n_cols, n_cols), -1, dtype=np.int64)
    k_pair = 0
    for j in range(n_cols - 1):
        for i in range(j + 1, n_cols):
            pair_idx[i, j] = k_pair
            k_pair += 1
    return n_pairs, pair_idx


def compute_sum_M_w_batched(L, phi, coeff, k, n_rows, n_cols, rows_per_cond,
                             data_per_cond, X, a0, pair_idx, n_pairs, eps):
    """Dense/batched computation of X_2 and sum_M_w (COLUMN_BY_COLUMN_X2 = False path)."""
    print('Computing X_2 (batched, dense) ...')
    M_v_dense = np.zeros((n_rows, n_cols), dtype=np.float64)
    for j in range(n_cols):
        rows_j = rows_per_cond[j]
        if rows_j.size == 0:
            continue
        M_v_dense[rows_j, j] = (data_per_cond[j] / X[rows_j]) * a0

    X_2 = expm_multiply(L, M_v_dense, phi, coeff, k)
    del M_v_dense

    print('Computing sum_M_w ...')
    sum_M_w = np.zeros(n_pairs, dtype=np.float64)
    for i in range(1, n_cols):
        rows_i = rows_per_cond[i]
        data_i = data_per_cond[i]
        if rows_i.size == 0:
            continue
        ratio = data_i[:, None] / (X_2[rows_i, :i] + eps)
        ratio_clipped = np.clip(ratio, eps, 1e32)
        contributions = np.log(ratio_clipped + eps) * data_i[:, None]
        sum_M_w[pair_idx[i, :i]] = contributions.sum(axis=0)

    del X_2
    return sum_M_w


def compute_sum_M_w_parallel(L, phi, coeff, k, n_rows, n_cols, rows_per_cond,
                              data_per_cond, X, a0, pair_idx, n_pairs, eps):
    """Column-by-column parallel computation of X_2 and sum_M_w (COLUMN_BY_COLUMN_X2 = True path)."""
    print('Computing X_2 + sum_M_w column-by-column (parallel) ...')
    sum_M_w = np.zeros(n_pairs, dtype=np.float64)

    results_per_col = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_process_column)(
            j, L, _build_M_v_col(j, rows_per_cond, data_per_cond, X, a0, n_rows),
            phi, coeff, k, rows_per_cond, data_per_cond, pair_idx, n_cols, eps
        )
        for j in range(n_cols)
    )
    for col_results in results_per_col:
        for pair_index, contribution in col_results:
            sum_M_w[pair_index] = contribution

    return sum_M_w


def assemble_D_matrix(sum_M_v, sum_M_w, n_cols, a0, t, condition_keys):
    print('Assembling D_matrix ...')
    const = 4 * a0 * t
    sum_M_v = sum_M_v * const
    sum_M_w = sum_M_w * const

    sum_M_v_ext = np.repeat(sum_M_v, np.arange(n_cols - 1, -1, -1))
    D_vec = sum_M_w + sum_M_v_ext

    D_matrix = np.zeros((n_cols, n_cols))
    upper_idx = np.triu_indices(n_cols, k=1)
    D_matrix[upper_idx] = D_vec
    D_matrix = D_matrix + D_matrix.T

    D_df = pd.DataFrame(D_matrix, index=condition_keys, columns=condition_keys)
    return D_matrix, D_df



def save_matrix(D_df, output_path):
    D_df.to_csv(output_path)
    print("\n── Exporting condition matrix ────────────────────────")
    print(f"Done. Written to {output_path}")


def plot_matrix(D_matrix, title='a'):
    plt.figure()
    plt.imshow(D_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()



def run_pipeline(df, source_col='file', t=10.0, k=5, knn=10, eps=1e-8,
                  column_by_column_x2=True, output_path=None):
    n_rows = len(df)

    grouped_indices = df.groupby(source_col).indices
    condition_keys = sorted(grouped_indices.keys())
    group_indices = {key: grouped_indices[key] for key in condition_keys}
    n_cols = len(condition_keys)

    L, _data = build_graph_laplacian(df, source_col=source_col, knn=knn)
    phi, coeff = compute_spectral_params(L, k, t)

    (condition_ncells, col_scales,
     rows_per_cond, data_per_cond) = build_group_index_arrays(condition_keys, group_indices)

    a, X = compute_X(coeff, k, n_rows, eps)
    sum_M_v = compute_sum_M_v(n_cols, rows_per_cond, data_per_cond, X, eps)

    n_pairs, pair_idx = compute_pair_indices(n_cols)

    if not column_by_column_x2:
        sum_M_w = compute_sum_M_w_batched(
            L, phi, coeff, k, n_rows, n_cols, rows_per_cond, data_per_cond,
            X, a[0], pair_idx, n_pairs, eps
        )
    else:
        sum_M_w = compute_sum_M_w_parallel(
            L, phi, coeff, k, n_rows, n_cols, rows_per_cond, data_per_cond,
            X, a[0], pair_idx, n_pairs, eps
        )

    D_matrix, D_df = assemble_D_matrix(sum_M_v, sum_M_w, n_cols, a[0], t, condition_keys)

    if output_path is not None:
        save_matrix(D_df, output_path)

    return D_df
