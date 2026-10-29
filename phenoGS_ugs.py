import numpy as np

def expm_multiply(L, X, phi, coeff, K): 
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


def UGS(
    L,
    m_0,
    m_1,
    phi,
    coeff,
    reg_m,
    t=50, 
    k=10,
    err=1e-32,
    stopThr=1e-4,
    max_iter=1e3, 
    verbose=False,
    P=None,
    eps = 1e-8,
    **kwargs,
):

    N = L.shape[0]
    v = np.ones(N)
    w = np.ones(N)
    a = np.ones(N) / N
    # reg_m: scalar or length-1/2 sequence -> unbalanced (KL marginal penalty);
    #        None -> balanced (exponents fi_1 = fi_2 = 1).
    if reg_m is None:
        fi_1 = fi_2 = 1.0
    else:
        reg_m1, reg_m2 = get_parameter_pair(reg_m)
        fi_1 = reg_m1 / (reg_m1 + t)
        fi_2 = reg_m2 / (reg_m2 + t)

    for i in range(1, int(max_iter) + 1):
        v_prev = v
        w_prev = w
        v = (m_0 / (expm_multiply(L, a * w, phi, coeff, k) + eps)) #** fi_1
        w = (m_1 / (expm_multiply(L, a * v, phi, coeff, k) + eps)) #** fi_2
        v = np.clip(v, eps, 1e32)
        w = np.clip(w, eps, 1e32)
        v = v ** fi_1
        w = w ** fi_2


        if (
            np.any(np.isnan(v))
            or np.any(np.isnan(w))
            or np.any(np.isinf(v))
            or np.any(np.isinf(w))
         
        ):
            v = v_prev
            w = w_prev

            print(
                f"Warning: numerical errors at iteration {i} cheb with t {t} and k {k}"
            )
            break
        if i % 10 == 0:
            if verbose: 
                print(i, np.sum(np.abs(v - v_prev)))
            if np.sum(np.abs(v - v_prev)) < stopThr: 
                if verbose:
                    print("converged at iteration %d" % i)
                break
    if P is None:
        return np.sum(4 * a * t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps)))
    else:
        dist = np.sum(4 * a * t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps)))
        I = np.eye(L.shape[0])
        K_heat = expm_multiply(L, I, phi, coeff, k)

        Plan = np.diag(v) @ K_heat @ np.diag(w)
        return dist, Plan
                
   
def get_parameter_pair(parameter):

    if isinstance(parameter, float) or isinstance(parameter, int):
        param_1, param_2 = parameter, parameter
    elif len(parameter) == 1:
        param_1, param_2 = parameter[0], parameter[0]
    else:
        if len(parameter) > 2:
            raise ValueError(
                "Parameter must be either a scalar, \
                             or an indexable object of length 1 or 2."
            )
        else:
            param_1, param_2 = parameter[0], parameter[1]

    return param_1, param_2