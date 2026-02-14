/**
 * @file dlagts.c
 * @brief DLAGTS solves (T - lambda*I)*x = y or (T - lambda*I)^T*x = y.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLAGTS may be used to solve one of the systems of equations
 *
 *    (T - lambda*I)*x = y   or   (T - lambda*I)**T*x = y,
 *
 * where T is an n by n tridiagonal matrix, for x, following the
 * factorization of (T - lambda*I) as
 *
 *    (T - lambda*I) = P*L*U,
 *
 * by routine DLAGTF. The choice of equation to be solved is
 * controlled by the argument job, and in each case there is an option
 * to perturb zero or very small diagonal elements of U, this option
 * being intended for use in applications such as inverse iteration.
 *
 * @param[in]     job   Specifies the job to be performed:
 *                      =  1: (T - lambda*I)x = y, no perturbation of U.
 *                      = -1: (T - lambda*I)x = y, with perturbation of U.
 *                      =  2: (T - lambda*I)^T x = y, no perturbation of U.
 *                      = -2: (T - lambda*I)^T x = y, with perturbation of U.
 * @param[in]     n     The order of the matrix T. n >= 0.
 * @param[in]     A     Double precision array, dimension (n).
 *                      The diagonal elements of U as returned from DLAGTF.
 * @param[in]     B     Double precision array, dimension (n-1).
 *                      The first super-diagonal elements of U as returned
 *                      from DLAGTF.
 * @param[in]     C     Double precision array, dimension (n-1).
 *                      The sub-diagonal elements of L as returned from DLAGTF.
 * @param[in]     D     Double precision array, dimension (n-2).
 *                      The second super-diagonal elements of U as returned
 *                      from DLAGTF.
 * @param[in]     in    Integer array, dimension (n).
 *                      Details of the permutation matrix P as returned from
 *                      DLAGTF.
 * @param[in,out] Y     Double precision array, dimension (n).
 *                      On entry, the right hand side vector y.
 *                      On exit, overwritten by the solution vector x.
 * @param[in,out] tol   On entry, with job < 0, tol should be the minimum
 *                      perturbation to be made to very small diagonal elements
 *                      of U. If tol is supplied as non-positive, then it is
 *                      reset to eps*max(|u(i,j)|).
 *                      If job > 0 then tol is not referenced.
 *                      On exit, tol is changed as described above, only if
 *                      tol is non-positive on entry.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: overflow would occur when computing the info-th
 *                           element of the solution vector x (job > 0 only).
 */
void dlagts(
    const int job,
    const int n,
    const f64* restrict A,
    const f64* restrict B,
    const f64* restrict C,
    const f64* restrict D,
    const int* restrict in,
    f64* restrict Y,
    f64* tol,
    int* info)
{
    int k;
    int absjob;
    f64 absak, ak, bignum, eps, pert, sfmin, temp;

    *info = 0;
    absjob = job < 0 ? -job : job;
    if ((absjob > 2) || (job == 0)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("DLAGTS", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    eps = DBL_EPSILON;
    sfmin = DBL_MIN;
    bignum = 1.0 / sfmin;

    if (job < 0) {
        if (*tol <= 0.0) {
            *tol = fabs(A[0]);
            if (n > 1) {
                *tol = fmax(*tol, fmax(fabs(A[1]), fabs(B[0])));
            }
            for (k = 2; k < n; k++) {
                *tol = fmax(*tol, fmax(fabs(A[k]), fmax(fabs(B[k - 1]), fabs(D[k - 2]))));
            }
            *tol = (*tol) * eps;
            if (*tol == 0.0) {
                *tol = eps;
            }
        }
    }

    if (absjob == 1) {
        /* Forward elimination (apply P*L to y) */
        for (k = 1; k < n; k++) {
            if (in[k - 1] == 0) {
                Y[k] = Y[k] - C[k - 1] * Y[k - 1];
            } else {
                temp = Y[k - 1];
                Y[k - 1] = Y[k];
                Y[k] = temp - C[k - 1] * Y[k];
            }
        }

        if (job == 1) {
            /* Back substitution without perturbation */
            for (k = n - 1; k >= 0; k--) {
                if (k <= n - 3) {
                    temp = Y[k] - B[k] * Y[k + 1] - D[k] * Y[k + 2];
                } else if (k == n - 2) {
                    temp = Y[k] - B[k] * Y[k + 1];
                } else {
                    temp = Y[k];
                }
                ak = A[k];
                absak = fabs(ak);
                if (absak < 1.0) {
                    if (absak < sfmin) {
                        if (absak == 0.0 || fabs(temp) * sfmin > absak) {
                            *info = k + 1;  /* 1-based for Fortran compatibility */
                            return;
                        } else {
                            temp = temp * bignum;
                            ak = ak * bignum;
                        }
                    } else if (fabs(temp) > absak * bignum) {
                        *info = k + 1;  /* 1-based */
                        return;
                    }
                }
                Y[k] = temp / ak;
            }
        } else {
            /* Back substitution with perturbation (job == -1) */
            for (k = n - 1; k >= 0; k--) {
                if (k <= n - 3) {
                    temp = Y[k] - B[k] * Y[k + 1] - D[k] * Y[k + 2];
                } else if (k == n - 2) {
                    temp = Y[k] - B[k] * Y[k + 1];
                } else {
                    temp = Y[k];
                }
                ak = A[k];
                pert = copysign(*tol, ak);
                for (;;) {
                    absak = fabs(ak);
                    if (absak < 1.0) {
                        if (absak < sfmin) {
                            if (absak == 0.0 || fabs(temp) * sfmin > absak) {
                                ak = ak + pert;
                                pert = 2.0 * pert;
                                continue;
                            } else {
                                temp = temp * bignum;
                                ak = ak * bignum;
                            }
                        } else if (fabs(temp) > absak * bignum) {
                            ak = ak + pert;
                            pert = 2.0 * pert;
                            continue;
                        }
                    }
                    break;
                }
                Y[k] = temp / ak;
            }
        }
    } else {
        /*
         * job == 2 or job == -2
         * Forward substitution (solve U^T * x = y)
         */
        if (job == 2) {
            /* Without perturbation */
            for (k = 0; k < n; k++) {
                if (k >= 2) {
                    temp = Y[k] - B[k - 1] * Y[k - 1] - D[k - 2] * Y[k - 2];
                } else if (k == 1) {
                    temp = Y[k] - B[k - 1] * Y[k - 1];
                } else {
                    temp = Y[k];
                }
                ak = A[k];
                absak = fabs(ak);
                if (absak < 1.0) {
                    if (absak < sfmin) {
                        if (absak == 0.0 || fabs(temp) * sfmin > absak) {
                            *info = k + 1;  /* 1-based */
                            return;
                        } else {
                            temp = temp * bignum;
                            ak = ak * bignum;
                        }
                    } else if (fabs(temp) > absak * bignum) {
                        *info = k + 1;  /* 1-based */
                        return;
                    }
                }
                Y[k] = temp / ak;
            }
        } else {
            /* With perturbation (job == -2) */
            for (k = 0; k < n; k++) {
                if (k >= 2) {
                    temp = Y[k] - B[k - 1] * Y[k - 1] - D[k - 2] * Y[k - 2];
                } else if (k == 1) {
                    temp = Y[k] - B[k - 1] * Y[k - 1];
                } else {
                    temp = Y[k];
                }
                ak = A[k];
                pert = copysign(*tol, ak);
                for (;;) {
                    absak = fabs(ak);
                    if (absak < 1.0) {
                        if (absak < sfmin) {
                            if (absak == 0.0 || fabs(temp) * sfmin > absak) {
                                ak = ak + pert;
                                pert = 2.0 * pert;
                                continue;
                            } else {
                                temp = temp * bignum;
                                ak = ak * bignum;
                            }
                        } else if (fabs(temp) > absak * bignum) {
                            ak = ak + pert;
                            pert = 2.0 * pert;
                            continue;
                        }
                    }
                    break;
                }
                Y[k] = temp / ak;
            }
        }

        /* Back permutation (apply L^T * P^T) */
        for (k = n - 1; k >= 1; k--) {
            if (in[k - 1] == 0) {
                Y[k - 1] = Y[k - 1] - C[k - 1] * Y[k];
            } else {
                temp = Y[k - 1];
                Y[k - 1] = Y[k];
                Y[k] = temp - C[k - 1] * Y[k];
            }
        }
    }
}
