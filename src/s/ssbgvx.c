/**
 * @file ssbgvx.c
 * @brief SSBGVX computes selected eigenvalues of a generalized symmetric-definite banded eigenproblem.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSBGVX computes selected eigenvalues, and optionally, eigenvectors
 * of a real generalized symmetric-definite banded eigenproblem, of
 * the form A*x = lambda*B*x. Here A and B are assumed to be symmetric
 * and banded, and B is also positive definite. Eigenvalues and
 * eigenvectors can be selected by specifying either all eigenvalues,
 * a range of values or a range of indices for the desired eigenvalues.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     range  = 'A': all eigenvalues will be found
 *                        = 'V': all eigenvalues in (vl,vu] will be found
 *                        = 'I': the il-th through iu-th eigenvalues will be found
 * @param[in]     uplo   = 'U': Upper triangles of A and B are stored
 *                        = 'L': Lower triangles of A and B are stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in]     ka     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of A. ka >= 0.
 * @param[in]     kb     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of B. kb >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, contents are destroyed.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in,out] BB     The banded matrix B. Array of dimension (ldbb, n).
 *                       On exit, the split Cholesky factor S from spbstf.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    Q      If jobz='V', the n-by-n transformation matrix.
 *                       Array of dimension (ldq, n).
 * @param[in]     ldq    The leading dimension of Q.
 *                       ldq >= 1, and if jobz='V', ldq >= max(1,n).
 * @param[in]     vl     If range='V', the lower bound of the interval.
 * @param[in]     vu     If range='V', the upper bound of the interval. vl < vu.
 * @param[in]     il     If range='I', the index of the smallest eigenvalue.
 * @param[in]     iu     If range='I', the index of the largest eigenvalue.
 * @param[in]     abstol The absolute error tolerance for eigenvalues.
 * @param[out]    m      The total number of eigenvalues found.
 * @param[out]    W      The eigenvalues in ascending order. Array of dimension (n).
 * @param[out]    Z      If jobz='V', the eigenvectors. Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of Z.
 *                       ldz >= 1, and if jobz='V', ldz >= max(1,n).
 * @param[out]    work   Workspace array of dimension (7*n).
 * @param[out]    iwork  Integer workspace array of dimension (5*n).
 * @param[out]    ifail  If jobz='V', indices of eigenvectors that failed to converge.
 *                       Array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - <= n: i eigenvectors failed to converge
 *                         - > n: spbstf returned info = i (B not positive definite)
 */
void ssbgvx(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    const int ka,
    const int kb,
    float* const restrict AB,
    const int ldab,
    float* const restrict BB,
    const int ldbb,
    float* const restrict Q,
    const int ldq,
    const float vl,
    const float vu,
    const int il,
    const int iu,
    const float abstol,
    int* m,
    float* const restrict W,
    float* const restrict Z,
    const int ldz,
    float* const restrict work,
    int* const restrict iwork,
    int* const restrict ifail,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int alleig, indeig, test, upper, valeig, wantz;
    int i, iinfo, indd, inde, indee, indisp, indiwo, indwrk, itmp1, j, jj, nsplit;
    float tmp1;
    char order, vect;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ka < 0) {
        *info = -5;
    } else if (kb < 0 || kb > ka) {
        *info = -6;
    } else if (ldab < ka + 1) {
        *info = -8;
    } else if (ldbb < kb + 1) {
        *info = -10;
    } else if (ldq < 1 || (wantz && ldq < n)) {
        *info = -12;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -14;
        } else if (indeig) {
            if (il < 1 || il > ((1 > n) ? 1 : n)) {
                *info = -15;
            } else if (iu < ((n < il) ? n : il) || iu > n) {
                *info = -16;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -21;
        }
    }

    if (*info != 0) {
        xerbla("SSBGVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    // Form a split Cholesky factorization of B
    spbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    // Transform problem to standard eigenvalue problem
    ssbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Q, ldq, work, &iinfo);

    // Reduce symmetric band matrix to tridiagonal form
    indd = 0;
    inde = indd + n;
    indwrk = inde + n;
    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    ssbtrd(&vect, uplo, n, ka, AB, ldab, &work[indd], &work[inde], Q, ldq, &work[indwrk], &iinfo);

    // If all eigenvalues are desired and abstol <= 0, then call ssterf or ssteqr.
    // If this fails for some eigenvalue, then try sstebz.
    test = 0;
    if (indeig) {
        if (il == 1 && iu == n) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_scopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
        if (!wantz) {
            ssterf(n, W, &work[indee], info);
        } else {
            slacpy("A", n, n, Q, ldq, Z, ldz);
            ssteqr(jobz, n, W, &work[indee], Z, ldz, &work[indwrk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    // Otherwise, call sstebz and, if eigenvectors are desired, call sstein.
    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indisp = n;
    indiwo = indisp + n;
    sstebz(range, &order, n, vl, vu, il, iu, abstol,
           &work[indd], &work[inde], m, &nsplit, W,
           &iwork[0], &iwork[indisp], &work[indwrk], &iwork[indiwo], info);

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W,
               &iwork[0], &iwork[indisp], Z, ldz,
               &work[indwrk], &iwork[indiwo], ifail, info);

        // Apply transformation matrix used in reduction to tridiagonal form
        for (j = 0; j < *m; j++) {
            cblas_scopy(n, &Z[j * ldz], 1, work, 1);
            cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, ONE, Q, ldq,
                        work, 1, ZERO, &Z[j * ldz], 1);
        }
    }

L30:
    // If eigenvalues are not in order, then sort them, along with eigenvectors.
    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = -1;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i >= 0) {
                itmp1 = iwork[i];
                W[i] = W[j];
                iwork[i] = iwork[j];
                W[j] = tmp1;
                iwork[j] = itmp1;
                cblas_sswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
