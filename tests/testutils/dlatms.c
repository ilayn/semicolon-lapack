/**
 * @file dlatms.c
 * @brief Matrix generator for testing LAPACK routines.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlatms.f
 * Uses xoshiro256+ PRNG instead of LAPACK's archaic 48-bit LCG.
 *
 * Supports:
 * - PACK='N': No packing (full matrix storage)
 * - PACK='U': Zero out all subdiagonal entries (symmetric only)
 * - PACK='L': Zero out all superdiagonal entries (symmetric only)
 * - PACK='B': Store lower triangle in band storage (symmetric/lower triangular)
 * - PACK='Q': Store upper triangle in band storage (symmetric/upper triangular)
 * - PACK='Z': Band storage format for general band matrices (GB)
 *
 * Generation methods:
 * - Uses orthogonal transformations via DLAGGE for nonsymmetric matrices
 * - Uses orthogonal transformations via DLAGSY for symmetric matrices
 */

#include <math.h>
#include <stdlib.h>
#include "verify.h"
#include <string.h>
#include <cblas.h>
#include "test_rng.h"

/* Forward declarations for routines not in verify.h */
extern void xerbla(const char* srname, const int info);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);

/**
 * DLATMS generates random matrices with specified singular values
 * (or symmetric/hermitian with specified eigenvalues) for testing
 * LAPACK programs.
 *
 * @param[in]     m       Number of rows.
 * @param[in]     n       Number of columns.
 * @param[in]     dist    Distribution for random numbers: 'U'=uniform(0,1),
 *                        'S'=symmetric(-1,1), 'N'=normal(0,1).
 * @param[in]     seed    Random number seed.
 * @param[in]     sym     Matrix symmetry: 'N'=nonsymmetric, 'S'=symmetric,
 *                        'H'=hermitian (same as S for real), 'P'=positive definite.
 * @param[out]    d       Array of singular values/eigenvalues (dimension min(m,n)).
 * @param[in]     mode    How to compute d:
 *                        0: Use d as input
 *                        1: d[0]=1, d[1:n-1]=1/cond
 *                        2: d[0:n-2]=1, d[n-1]=1/cond
 *                        3: d[i]=cond^(-(i)/(n-1)) (geometric)
 *                        4: d[i]=1-(i)/(n-1)*(1-1/cond) (arithmetic)
 *                        5: Random in [1/cond, 1] with log uniform distribution
 *                        6: Random from same distribution as matrix
 *                        MODE < 0 has same meaning as ABS(MODE), order reversed.
 * @param[in]     cond    Condition number (>= 1).
 * @param[in]     dmax    Maximum singular value (d is scaled so max|d[i]|=|dmax|).
 * @param[in]     kl      Lower bandwidth for band storage.
 * @param[in]     ku      Upper bandwidth for band storage.
 * @param[in]     pack    Packing: 'N'=no packing, 'Z'=band storage.
 * @param[out]    A       Output matrix.
 *                        If pack='N': dimension (lda, n), lda >= m.
 *                        If pack='Z': dimension (lda, n), lda >= kl+ku+1.
 * @param[in]     lda     Leading dimension of A.
 * @param[out]    work    Workspace, dimension (3*max(m,n) + m*n) for pack='Z',
 *                        dimension (m+n) for pack='N'.
 * @param[out]    info    0=success, <0=argument error, >0=other error.
 */
void dlatms(
    const int m,
    const int n,
    const char* dist,
    uint64_t seed,
    const char* sym,
    double* d,
    const int mode,
    const double cond,
    const double dmax,
    const int kl,
    const int ku,
    const char* pack,
    double* A,
    const int lda,
    double* work,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j;
    int mnmin;
    double temp, alpha;
    int isym;    /* 1 = nonsymmetric, 2 = symmetric */
    int irsign = 0;  /* 1 = random signs on diagonal */
    int idist;   /* 1 = U(0,1), 2 = U(-1,1), 3 = N(0,1) */
    int ipack;   /* 0=N, 1=U, 2=L, 5=B, 6=Q, 7=Z */
    int iinfo;

    *info = 0;

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Decode DIST */
    if (dist[0] == 'U' || dist[0] == 'u') {
        idist = 1;
    } else if (dist[0] == 'S' || dist[0] == 's') {
        idist = 2;
    } else if (dist[0] == 'N' || dist[0] == 'n') {
        idist = 3;
    } else {
        idist = -1;
    }

    /* Decode SYM */
    if (sym[0] == 'N' || sym[0] == 'n') {
        isym = 1;
        irsign = 0;
    } else if (sym[0] == 'P' || sym[0] == 'p') {
        isym = 2;
        irsign = 0;
    } else if (sym[0] == 'S' || sym[0] == 's' ||
               sym[0] == 'H' || sym[0] == 'h') {
        isym = 2;
        irsign = 1;
    } else {
        isym = -1;
    }

    /* Decode PACK */
    int isympk = 0;  /* Symmetric packing indicator */
    if (pack[0] == 'N' || pack[0] == 'n') {
        ipack = 0;
    } else if (pack[0] == 'U' || pack[0] == 'u') {
        ipack = 1;
        isympk = 1;
    } else if (pack[0] == 'L' || pack[0] == 'l') {
        ipack = 2;
        isympk = 1;
    } else if (pack[0] == 'C' || pack[0] == 'c') {
        ipack = 3;
        isympk = 2;
    } else if (pack[0] == 'R' || pack[0] == 'r') {
        ipack = 4;
        isympk = 3;
    } else if (pack[0] == 'B' || pack[0] == 'b') {
        ipack = 5;
        isympk = 3;
    } else if (pack[0] == 'Q' || pack[0] == 'q') {
        ipack = 6;
        isympk = 2;
    } else if (pack[0] == 'Z' || pack[0] == 'z') {
        ipack = 7;
    } else {
        ipack = -1;
    }

    /* Compute limits on bandwidth */
    int llb = (kl < m - 1) ? kl : m - 1;
    int uub = (ku < n - 1) ? ku : n - 1;
    mnmin = (m < n) ? m : n;

    /* Set INFO if an error */
    if (m < 0) {
        *info = -1;
    } else if (m != n && isym != 1) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (idist == -1) {
        *info = -3;
    } else if (isym == -1) {
        *info = -5;
    } else if (mode < -6 || mode > 6) {
        *info = -7;
    } else if ((mode != 0 && mode != 6 && mode != -6) && cond < ONE) {
        *info = -8;
    } else if (kl < 0) {
        *info = -10;
    } else if (ku < 0 || (isym != 1 && kl != ku)) {
        *info = -11;
    } else if (ipack == -1 || (isympk == 1 && isym == 1) ||
               (isympk == 2 && isym == 1 && kl > 0) ||
               (isympk == 3 && isym == 1 && ku > 0) ||
               (isympk != 0 && m != n)) {
        /* PACK='U' or 'L' requires symmetric matrix (SYM != 'N')
         * PACK='C' or 'Q' requires SYM='S'/'H'/'P' and KL=0
         * PACK='R' or 'B' requires SYM='S'/'H'/'P' and KU=0
         * Symmetric packing requires square matrix */
        *info = -12;
    } else if ((ipack == 0 || ipack == 1 || ipack == 2) && lda < m) {
        *info = -14;
    } else if ((ipack == 5 || ipack == 6) && lda < uub + 1) {
        *info = -14;
    } else if (ipack == 7 && lda < llb + uub + 1) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("DLATMS", -(*info));
        return;
    }

    /* Initialize RNG with seed */
    rng_seed(seed);

    /* Generate diagonal (singular values or eigenvalues) via dlatm1 */
    dlatm1(mode, cond, irsign, idist, d, mnmin, &iinfo);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    /* Scale by dmax */
    if (mode != 0 && mode != 6 && mode != -6) {
        temp = fabs(d[0]);
        for (i = 1; i < mnmin; i++) {
            if (fabs(d[i]) > temp) temp = fabs(d[i]);
        }
        if (temp > ZERO) {
            alpha = dmax / temp;
            for (i = 0; i < mnmin; i++) {
                d[i] *= alpha;
            }
        } else {
            *info = 2;  /* Cannot scale to dmax (max. sing. value is 0) */
            return;
        }
    }

    /* Generate the matrix */
    if (ipack == 0 || ipack == 1 || ipack == 2 || ipack == 3 || ipack == 4) {
        /* No packing, triangular packing, or packed triangular - generate directly into A first */

        /* Special case: diagonal matrix (llb==0 && uub==0) */
        if (llb == 0 && uub == 0) {
            /* Just copy D to the diagonal, no orthogonal transformations needed */
            dlaset("F", m, n, ZERO, ZERO, A, lda);
            for (i = 0; i < mnmin; i++) {
                A[i + i * lda] = d[i];
            }
        } else if (isym == 1) {
            /* Nonsymmetric: A = U * D * V' */
            dlagge(m, n, llb, uub, d, A, lda, seed, work, &iinfo);
            if (iinfo != 0) {
                *info = 3;
                return;
            }
        } else {
            /* Symmetric: A = U * D * U' */
            dlagsy(n, llb, d, A, lda, work, &iinfo);
            if (iinfo != 0) {
                *info = 3;
                return;
            }
        }

        /* Apply packing if needed */
        if (ipack == 1) {
            /* PACK='U': Zero out all subdiagonal entries */
            for (j = 0; j < m; j++) {
                for (i = j + 1; i < m; i++) {
                    A[i + j * lda] = ZERO;
                }
            }
        } else if (ipack == 2) {
            /* PACK='L': Zero out all superdiagonal entries */
            for (j = 1; j < m; j++) {
                for (i = 0; i < j; i++) {
                    A[i + j * lda] = ZERO;
                }
            }
        } else if (ipack == 3) {
            /* PACK='C': Upper triangle packed columnwise.
             * Pack A[i,j] for i <= j into linear array. */
            int jc = 0;
            for (j = 0; j < m; j++) {
                for (i = 0; i <= j; i++) {
                    A[jc + i] = A[i + j * lda];
                }
                jc += j + 1;
            }
        } else if (ipack == 4) {
            /* PACK='R': Lower triangle packed columnwise.
             * Pack A[i,j] for i >= j into linear array. */
            int jc = 0;
            for (j = 0; j < m; j++) {
                for (i = j; i < m; i++) {
                    A[jc + i - j] = A[i + j * lda];
                }
                jc += m - j;
            }
        }
    } else if (ipack == 7) {
        /* Pack='Z': Band storage */

        /* Special case: diagonal matrix (llb==0 && uub==0) */
        if (llb == 0 && uub == 0) {
            /* Just copy D to the diagonal in band storage */
            dlaset("F", 1, n, ZERO, ZERO, A, lda);
            for (i = 0; i < mnmin; i++) {
                A[ku + i * lda] = d[i];  /* ku=0 for diagonal, so A[0 + i*lda] */
            }
        } else {
            /* Generate full matrix first, then pack */
            int ldtmp = (m > 1) ? m : 1;
            double* Afull = work;              /* Temporary full matrix */
            double* work2 = work + ldtmp * n;  /* Remaining workspace */

            if (isym == 1) {
                /* Nonsymmetric: generate full matrix first */
                dlagge(m, n, llb, uub, d, Afull, ldtmp, seed, work2, &iinfo);
            } else {
                /* Symmetric: generate full matrix first */
                dlagsy(n, llb, d, Afull, ldtmp, work2, &iinfo);
            }
            if (iinfo != 0) {
                *info = 3;
                return;
            }

            /* Pack into band storage format 'Z'
             * Band storage: A[ku+i-j + j*lda] = Afull[i,j]
             * for max(0,j-ku) <= i <= min(m-1,j+kl) */
            dlaset("F", kl + ku + 1, n, ZERO, ZERO, A, lda);
            for (j = 0; j < n; j++) {
                int i_start = (j - ku > 0) ? j - ku : 0;
                int i_end = (j + kl < m - 1) ? j + kl : m - 1;
                for (i = i_start; i <= i_end; i++) {
                    A[ku + i - j + j * lda] = Afull[i + j * ldtmp];
                }
            }
        }
    } else if (ipack == 5 || ipack == 6) {
        /* Pack='B' or 'Q': Band storage for symmetric matrices
         * 'B' stores lower triangle in band format (uub=0)
         * 'Q' stores upper triangle in band format (llb=0) */

        int llb_pack = (ipack == 6) ? 0 : llb;
        int uub_pack = (ipack == 5) ? 0 : uub;

        /* Special case: diagonal matrix */
        if (llb == 0 && uub == 0) {
            dlaset("F", 1, n, ZERO, ZERO, A, lda);
            for (i = 0; i < mnmin; i++) {
                A[i * lda] = d[i];
            }
        } else {
            /* Generate full matrix first, then pack */
            int ldtmp = (m > 1) ? m : 1;
            double* Afull = work;
            double* work2 = work + ldtmp * n;

            /* Symmetric: A = U * D * U' */
            dlagsy(n, llb, d, Afull, ldtmp, work2, &iinfo);
            if (iinfo != 0) {
                *info = 3;
                return;
            }

            /* Pack into band storage format
             * For 'B': A[i-j + j*lda] = Afull[i,j] for j <= i <= min(n-1,j+kl)
             * For 'Q': A[kl+i-j + j*lda] = Afull[i,j] for max(0,j-kl) <= i <= j */
            int lda_band = uub_pack + llb_pack + 1;
            dlaset("F", lda_band, n, ZERO, ZERO, A, lda);

            if (ipack == 5) {
                /* 'B': lower triangle in band storage */
                for (j = 0; j < n; j++) {
                    int i_end = (j + llb < n - 1) ? j + llb : n - 1;
                    for (i = j; i <= i_end; i++) {
                        A[i - j + j * lda] = Afull[i + j * ldtmp];
                    }
                }
            } else {
                /* 'Q': upper triangle in band storage */
                for (j = 0; j < n; j++) {
                    int i_start = (j - uub > 0) ? j - uub : 0;
                    for (i = i_start; i <= j; i++) {
                        A[uub + i - j + j * lda] = Afull[i + j * ldtmp];
                    }
                }
            }
        }
    }
}
