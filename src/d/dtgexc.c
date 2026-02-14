/**
 * @file dtgexc.c
 * @brief DTGEXC reorders the generalized real Schur decomposition.
 */

#include "semicolon_lapack_double.h"

/**
 * DTGEXC reorders the generalized real Schur decomposition of a real
 * matrix pair (A,B) using an orthogonal equivalence transformation
 *
 *                (A, B) = Q * (A, B) * Z**T,
 *
 * so that the diagonal block of (A, B) with row index IFST is moved
 * to row ILST.
 *
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Array of dimension (lda, n). Generalized real Schur form.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Array of dimension (ldb, n). Upper triangular part of Schur form.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q       Array of dimension (ldq, n). The orthogonal matrix Q.
 *                        Not referenced if wantq = 0.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Array of dimension (ldz, n). The orthogonal matrix Z.
 *                        Not referenced if wantz = 0.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[in,out] ifst    On entry, index of block to move (0-based). On exit, adjusted
 *                        to point to first row if it pointed to second row of 2x2 block.
 * @param[in,out] ilst    On entry, target index (0-based). On exit, actual position
 *                        of the block.
 * @param[out]    work    Array of dimension (lwork).
 * @param[in]     lwork   The dimension of work. lwork >= 4*n+16 for n > 1, else >= 1.
 *                        If lwork = -1, workspace query.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: swap failed, matrices partially reordered
 */
void dtgexc(
    const int wantq,
    const int wantz,
    const int n,
    f64* restrict A,
    const int lda,
    f64* restrict B,
    const int ldb,
    f64* restrict Q,
    const int ldq,
    f64* restrict Z,
    const int ldz,
    int* ifst,
    int* ilst,
    f64* restrict work,
    const int lwork,
    int* info)
{
    const f64 ZERO = 0.0;

    int lquery;
    int here, lwmin, nbf, nbl, nbnext;

    *info = 0;
    lquery = (lwork == -1);

    if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldq < 1 || (wantq && ldq < (1 > n ? 1 : n))) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < (1 > n ? 1 : n))) {
        *info = -11;
    } else if (*ifst < 0 || *ifst >= n) {
        *info = -12;
    } else if (*ilst < 0 || *ilst >= n) {
        *info = -13;
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
        } else {
            lwmin = 4 * n + 16;
        }
        work[0] = (f64)lwmin;

        if (lwork < lwmin && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("DTGEXC", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n <= 1) {
        return;
    }

    /* Determine the first row of the specified block and find out
       if it is 1-by-1 or 2-by-2. */
    if (*ifst > 0) {
        if (A[*ifst + (*ifst - 1) * lda] != ZERO) {
            *ifst = *ifst - 1;
        }
    }
    nbf = 1;
    if (*ifst < n - 1) {
        if (A[(*ifst + 1) + *ifst * lda] != ZERO) {
            nbf = 2;
        }
    }

    /* Determine the first row of the final block
       and find out if it is 1-by-1 or 2-by-2. */
    if (*ilst > 0) {
        if (A[*ilst + (*ilst - 1) * lda] != ZERO) {
            *ilst = *ilst - 1;
        }
    }
    nbl = 1;
    if (*ilst < n - 1) {
        if (A[(*ilst + 1) + *ilst * lda] != ZERO) {
            nbl = 2;
        }
    }
    if (*ifst == *ilst) {
        return;
    }

    if (*ifst < *ilst) {

        /* Update ILST. */
        if (nbf == 2 && nbl == 1) {
            *ilst = *ilst - 1;
        }
        if (nbf == 1 && nbl == 2) {
            *ilst = *ilst + 1;
        }

        here = *ifst;

        while (1) {

            /* Swap with next one below. */
            if (nbf == 1 || nbf == 2) {

                /* Current block either 1-by-1 or 2-by-2. */
                nbnext = 1;
                if (here + nbf + 1 <= n - 1) {
                    if (A[(here + nbf + 1) + (here + nbf) * lda] != ZERO) {
                        nbnext = 2;
                    }
                }
                dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                       here, nbf, nbnext, work, lwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                here = here + nbnext;

                /* Test if 2-by-2 block breaks into two 1-by-1 blocks. */
                if (nbf == 2) {
                    if (A[(here + 1) + here * lda] == ZERO) {
                        nbf = 3;
                    }
                }

            } else {

                /* Current block consists of two 1-by-1 blocks, each of which
                   must be swapped individually. */
                nbnext = 1;
                if (here + 3 <= n - 1) {
                    if (A[(here + 3) + (here + 2) * lda] != ZERO) {
                        nbnext = 2;
                    }
                }
                dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                       here + 1, 1, nbnext, work, lwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                if (nbnext == 1) {

                    /* Swap two 1-by-1 blocks. */
                    dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                           here, 1, 1, work, lwork, info);
                    if (*info != 0) {
                        *ilst = here;
                        return;
                    }
                    here = here + 1;

                } else {

                    /* Recompute NBNEXT in case of 2-by-2 split. */
                    if (A[(here + 2) + (here + 1) * lda] == ZERO) {
                        nbnext = 1;
                    }
                    if (nbnext == 2) {

                        /* 2-by-2 block did not split. */
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here, 1, nbnext, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here + 2;
                    } else {

                        /* 2-by-2 block did split. */
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here, 1, 1, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here + 1;
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here, 1, 1, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here + 1;
                    }

                }
            }
            if (here >= *ilst) {
                break;
            }
        }
    } else {
        here = *ifst;

        while (1) {

            /* Swap with next one above. */
            if (nbf == 1 || nbf == 2) {

                /* Current block either 1-by-1 or 2-by-2. */
                nbnext = 1;
                if (here >= 2) {
                    if (A[(here - 1) + (here - 2) * lda] != ZERO) {
                        nbnext = 2;
                    }
                }
                dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                       here - nbnext, nbnext, nbf, work, lwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                here = here - nbnext;

                /* Test if 2-by-2 block breaks into two 1-by-1 blocks. */
                if (nbf == 2) {
                    if (A[(here + 1) + here * lda] == ZERO) {
                        nbf = 3;
                    }
                }

            } else {

                /* Current block consists of two 1-by-1 blocks, each of which
                   must be swapped individually. */
                nbnext = 1;
                if (here >= 2) {
                    if (A[(here - 1) + (here - 2) * lda] != ZERO) {
                        nbnext = 2;
                    }
                }
                dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                       here - nbnext, nbnext, 1, work, lwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                if (nbnext == 1) {

                    /* Swap two 1-by-1 blocks. */
                    dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                           here, nbnext, 1, work, lwork, info);
                    if (*info != 0) {
                        *ilst = here;
                        return;
                    }
                    here = here - 1;
                } else {

                    /* Recompute NBNEXT in case of 2-by-2 split. */
                    if (A[here + (here - 1) * lda] == ZERO) {
                        nbnext = 1;
                    }
                    if (nbnext == 2) {

                        /* 2-by-2 block did not split. */
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here - 1, 2, 1, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here - 2;
                    } else {

                        /* 2-by-2 block did split. */
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here, 1, 1, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here - 1;
                        dtgex2(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz,
                               here, 1, 1, work, lwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here = here - 1;
                    }
                }
            }
            if (here <= *ilst) {
                break;
            }
        }
    }

    *ilst = here;
    work[0] = (f64)lwmin;
}
