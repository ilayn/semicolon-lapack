/**
 * @file strexc.c
 * @brief STREXC reorders the real Schur factorization of a real matrix.
 */

#include "semicolon_lapack_single.h"

/**
 * STREXC reorders the real Schur factorization of a real matrix
 * A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
 * moved to row ILST.
 *
 * The real Schur form T is reordered by an orthogonal similarity
 * transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors
 * is updated by postmultiplying it with Z.
 *
 * T must be in Schur canonical form (as returned by SHSEQR), that is,
 * block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 * 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.
 *
 * @param[in]     compq  'V': update the matrix Q of Schur vectors.
 *                       'N': do not update Q.
 * @param[in]     n      The order of the matrix T. n >= 0.
 *                       If n == 0, arguments ilst and ifst may be any value.
 * @param[in,out] T      On entry, the upper quasi-triangular matrix T, in
 *                       Schur canonical form. On exit, the reordered matrix.
 *                       Dimension (ldt, n).
 * @param[in]     ldt    The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] Q      On entry, if compq = 'V', the matrix Q of Schur vectors.
 *                       On exit, if compq = 'V', Q has been postmultiplied by
 *                       the orthogonal transformation matrix Z.
 *                       If compq = 'N', Q is not referenced. Dimension (ldq, n).
 * @param[in]     ldq    The leading dimension of Q. ldq >= 1, and if
 *                       compq = 'V', ldq >= max(1, n).
 * @param[in,out] ifst   On entry, the row index of the block to move (1-based).
 *                       On exit, if ifst pointed to the second row of a 2-by-2
 *                       block, it is changed to point to the first row.
 * @param[in,out] ilst   On entry, the target row index for the block (1-based).
 *                       On exit, points to the first row of the block in its
 *                       final position.
 * @param[out]    work   Workspace array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - = 1: two adjacent blocks were too close to swap; T may
 *                           have been partially reordered, and ilst points to
 *                           the first row of the current position of the block
 *                           being moved.
 */
void strexc(const char* compq, const int n, f32* T, const int ldt,
            f32* Q, const int ldq, int* ifst, int* ilst,
            f32* work, int* info)
{
    const f32 ZERO = 0.0f;

    int wantq;
    int here, nbf, nbl, nbnext;

    /* Decode and test the input arguments. */
    *info = 0;
    wantq = (compq[0] == 'V' || compq[0] == 'v');
    if (!wantq && compq[0] != 'N' && compq[0] != 'n') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldq < 1 || (wantq && ldq < (n > 1 ? n : 1))) {
        *info = -6;
    } else if (((*ifst < 1 || *ifst > n)) && (n > 0)) {
        *info = -7;
    } else if (((*ilst < 1 || *ilst > n)) && (n > 0)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("STREXC", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 1)
        return;

    /* Convert to 0-based indexing for internal use */
    int ifst0 = *ifst - 1;
    int ilst0 = *ilst - 1;

    /* Determine the first row of specified block
       and find out if it is 1 by 1 or 2 by 2. */
    if (ifst0 > 0) {
        if (T[ifst0 + (ifst0 - 1) * ldt] != ZERO)
            ifst0 = ifst0 - 1;
    }
    nbf = 1;
    if (ifst0 < n - 1) {
        if (T[ifst0 + 1 + ifst0 * ldt] != ZERO)
            nbf = 2;
    }

    /* Determine the first row of the final block
       and find out if it is 1 by 1 or 2 by 2. */
    if (ilst0 > 0) {
        if (T[ilst0 + (ilst0 - 1) * ldt] != ZERO)
            ilst0 = ilst0 - 1;
    }
    nbl = 1;
    if (ilst0 < n - 1) {
        if (T[ilst0 + 1 + ilst0 * ldt] != ZERO)
            nbl = 2;
    }

    if (ifst0 == ilst0) {
        *ifst = ifst0 + 1;  /* Convert back to 1-based */
        *ilst = ilst0 + 1;
        return;
    }

    if (ifst0 < ilst0) {
        /* Update ILST */
        if (nbf == 2 && nbl == 1)
            ilst0 = ilst0 - 1;
        if (nbf == 1 && nbl == 2)
            ilst0 = ilst0 + 1;

        here = ifst0;

        while (here < ilst0) {
            /* Swap block with next one below */
            if (nbf == 1 || nbf == 2) {
                /* Current block either 1 by 1 or 2 by 2 */
                nbnext = 1;
                if (here + nbf + 1 <= n - 1) {
                    if (T[here + nbf + 1 + (here + nbf) * ldt] != ZERO)
                        nbnext = 2;
                }
                slaexc(wantq, n, T, ldt, Q, ldq, here, nbf, nbnext, work, info);
                if (*info != 0) {
                    *ifst = ifst0 + 1;
                    *ilst = here + 1;  /* Convert back to 1-based */
                    return;
                }
                here = here + nbnext;

                /* Test if 2 by 2 block breaks into two 1 by 1 blocks */
                if (nbf == 2) {
                    if (T[here + 1 + here * ldt] == ZERO)
                        nbf = 3;
                }

            } else {
                /* Current block consists of two 1 by 1 blocks each of which
                   must be swapped individually */
                nbnext = 1;
                if (here + 3 <= n - 1) {
                    if (T[here + 3 + (here + 2) * ldt] != ZERO)
                        nbnext = 2;
                }
                slaexc(wantq, n, T, ldt, Q, ldq, here + 1, 1, nbnext, work, info);
                if (*info != 0) {
                    *ifst = ifst0 + 1;
                    *ilst = here + 1;
                    return;
                }
                if (nbnext == 1) {
                    /* Swap two 1 by 1 blocks, no problems possible */
                    slaexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work, info);
                    here = here + 1;
                } else {
                    /* Recompute NBNEXT in case 2 by 2 split */
                    if (T[here + 2 + (here + 1) * ldt] == ZERO)
                        nbnext = 1;
                    if (nbnext == 2) {
                        /* 2 by 2 Block did not split */
                        slaexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work, info);
                        if (*info != 0) {
                            *ifst = ifst0 + 1;
                            *ilst = here + 1;
                            return;
                        }
                        here = here + 2;
                    } else {
                        /* 2 by 2 Block did split */
                        slaexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work, info);
                        slaexc(wantq, n, T, ldt, Q, ldq, here + 1, 1, 1, work, info);
                        here = here + 2;
                    }
                }
            }
        }

    } else {
        here = ifst0;

        while (here > ilst0) {
            /* Swap block with next one above */
            if (nbf == 1 || nbf == 2) {
                /* Current block either 1 by 1 or 2 by 2 */
                nbnext = 1;
                if (here >= 2) {
                    if (T[here - 1 + (here - 2) * ldt] != ZERO)
                        nbnext = 2;
                }
                slaexc(wantq, n, T, ldt, Q, ldq, here - nbnext, nbnext, nbf, work, info);
                if (*info != 0) {
                    *ifst = ifst0 + 1;
                    *ilst = here + 1;
                    return;
                }
                here = here - nbnext;

                /* Test if 2 by 2 block breaks into two 1 by 1 blocks */
                if (nbf == 2) {
                    if (T[here + 1 + here * ldt] == ZERO)
                        nbf = 3;
                }

            } else {
                /* Current block consists of two 1 by 1 blocks each of which
                   must be swapped individually */
                nbnext = 1;
                if (here >= 2) {
                    if (T[here - 1 + (here - 2) * ldt] != ZERO)
                        nbnext = 2;
                }
                slaexc(wantq, n, T, ldt, Q, ldq, here - nbnext, nbnext, 1, work, info);
                if (*info != 0) {
                    *ifst = ifst0 + 1;
                    *ilst = here + 1;
                    return;
                }
                if (nbnext == 1) {
                    /* Swap two 1 by 1 blocks, no problems possible */
                    slaexc(wantq, n, T, ldt, Q, ldq, here, nbnext, 1, work, info);
                    here = here - 1;
                } else {
                    /* Recompute NBNEXT in case 2 by 2 split */
                    if (T[here + (here - 1) * ldt] == ZERO)
                        nbnext = 1;
                    if (nbnext == 2) {
                        /* 2 by 2 Block did not split */
                        slaexc(wantq, n, T, ldt, Q, ldq, here - 1, 2, 1, work, info);
                        if (*info != 0) {
                            *ifst = ifst0 + 1;
                            *ilst = here + 1;
                            return;
                        }
                        here = here - 2;
                    } else {
                        /* 2 by 2 Block did split */
                        slaexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work, info);
                        slaexc(wantq, n, T, ldt, Q, ldq, here - 1, 1, 1, work, info);
                        here = here - 2;
                    }
                }
            }
        }
    }

    *ifst = ifst0 + 1;  /* Convert back to 1-based */
    *ilst = here + 1;
}
