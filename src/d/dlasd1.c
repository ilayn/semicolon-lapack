/**
 * @file dlasd1.c
 * @brief DLASD1 computes the SVD of an upper bidiagonal matrix B of the
 *        specified size. Used by sbdsdc.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLASD1 computes the SVD of an upper bidiagonal N-by-M matrix B,
 * where N = NL + NR + 1 and M = N + SQRE. DLASD1 is called from DLASD0.
 *
 * The algorithm consists of three stages:
 *   1. Deflation (DLASD2)
 *   2. Secular equation solving (DLASD3)
 *   3. Singular vector update
 *
 * @param[in]     nl      Row dimension of upper block. nl >= 1.
 * @param[in]     nr      Row dimension of lower block. nr >= 1.
 * @param[in]     sqre    = 0: lower block is nr-by-nr square.
 *                         = 1: lower block is nr-by-(nr+1) rectangular.
 * @param[in,out] D       Array of dimension n. Singular values on entry/exit.
 * @param[in,out] alpha   Diagonal element of added row.
 * @param[in,out] beta    Off-diagonal element of added row.
 * @param[in,out] U       Array (ldu, n). Left singular vectors.
 * @param[in]     ldu     Leading dimension of U. ldu >= n.
 * @param[in,out] VT      Array (ldvt, m). Right singular vectors transposed.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= m.
 * @param[in,out] IDXQ    Integer array of dimension n. Sorting permutation.
 * @param[out]    IWORK   Integer array of dimension 4*n.
 * @param[out]    work    Double array of dimension 3*m^2 + 2*m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dlasd1(const INT nl, const INT nr, const INT sqre,
            f64* restrict D, f64* alpha, f64* beta,
            f64* restrict U, const INT ldu,
            f64* restrict VT, const INT ldvt,
            INT* restrict IDXQ, INT* restrict IWORK,
            f64* restrict work, INT* info)
{
    INT coltyp, i, idx, idxc, idxp, iq, isigma, iu2, ivt2, iz;
    INT k, ldq, ldu2, ldvt2, m, n, n1, n2;
    f64 orgnrm;

    *info = 0;

    if (nl < 1) {
        *info = -1;
    } else if (nr < 1) {
        *info = -2;
    } else if (sqre < 0 || sqre > 1) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("DLASD1", -(*info));
        return;
    }

    n = nl + nr + 1;
    m = n + sqre;

    ldu2 = n;
    ldvt2 = m;

    iz = 0;
    isigma = iz + m;
    iu2 = isigma + n;
    ivt2 = iu2 + ldu2 * n;
    iq = ivt2 + ldvt2 * m;

    idx = 0;
    idxc = idx + n;
    coltyp = idxc + n;
    idxp = coltyp + n;

    orgnrm = fabs(*alpha) > fabs(*beta) ? fabs(*alpha) : fabs(*beta);
    D[nl] = 0.0;
    for (i = 0; i < n; i++) {
        if (fabs(D[i]) > orgnrm) {
            orgnrm = fabs(D[i]);
        }
    }
    dlascl("G", 0, 0, orgnrm, 1.0, n, 1, D, n, info);
    *alpha = *alpha / orgnrm;
    *beta = *beta / orgnrm;

    dlasd2(nl, nr, sqre, &k, D, &work[iz], *alpha, *beta, U, ldu,
           VT, ldvt, &work[isigma], &work[iu2], ldu2,
           &work[ivt2], ldvt2, &IWORK[idxp], &IWORK[idx],
           &IWORK[idxc], IDXQ, &IWORK[coltyp], info);

    ldq = k;
    dlasd3(nl, nr, sqre, k, D, &work[iq], ldq, &work[isigma],
           U, ldu, &work[iu2], ldu2, VT, ldvt, &work[ivt2],
           ldvt2, &IWORK[idxc], &IWORK[coltyp], &work[iz], info);

    if (*info != 0) {
        return;
    }

    dlascl("G", 0, 0, 1.0, orgnrm, n, 1, D, n, info);

    n1 = k;
    n2 = n - k;
    dlamrg(n1, n2, D, 1, -1, IDXQ);
}
