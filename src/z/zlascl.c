/**
 * @file zlascl.c
 * @brief ZLASCL multiplies a general rectangular matrix by a real scalar
 *        defined as cto/cfrom.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLASCL multiplies the M by N complex matrix A by the real scalar
 * CTO/CFROM.  This is done without over/underflow as long as the final
 * result CTO*A(I,J)/CFROM does not over/underflow.  TYPE specifies that
 * A may be full, upper triangular, lower triangular, upper Hessenberg,
 * or banded.
 *
 * @param[in]     type   TYPE indices the storage type of the input matrix.
 *                       = 'G':  A is a full matrix.
 *                       = 'L':  A is a lower triangular matrix.
 *                       = 'U':  A is an upper triangular matrix.
 *                       = 'H':  A is an upper Hessenberg matrix.
 *                       = 'B':  A is a symmetric band matrix with lower
 *                               bandwidth KL and upper bandwidth KU and with
 *                               only the lower half stored.
 *                       = 'Q':  A is a symmetric band matrix with lower
 *                               bandwidth KL and upper bandwidth KU and with
 *                               only the upper half stored.
 *                       = 'Z':  A is a band matrix with lower bandwidth KL
 *                               and upper bandwidth KU. See ZGBTRF for
 *                               storage details.
 * @param[in]     kl     The lower bandwidth of A.  Referenced only if
 *                       TYPE = "B", 'Q' or 'Z'.
 * @param[in]     ku     The upper bandwidth of A.  Referenced only if
 *                       TYPE = "B", 'Q' or 'Z'.
 * @param[in]     cfrom  Must be nonzero.
 * @param[in]     cto    The matrix A is multiplied by CTO/CFROM. A(I,J) is
 *                       computed without over/underflow if the final result
 *                       CTO*A(I,J)/CFROM can be represented without
 *                       over/underflow.  CFROM must be nonzero.
 * @param[in]     m      The number of rows of the matrix A.  M >= 0.
 * @param[in]     n      The number of columns of the matrix A.  N >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       The matrix to be multiplied by CTO/CFROM.  See TYPE
 *                       for the storage type.
 * @param[in]     lda    The leading dimension of the array A.
 *                       If TYPE = "G", "L", "U", "H", LDA >= max(1,M);
 *                          TYPE = "B", LDA >= KL+1;
 *                          TYPE = "Q", LDA >= KU+1;
 *                          TYPE = "Z", LDA >= 2*KL+KU+1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value.
 */
void zlascl(const char* type, const INT kl, const INT ku,
            const f64 cfrom, const f64 cto,
            const INT m, const INT n,
            c128* restrict A, const INT lda,
            INT* info)
{
    const f64 ZERO = 0.0, ONE = 1.0;
    INT i, j, itype, k1, k2, k3, k4;
    INT done;
    f64 bignum, cfrom1, cfromc, cto1, ctoc, mul, smlnum;

    /* Test the input arguments */
    *info = 0;


    if (type[0] == 'G' || type[0] == 'g') {
        itype = 0;
    } else if (type[0] == 'L' || type[0] == 'l') {
        itype = 1;
    } else if (type[0] == 'U' || type[0] == 'u') {
        itype = 2;
    } else if (type[0] == 'H' || type[0] == 'h') {
        itype = 3;
    } else if (type[0] == 'B' || type[0] == 'b') {
        itype = 4;
    } else if (type[0] == 'Q' || type[0] == 'q') {
        itype = 5;
    } else if (type[0] == 'Z' || type[0] == 'z') {
        itype = 6;
    } else {
        itype = -1;
    }

    if (itype == -1) {
        *info = -1;
    } else if (cfrom == ZERO || isnan(cfrom)) {
        *info = -4;
    } else if (isnan(cto)) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0 || (itype == 4 && n != m) ||
               (itype == 5 && n != m)) {
        *info = -7;
    } else if (itype <= 3 && lda < (1 > m ? 1 : m)) {
        *info = -9;
    } else if (itype >= 4) {
        if (kl < 0 || kl > (m - 1 > 0 ? m - 1 : 0)) {
            *info = -2;
        } else if (ku < 0 || ku > (n - 1 > 0 ? n - 1 : 0) ||
                   ((itype == 4 || itype == 5) && kl != ku)) {
            *info = -3;
        } else if ((itype == 4 && lda < kl + 1) ||
                   (itype == 5 && lda < ku + 1) ||
                   (itype == 6 && lda < 2 * kl + ku + 1)) {
            *info = -9;
        }
    }

    if (*info != 0) {
        xerbla("ZLASCL", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0) {
        return;
    }

    /* Get machine parameters */
    smlnum = DBL_MIN;
    bignum = ONE / smlnum;

    cfromc = cfrom;
    ctoc = cto;

    done = 0;
    while (!done) {
        cfrom1 = cfromc * smlnum;
        if (cfrom1 == cfromc) {
            /* CFROMC is an inf.  Multiply by a correctly signed zero for
               finite CTOC, or a NaN if CTOC is infinite. */
            mul = ctoc / cfromc;
            done = 1;
        } else {
            cto1 = ctoc / bignum;
            if (cto1 == ctoc) {
                /* CTOC is either 0 or an inf.  In both cases, CTOC itself
                   serves as the correct multiplication factor. */
                mul = ctoc;
                done = 1;
                cfromc = ONE;
            } else if (fabs(cfrom1) > fabs(ctoc) && ctoc != ZERO) {
                mul = smlnum;
                done = 0;
                cfromc = cfrom1;
            } else if (fabs(cto1) > fabs(cfromc)) {
                mul = bignum;
                done = 0;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = 1;
                if (mul == ONE) {
                    return;
                }
            }
        }

        if (itype == 0) {
            /* Full matrix */
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 1) {
            /* Lower triangular matrix */
            for (j = 0; j < n; j++) {
                for (i = j; i < m; i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 2) {
            /* Upper triangular matrix */
            for (j = 0; j < n; j++) {
                for (i = 0; i <= (j < m - 1 ? j : m - 1); i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 3) {
            /* Upper Hessenberg matrix */
            for (j = 0; j < n; j++) {
                for (i = 0; i <= (j + 1 < m - 1 ? j + 1 : m - 1); i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 4) {
            /* Lower half of a symmetric band matrix */
            k3 = kl + 1;
            k4 = n + 1;
            for (j = 0; j < n; j++) {
                /* Fortran: DO I = 1, MIN(K3, K4-J)
                   With 1-based J: MIN(K3, K4 - (j+1)) = MIN(KL+1, N+1-(j+1)) = MIN(KL+1, N-j)
                   In 0-based: loop i = 0 to MIN(KL+1, N-j) - 1 */
                INT ilim = k3 < (k4 - (j + 1)) ? k3 : (k4 - (j + 1));
                for (i = 0; i < ilim; i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 5) {
            /* Upper half of a symmetric band matrix */
            k1 = ku + 2;
            k3 = ku + 1;
            for (j = 0; j < n; j++) {
                /* Fortran: DO I = MAX(K1-J, 1), K3
                   With 1-based J: MAX(KU+2-(j+1), 1) = MAX(KU+1-j, 1)
                   In 0-based: istart = MAX(KU-j, 0), iend = KU (inclusive)
                   i.e., loop i from MAX(K1-(j+1), 1)-1 = MAX(KU+1-j, 1)-1 = MAX(KU-j, 0)
                   to K3-1 = KU */
                INT istart = (k1 - (j + 1)) > 1 ? (k1 - (j + 1) - 1) : 0;
                for (i = istart; i < k3; i++) {
                    A[i + j * lda] *= mul;
                }
            }

        } else if (itype == 6) {
            /* Band matrix */
            k1 = kl + ku + 2;
            k2 = kl + 1;
            k3 = 2 * kl + ku + 1;
            k4 = kl + ku + 1 + m;
            for (j = 0; j < n; j++) {
                /* Fortran: DO I = MAX(K1-J, K2), MIN(K3, K4-J)
                   With 1-based J: MAX(KL+KU+2-(j+1), KL+1) = MAX(KL+KU+1-j, KL+1)
                                   MIN(2*KL+KU+1, KL+KU+1+M-(j+1)) = MIN(2*KL+KU+1, KL+KU+M-j)
                   Convert to 0-based row indices: subtract 1 from both bounds */
                INT istart_f = (k1 - (j + 1)) > k2 ? (k1 - (j + 1)) : k2;
                INT iend_f = k3 < (k4 - (j + 1)) ? k3 : (k4 - (j + 1));
                for (i = istart_f - 1; i < iend_f; i++) {
                    A[i + j * lda] *= mul;
                }
            }
        }
    }
}
