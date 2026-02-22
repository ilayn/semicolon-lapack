/**
 * @file dlags2.c
 * @brief DLAGS2 computes 2-by-2 orthogonal matrices U, V, and Q.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAGS2 computes 2-by-2 orthogonal matrices U, V and Q, such
 * that if ( UPPER ) then
 *
 *           U**T *A*Q = U**T *( A1 A2 )*Q = ( x  0  )
 *                             ( 0  A3 )     ( x  x  )
 * and
 *           V**T*B*Q = V**T *( B1 B2 )*Q = ( x  0  )
 *                            ( 0  B3 )     ( x  x  )
 *
 * or if ( .NOT.UPPER ) then
 *
 *           U**T *A*Q = U**T *( A1 0  )*Q = ( x  x  )
 *                             ( A2 A3 )     ( 0  x  )
 * and
 *           V**T*B*Q = V**T*( B1 0  )*Q = ( x  x  )
 *                           ( B2 B3 )     ( 0  x  )
 *
 * The rows of the transformed A and B are parallel, where
 *
 *   U = (  CSU  SNU ), V = (  CSV SNV ), Q = (  CSQ   SNQ )
 *       ( -SNU  CSU )      ( -SNV CSV )      ( -SNQ   CSQ )
 *
 * Z**T denotes the transpose of Z.
 *
 * @param[in]  upper  = nonzero: the input matrices A and B are upper triangular.
 *                    = 0: the input matrices A and B are lower triangular.
 * @param[in]  a1     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  a2     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  a3     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  b1     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[in]  b2     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[in]  b3     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[out] csu    The desired orthogonal matrix U.
 * @param[out] snu    The desired orthogonal matrix U.
 * @param[out] csv    The desired orthogonal matrix V.
 * @param[out] snv    The desired orthogonal matrix V.
 * @param[out] csq    The desired orthogonal matrix Q.
 * @param[out] snq    The desired orthogonal matrix Q.
 */
void dlags2(const INT upper,
            const f64 a1, const f64 a2, const f64 a3,
            const f64 b1, const f64 b2, const f64 b3,
            f64* csu, f64* snu, f64* csv, f64* snv,
            f64* csq, f64* snq)
{
    f64 a, b, c, d, r, s1, s2;
    f64 csl, csr, snl, snr;
    f64 ua11, ua11r, ua12, ua21, ua22, ua22r;
    f64 vb11, vb11r, vb12, vb21, vb22, vb22r;
    f64 aua11, aua12, aua21, aua22;
    f64 avb11, avb12, avb21, avb22;

    if (upper) {
        a = a1 * b3;
        d = a3 * b1;
        b = a2 * b1 - a1 * b2;

        dlasv2(a, b, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabs(csl) >= fabs(snl) || fabs(csr) >= fabs(snr)) {
            ua11r = csl * a1;
            ua12 = csl * a2 + snl * a3;

            vb11r = csr * b1;
            vb12 = csr * b2 + snr * b3;

            aua12 = fabs(csl) * fabs(a2) + fabs(snl) * fabs(a3);
            avb12 = fabs(csr) * fabs(b2) + fabs(snr) * fabs(b3);

            if ((fabs(ua11r) + fabs(ua12)) != 0.0) {
                if (aua12 / (fabs(ua11r) + fabs(ua12)) <=
                    avb12 / (fabs(vb11r) + fabs(vb12))) {
                    dlartg(-ua11r, ua12, csq, snq, &r);
                } else {
                    dlartg(-vb11r, vb12, csq, snq, &r);
                }
            } else {
                dlartg(-vb11r, vb12, csq, snq, &r);
            }

            *csu = csl;
            *snu = -snl;
            *csv = csr;
            *snv = -snr;
        } else {
            ua21 = -snl * a1;
            ua22 = -snl * a2 + csl * a3;

            vb21 = -snr * b1;
            vb22 = -snr * b2 + csr * b3;

            aua22 = fabs(snl) * fabs(a2) + fabs(csl) * fabs(a3);
            avb22 = fabs(snr) * fabs(b2) + fabs(csr) * fabs(b3);

            if ((fabs(ua21) + fabs(ua22)) != 0.0) {
                if (aua22 / (fabs(ua21) + fabs(ua22)) <=
                    avb22 / (fabs(vb21) + fabs(vb22))) {
                    dlartg(-ua21, ua22, csq, snq, &r);
                } else {
                    dlartg(-vb21, vb22, csq, snq, &r);
                }
            } else {
                dlartg(-vb21, vb22, csq, snq, &r);
            }

            *csu = snl;
            *snu = csl;
            *csv = snr;
            *snv = csr;
        }
    } else {
        a = a1 * b3;
        d = a3 * b1;
        c = a2 * b3 - a3 * b2;

        dlasv2(a, c, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabs(csr) >= fabs(snr) || fabs(csl) >= fabs(snl)) {
            ua21 = -snr * a1 + csr * a2;
            ua22r = csr * a3;

            vb21 = -snl * b1 + csl * b2;
            vb22r = csl * b3;

            aua21 = fabs(snr) * fabs(a1) + fabs(csr) * fabs(a2);
            avb21 = fabs(snl) * fabs(b1) + fabs(csl) * fabs(b2);

            if ((fabs(ua21) + fabs(ua22r)) != 0.0) {
                if (aua21 / (fabs(ua21) + fabs(ua22r)) <=
                    avb21 / (fabs(vb21) + fabs(vb22r))) {
                    dlartg(ua22r, ua21, csq, snq, &r);
                } else {
                    dlartg(vb22r, vb21, csq, snq, &r);
                }
            } else {
                dlartg(vb22r, vb21, csq, snq, &r);
            }

            *csu = csr;
            *snu = -snr;
            *csv = csl;
            *snv = -snl;
        } else {
            ua11 = csr * a1 + snr * a2;
            ua12 = snr * a3;

            vb11 = csl * b1 + snl * b2;
            vb12 = snl * b3;

            aua11 = fabs(csr) * fabs(a1) + fabs(snr) * fabs(a2);
            avb11 = fabs(csl) * fabs(b1) + fabs(snl) * fabs(b2);

            if ((fabs(ua11) + fabs(ua12)) != 0.0) {
                if (aua11 / (fabs(ua11) + fabs(ua12)) <=
                    avb11 / (fabs(vb11) + fabs(vb12))) {
                    dlartg(ua12, ua11, csq, snq, &r);
                } else {
                    dlartg(vb12, vb11, csq, snq, &r);
                }
            } else {
                dlartg(vb12, vb11, csq, snq, &r);
            }

            *csu = snr;
            *snu = csr;
            *csv = snl;
            *snv = csl;
        }
    }
}
