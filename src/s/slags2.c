/**
 * @file slags2.c
 * @brief SLAGS2 computes 2-by-2 orthogonal matrices U, V, and Q.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAGS2 computes 2-by-2 orthogonal matrices U, V and Q, such
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
void slags2(const int upper,
            const f32 a1, const f32 a2, const f32 a3,
            const f32 b1, const f32 b2, const f32 b3,
            f32* csu, f32* snu, f32* csv, f32* snv,
            f32* csq, f32* snq)
{
    f32 a, b, c, d, r, s1, s2;
    f32 csl, csr, snl, snr;
    f32 ua11, ua11r, ua12, ua21, ua22, ua22r;
    f32 vb11, vb11r, vb12, vb21, vb22, vb22r;
    f32 aua11, aua12, aua21, aua22;
    f32 avb11, avb12, avb21, avb22;

    if (upper) {
        a = a1 * b3;
        d = a3 * b1;
        b = a2 * b1 - a1 * b2;

        slasv2(a, b, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabsf(csl) >= fabsf(snl) || fabsf(csr) >= fabsf(snr)) {
            ua11r = csl * a1;
            ua12 = csl * a2 + snl * a3;

            vb11r = csr * b1;
            vb12 = csr * b2 + snr * b3;

            aua12 = fabsf(csl) * fabsf(a2) + fabsf(snl) * fabsf(a3);
            avb12 = fabsf(csr) * fabsf(b2) + fabsf(snr) * fabsf(b3);

            if ((fabsf(ua11r) + fabsf(ua12)) != 0.0f) {
                if (aua12 / (fabsf(ua11r) + fabsf(ua12)) <=
                    avb12 / (fabsf(vb11r) + fabsf(vb12))) {
                    slartg(-ua11r, ua12, csq, snq, &r);
                } else {
                    slartg(-vb11r, vb12, csq, snq, &r);
                }
            } else {
                slartg(-vb11r, vb12, csq, snq, &r);
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

            aua22 = fabsf(snl) * fabsf(a2) + fabsf(csl) * fabsf(a3);
            avb22 = fabsf(snr) * fabsf(b2) + fabsf(csr) * fabsf(b3);

            if ((fabsf(ua21) + fabsf(ua22)) != 0.0f) {
                if (aua22 / (fabsf(ua21) + fabsf(ua22)) <=
                    avb22 / (fabsf(vb21) + fabsf(vb22))) {
                    slartg(-ua21, ua22, csq, snq, &r);
                } else {
                    slartg(-vb21, vb22, csq, snq, &r);
                }
            } else {
                slartg(-vb21, vb22, csq, snq, &r);
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

        slasv2(a, c, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabsf(csr) >= fabsf(snr) || fabsf(csl) >= fabsf(snl)) {
            ua21 = -snr * a1 + csr * a2;
            ua22r = csr * a3;

            vb21 = -snl * b1 + csl * b2;
            vb22r = csl * b3;

            aua21 = fabsf(snr) * fabsf(a1) + fabsf(csr) * fabsf(a2);
            avb21 = fabsf(snl) * fabsf(b1) + fabsf(csl) * fabsf(b2);

            if ((fabsf(ua21) + fabsf(ua22r)) != 0.0f) {
                if (aua21 / (fabsf(ua21) + fabsf(ua22r)) <=
                    avb21 / (fabsf(vb21) + fabsf(vb22r))) {
                    slartg(ua22r, ua21, csq, snq, &r);
                } else {
                    slartg(vb22r, vb21, csq, snq, &r);
                }
            } else {
                slartg(vb22r, vb21, csq, snq, &r);
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

            aua11 = fabsf(csr) * fabsf(a1) + fabsf(snr) * fabsf(a2);
            avb11 = fabsf(csl) * fabsf(b1) + fabsf(snl) * fabsf(b2);

            if ((fabsf(ua11) + fabsf(ua12)) != 0.0f) {
                if (aua11 / (fabsf(ua11) + fabsf(ua12)) <=
                    avb11 / (fabsf(vb11) + fabsf(vb12))) {
                    slartg(ua12, ua11, csq, snq, &r);
                } else {
                    slartg(vb12, vb11, csq, snq, &r);
                }
            } else {
                slartg(vb12, vb11, csq, snq, &r);
            }

            *csu = snr;
            *snu = csr;
            *csv = snl;
            *snv = csl;
        }
    }
}
