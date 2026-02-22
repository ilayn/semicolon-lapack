/*
 * probe_blas_int.c — Configure-time detection of BLAS integer width.
 *
 * Technique adapted from libblastrampoline (Julia):
 * https://github.com/JuliaLinearAlgebra/libblastrampoline/blob/main/src/autodetection.c
 *
 * libblastrampoline uses Fortran isamax (pass-by-pointer, endian-dependent).
 * We use CBLAS cblas_idamax (pass-by-value, register-based, endian-independent).
 *
 * Return codes:
 *   0 — LP64  (32-bit BLAS integers)
 *   1 — ILP64 (64-bit BLAS integers)
 *   2 — broken BLAS or unknown
 *
 * ABI note: this file declares cblas_idamax with int64_t parameters regardless
 * of the actual library's integer width. When linked against an LP64 library,
 * the callee reads only the lower 32 bits of each register (guaranteed by
 * System V AMD64 ABI, AArch64 AAPCS, RISC-V LP64). This is technically
 * undefined behavior per the C standard but well-defined by all 64-bit
 * hardware ABIs in use today.
 */

#include <stdint.h>

extern int64_t cblas_idamax(int64_t n, const double* x, int64_t incx);

int main(void) {
    double x[] = {1.0, 3.0, 2.0};

    /* Sanity: does this BLAS work at all? */
    int64_t sanity = cblas_idamax(3, x, 1);

    /* Detection: lower 32 bits = 3, full 64 bits = negative */
    int64_t detect = cblas_idamax((int64_t)0xFFFFFFFF00000003, x, 1);

    if (sanity != 1) return 2;   /* broken BLAS */
    if (detect == 1) return 0;   /* LP64: read lower 32 bits, saw n=3 */
    if (detect == 0) return 1;   /* ILP64: read full 64 bits, saw n<0 */
    return 2;                    /* unknown */
}
