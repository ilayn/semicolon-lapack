/*
 * probe_gemmtr.c — Configure-time check that cblas_?gemmtr computes GEMMTR.
 *
 * OpenBLAS up to 0.3.34, when built with CMake, compiles cblas_?gemmtr from
 * gemm.c instead of gemmt.c (fixed by OpenBLAS PR 6118), so the symbol exists
 * but is GEMM under another name and computes nothing useful for a GEMMTR
 * call. With -DPROBE_GEMMT the probe checks the older name cblas_?gemmt.
 *
 * C := C - A*B**T on the upper and on the lower triangle, column major,
 * N = 3, K = 2, in all four precisions, on integer data so the result is
 * exact. The other triangle must be left unchanged.
 *
 * Return codes:
 *   0 — correct
 *   1 — wrong result
 */

#include <complex.h>
#include <stdint.h>
#include <string.h>

#if PROBE_ILP64
typedef int64_t blas_int;
#else
typedef int32_t blas_int;
#endif

enum { COLMAJOR = 102, UPPER = 121, LOWER = 122, NOTRANS = 111, TRANS = 112 };

#ifdef PROBE_GEMMT
#define SGEMMTR cblas_sgemmt
#define DGEMMTR cblas_dgemmt
#define CGEMMTR cblas_cgemmt
#define ZGEMMTR cblas_zgemmt
#else
#define SGEMMTR cblas_sgemmtr
#define DGEMMTR cblas_dgemmtr
#define CGEMMTR cblas_cgemmtr
#define ZGEMMTR cblas_zgemmtr
#endif

void SGEMMTR(int, int, int, int, blas_int, blas_int, float, const float*, blas_int,
             const float*, blas_int, float, float*, blas_int);
void DGEMMTR(int, int, int, int, blas_int, blas_int, double, const double*, blas_int,
             const double*, blas_int, double, double*, blas_int);
void CGEMMTR(int, int, int, int, blas_int, blas_int, const void*, const void*, blas_int,
             const void*, blas_int, const void*, void*, blas_int);
void ZGEMMTR(int, int, int, int, blas_int, blas_int, const void*, const void*, blas_int,
             const void*, blas_int, const void*, void*, blas_int);

#define N 3
#define K 2

/* A is N x K, B is N x K, C is N x N, all column major. */
static const double a[N * K] = {1, 2, 3, 4, 5, 6};
static const double b[N * K] = {2, 1, 3, 1, 2, 2};
static const double c[N * N] = {10, 20, 30, 40, 50, 60, 70, 80, 90};

static int expect(const double* got, int uplo)
{
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double want = c[i + j * N];
            if ((uplo == UPPER) ? (i <= j) : (i >= j)) {
                for (int l = 0; l < K; l++) {
                    want -= a[i + l * N] * b[j + l * N];
                }
            }
            if (got[i + j * N] != want) {
                return 1;
            }
        }
    }
    return 0;
}

int main(void)
{
    int bad = 0;
    for (int u = 0; u < 2; u++) {
        int uplo = u ? LOWER : UPPER;
        double got[N * N];

        float sa[N * K], sb[N * K], sc[N * N];
        for (int i = 0; i < N * K; i++) { sa[i] = (float)a[i]; sb[i] = (float)b[i]; }
        for (int i = 0; i < N * N; i++) { sc[i] = (float)c[i]; }
        SGEMMTR(COLMAJOR, uplo, NOTRANS, TRANS, N, K, -1.0f, sa, N, sb, N, 1.0f, sc, N);
        for (int i = 0; i < N * N; i++) { got[i] = sc[i]; }
        bad |= expect(got, uplo);

        double dc[N * N];
        memcpy(dc, c, sizeof dc);
        DGEMMTR(COLMAJOR, uplo, NOTRANS, TRANS, N, K, -1.0, a, N, b, N, 1.0, dc, N);
        bad |= expect(dc, uplo);

        float complex ca[N * K], cb[N * K], cc[N * N];
        const float complex cm = -1.0f, co = 1.0f;
        for (int i = 0; i < N * K; i++) { ca[i] = (float)a[i]; cb[i] = (float)b[i]; }
        for (int i = 0; i < N * N; i++) { cc[i] = (float)c[i]; }
        CGEMMTR(COLMAJOR, uplo, NOTRANS, TRANS, N, K, &cm, ca, N, cb, N, &co, cc, N);
        for (int i = 0; i < N * N; i++) {
            got[i] = crealf(cc[i]);
            bad |= cimagf(cc[i]) != 0.0f;
        }
        bad |= expect(got, uplo);

        double complex za[N * K], zb[N * K], zc[N * N];
        const double complex zm = -1.0, zo = 1.0;
        for (int i = 0; i < N * K; i++) { za[i] = a[i]; zb[i] = b[i]; }
        for (int i = 0; i < N * N; i++) { zc[i] = c[i]; }
        ZGEMMTR(COLMAJOR, uplo, NOTRANS, TRANS, N, K, &zm, za, N, zb, N, &zo, zc, N);
        for (int i = 0; i < N * N; i++) {
            got[i] = creal(zc[i]);
            bad |= cimag(zc[i]) != 0.0;
        }
        bad |= expect(got, uplo);
    }
    return bad;
}
