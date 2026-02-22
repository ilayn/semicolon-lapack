/* semicolon_cblas.h — Project-owned CBLAS declarations using INT.
   Replaces vendor <cblas.h> so that integer parameters track INT
   (i32 for LP64, i64 for ILP64) without depending on vendor headers. */
#ifndef SEMICOLON_CBLAS_H
#define SEMICOLON_CBLAS_H

#include <stddef.h>
#include "semicolon_lapack/types.h"

#ifndef INT
    #ifdef SEMICOLON_ILP64
        #define INT i64
    #else
        #define INT i32
    #endif
#endif

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;

#define CBLAS_INDEX size_t

/* ===================================================================
   Level 1 — dot, nrm2, asum, iamax, axpy, copy, swap, scal, rot, rotm
   =================================================================== */

/* dot */
float  cblas_sdot(const INT n, const float* x, const INT incx, const float* y, const INT incy);
double cblas_ddot(const INT n, const double* x, const INT incx, const double* y, const INT incy);

/* complex dot — _sub form (CBLAS standard; return-by-value is non-standard) */
void cblas_cdotu_sub(const INT n, const void* x, const INT incx, const void* y, const INT incy, void* ret);
void cblas_cdotc_sub(const INT n, const void* x, const INT incx, const void* y, const INT incy, void* ret);
void cblas_zdotu_sub(const INT n, const void* x, const INT incx, const void* y, const INT incy, void* ret);
void cblas_zdotc_sub(const INT n, const void* x, const INT incx, const void* y, const INT incy, void* ret);

/* nrm2 */
float  cblas_snrm2 (const INT n, const float* x, const INT incx);
double cblas_dnrm2 (const INT n, const double* x, const INT incx);
float  cblas_scnrm2(const INT n, const void* x, const INT incx);
double cblas_dznrm2(const INT n, const void* x, const INT incx);

/* asum */
float  cblas_sasum (const INT n, const float* x, const INT incx);
double cblas_dasum (const INT n, const double* x, const INT incx);
float  cblas_scasum(const INT n, const void* x, const INT incx);
double cblas_dzasum(const INT n, const void* x, const INT incx);

/* iamax */
CBLAS_INDEX cblas_isamax(const INT n, const float* x, const INT incx);
CBLAS_INDEX cblas_idamax(const INT n, const double* x, const INT incx);
CBLAS_INDEX cblas_icamax(const INT n, const void* x, const INT incx);
CBLAS_INDEX cblas_izamax(const INT n, const void* x, const INT incx);

/* axpy */
void cblas_saxpy(const INT n, const float alpha, const float* x, const INT incx, float* y, const INT incy);
void cblas_daxpy(const INT n, const double alpha, const double* x, const INT incx, double* y, const INT incy);
void cblas_caxpy(const INT n, const void* alpha, const void* x, const INT incx, void* y, const INT incy);
void cblas_zaxpy(const INT n, const void* alpha, const void* x, const INT incx, void* y, const INT incy);

/* copy */
void cblas_scopy(const INT n, const float* x, const INT incx, float* y, const INT incy);
void cblas_dcopy(const INT n, const double* x, const INT incx, double* y, const INT incy);
void cblas_ccopy(const INT n, const void* x, const INT incx, void* y, const INT incy);
void cblas_zcopy(const INT n, const void* x, const INT incx, void* y, const INT incy);

/* swap */
void cblas_sswap(const INT n, float* x, const INT incx, float* y, const INT incy);
void cblas_dswap(const INT n, double* x, const INT incx, double* y, const INT incy);
void cblas_cswap(const INT n, void* x, const INT incx, void* y, const INT incy);
void cblas_zswap(const INT n, void* x, const INT incx, void* y, const INT incy);

/* scal */
void cblas_sscal(const INT n, const float alpha, float* x, const INT incx);
void cblas_dscal(const INT n, const double alpha, double* x, const INT incx);
void cblas_cscal(const INT n, const void* alpha, void* x, const INT incx);
void cblas_zscal(const INT n, const void* alpha, void* x, const INT incx);
void cblas_csscal(const INT n, const float alpha, void* x, const INT incx);
void cblas_zdscal(const INT n, const double alpha, void* x, const INT incx);

/* rot */
void cblas_srot(const INT n, float* x, const INT incx, float* y, const INT incy, const float c, const float s);
void cblas_drot(const INT n, double* x, const INT incx, double* y, const INT incy, const double c, const double s);
void cblas_csrot(const INT n, const void* x, const INT incx, void* y, const INT incy, const float c, const float s);
void cblas_zdrot(const INT n, const void* x, const INT incx, void* y, const INT incy, const double c, const double s);

/* rotg */
void cblas_srotg(float* a, float* b, float* c, float* s);
void cblas_drotg(double* a, double* b, double* c, double* s);
void cblas_crotg(void* a, void* b, float* c, void* s);
void cblas_zrotg(void* a, void* b, double* c, void* s);

/* rotm */
void cblas_srotm(const INT n, float* x, const INT incx, float* y, const INT incy, const float* p);
void cblas_drotm(const INT n, double* x, const INT incx, double* y, const INT incy, const double* p);

/* ===================================================================
   Level 2 — matrix-vector operations
   =================================================================== */

/* gemv */
void cblas_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const float alpha, const float* a, const INT lda, const float* x, const INT incx, const float beta, float* y, const INT incy);
void cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const double alpha, const double* a, const INT lda, const double* x, const INT incx, const double beta, double* y, const INT incy);
void cblas_cgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);
void cblas_zgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);

/* gbmv */
void cblas_sgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const INT kl, const INT ku, const float alpha, const float* a, const INT lda, const float* x, const INT incx, const float beta, float* y, const INT incy);
void cblas_dgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const INT kl, const INT ku, const double alpha, const double* a, const INT lda, const double* x, const INT incx, const double beta, double* y, const INT incy);
void cblas_cgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const INT kl, const INT ku, const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);
void cblas_zgbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const INT m, const INT n,
                 const INT kl, const INT ku, const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);

/* symv */
void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha, const float* a,
                 const INT lda, const float* x, const INT incx, const float beta, float* y, const INT incy);
void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha, const double* a,
                 const INT lda, const double* x, const INT incx, const double beta, double* y, const INT incy);

/* hemv */
void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha, const void* a,
                 const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);
void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha, const void* a,
                 const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);

/* sbmv */
void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const INT k, const float alpha, const float* a,
                 const INT lda, const float* x, const INT incx, const float beta, float* y, const INT incy);
void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const INT k, const double alpha, const double* a,
                 const INT lda, const double* x, const INT incx, const double beta, double* y, const INT incy);

/* hbmv */
void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const INT k,
                 const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);
void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const INT k,
                 const void* alpha, const void* a, const INT lda, const void* x, const INT incx, const void* beta, void* y, const INT incy);

/* spmv */
void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha, const float* ap,
                 const float* x, const INT incx, const float beta, float* y, const INT incy);
void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha, const double* ap,
                 const double* x, const INT incx, const double beta, double* y, const INT incy);

/* hpmv */
void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n,
                 const void* alpha, const void* ap, const void* x, const INT incx, const void* beta, void* y, const INT incy);
void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n,
                 const void* alpha, const void* ap, const void* x, const INT incx, const void* beta, void* y, const INT incy);

/* trmv */
void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const float* a, const INT lda, float* x, const INT incx);
void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const double* a, const INT lda, double* x, const INT incx);
void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* a, const INT lda, void* x, const INT incx);
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* a, const INT lda, void* x, const INT incx);

/* trsv */
void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const float* a, const INT lda, float* x, const INT incx);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const double* a, const INT lda, double* x, const INT incx);
void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* a, const INT lda, void* x, const INT incx);
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* a, const INT lda, void* x, const INT incx);

/* tbmv */
void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const float* a, const INT lda, float* x, const INT incx);
void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const double* a, const INT lda, double* x, const INT incx);
void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const void* a, const INT lda, void* x, const INT incx);
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const void* a, const INT lda, void* x, const INT incx);

/* tbsv */
void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const float* a, const INT lda, float* x, const INT incx);
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const double* a, const INT lda, double* x, const INT incx);
void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const void* a, const INT lda, void* x, const INT incx);
void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const INT k, const void* a, const INT lda, void* x, const INT incx);

/* tpmv */
void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const float* ap, float* x, const INT incx);
void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const double* ap, double* x, const INT incx);
void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* ap, void* x, const INT incx);
void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* ap, void* x, const INT incx);

/* tpsv */
void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const float* ap, float* x, const INT incx);
void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const double* ap, double* x, const INT incx);
void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* ap, void* x, const INT incx);
void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag,
                 const INT n, const void* ap, void* x, const INT incx);

/* ger */
void cblas_sger(const enum CBLAS_ORDER order, const INT m, const INT n, const float alpha,
                const float* x, const INT incx, const float* y, const INT incy, float* a, const INT lda);
void cblas_dger(const enum CBLAS_ORDER order, const INT m, const INT n, const double alpha,
                const double* x, const INT incx, const double* y, const INT incy, double* a, const INT lda);
void cblas_cgeru(const enum CBLAS_ORDER order, const INT m, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);
void cblas_cgerc(const enum CBLAS_ORDER order, const INT m, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);
void cblas_zgeru(const enum CBLAS_ORDER order, const INT m, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);
void cblas_zgerc(const enum CBLAS_ORDER order, const INT m, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);

/* syr */
void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                const float* x, const INT incx, float* a, const INT lda);
void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                const double* x, const INT incx, double* a, const INT lda);

/* her */
void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                const void* x, const INT incx, void* a, const INT lda);
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                const void* x, const INT incx, void* a, const INT lda);

/* syr2 */
void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                 const float* x, const INT incx, const float* y, const INT incy, float* a, const INT lda);
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                 const double* x, const INT incx, const double* y, const INT incy, double* a, const INT lda);

/* her2 */
void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* a, const INT lda);

/* spr */
void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                const float* x, const INT incx, float* ap);
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                const double* x, const INT incx, double* ap);

/* hpr */
void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                const void* x, const INT incx, void* a);
void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                const void* x, const INT incx, void* a);

/* spr2 */
void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const float alpha,
                 const float* x, const INT incx, const float* y, const INT incy, float* a);
void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const double alpha,
                 const double* x, const INT incx, const double* y, const INT incy, double* a);

/* hpr2 */
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* ap);
void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const INT n, const void* alpha,
                 const void* x, const INT incx, const void* y, const INT incy, void* ap);

/* ===================================================================
   Level 3 — matrix-matrix operations
   =================================================================== */

/* gemm */
void cblas_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                 const INT m, const INT n, const INT k, const float alpha, const float* a, const INT lda,
                 const float* b, const INT ldb, const float beta, float* c, const INT ldc);
void cblas_dgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                 const INT m, const INT n, const INT k, const double alpha, const double* a, const INT lda,
                 const double* b, const INT ldb, const double beta, double* c, const INT ldc);
void cblas_cgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                 const INT m, const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);
void cblas_zgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                 const INT m, const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);

/* gemmt */
void cblas_sgemmt(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                  const INT n, const INT k, const float alpha, const float* a, const INT lda,
                  const float* b, const INT ldb, const float beta, float* c, const INT ldc);
void cblas_dgemmt(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                  const INT n, const INT k, const double alpha, const double* a, const INT lda,
                  const double* b, const INT ldb, const double beta, double* c, const INT ldc);
void cblas_cgemmt(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const void* beta, void* c, const INT ldc);
void cblas_zgemmt(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const void* beta, void* c, const INT ldc);

/* symm */
void cblas_ssymm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const float alpha, const float* a, const INT lda,
                 const float* b, const INT ldb, const float beta, float* c, const INT ldc);
void cblas_dsymm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const double alpha, const double* a, const INT lda,
                 const double* b, const INT ldb, const double beta, double* c, const INT ldc);
void cblas_csymm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);
void cblas_zsymm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);

/* hemm */
void cblas_chemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);
void cblas_zhemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const INT m, const INT n, const void* alpha, const void* a, const INT lda,
                 const void* b, const INT ldb, const void* beta, void* c, const INT ldc);

/* syrk */
void cblas_ssyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const float alpha, const float* a, const INT lda, const float beta, float* c, const INT ldc);
void cblas_dsyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const double alpha, const double* a, const INT lda, const double beta, double* c, const INT ldc);
void cblas_csyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const void* alpha, const void* a, const INT lda, const void* beta, void* c, const INT ldc);
void cblas_zsyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const void* alpha, const void* a, const INT lda, const void* beta, void* c, const INT ldc);

/* herk */
void cblas_cherk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const float alpha, const void* a, const INT lda, const float beta, void* c, const INT ldc);
void cblas_zherk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const INT n, const INT k, const double alpha, const void* a, const INT lda, const double beta, void* c, const INT ldc);

/* syr2k */
void cblas_ssyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const float alpha, const float* a, const INT lda,
                  const float* b, const INT ldb, const float beta, float* c, const INT ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const double alpha, const double* a, const INT lda,
                  const double* b, const INT ldb, const double beta, double* c, const INT ldc);
void cblas_csyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const void* beta, void* c, const INT ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const void* beta, void* c, const INT ldc);

/* her2k */
void cblas_cher2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const float beta, void* c, const INT ldc);
void cblas_zher2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                  const INT n, const INT k, const void* alpha, const void* a, const INT lda,
                  const void* b, const INT ldb, const double beta, void* c, const INT ldc);

/* trmm */
void cblas_strmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const float alpha, const float* a, const INT lda, float* b, const INT ldb);
void cblas_dtrmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const double alpha, const double* a, const INT lda, double* b, const INT ldb);
void cblas_ctrmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, void* b, const INT ldb);
void cblas_ztrmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, void* b, const INT ldb);

/* trsm */
void cblas_strsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const float alpha, const float* a, const INT lda, float* b, const INT ldb);
void cblas_dtrsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const double alpha, const double* a, const INT lda, double* b, const INT ldb);
void cblas_ctrsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, void* b, const INT ldb);
void cblas_ztrsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const enum CBLAS_DIAG diag, const INT m, const INT n,
                 const void* alpha, const void* a, const INT lda, void* b, const INT ldb);

#endif /* SEMICOLON_CBLAS_H */
