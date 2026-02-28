#include <stdio.h>
#include <cblas.h>

extern char* openblas_get_corename(void);
extern char* openblas_get_config(void);

int main(void) {
    printf("OpenBLAS Core: %s\n", openblas_get_corename());
    printf("OpenBLAS Config: %s\n", openblas_get_config());
    return 0;
}
