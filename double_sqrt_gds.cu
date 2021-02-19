#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include "cufile.h"

#define KB(x) ((x)*1024L)
#define N 8

__global__ void vector_sqrt(double *s, double *t, double *u) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        u[idx] = sqrt(s[idx]*s[idx] + t[idx]*t[idx]);
        /* printf("idx:%d, blockId.x:%d, threadIdx.x:%d\n",idx, blockIdx.x, threadIdx.x); */
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpz;
	int fpc, fpd, fpx;
	double *a,*b,*z,*c,*d,*x;
	double *c_d,*d_d,*x_d;
	int n;
        CUfileDescr_t cf_desc_c;
        CUfileDescr_t cf_desc_d;
        CUfileDescr_t cf_desc_x;
        CUfileHandle_t cf_handle_c;
        CUfileHandle_t cf_handle_d;
        CUfileHandle_t cf_handle_x;
	if(argc < 2) {
		n = N;
	} else {
		n = atoi(argv[1]);
	}
	a = (double*)malloc(sizeof(double)*n);
	b = (double*)malloc(sizeof(double)*n);
	z = (double*)malloc(sizeof(double)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
		z[i] = 0.0;
	}

	fpa = fopen("./double_a.bin", "wr");
	fpb = fopen("./double_b.bin", "wr");
	fpz = fopen("./double_z.bin", "wr");
	fwrite(a, sizeof(double), n, fpa);
	fwrite(b, sizeof(double), n, fpb);
	fwrite(z, sizeof(double), n, fpz);
	fclose(fpa);
	fclose(fpb);
	fclose(fpz);

	c = (double*)malloc(sizeof(double)*n);
	d = (double*)malloc(sizeof(double)*n);
	x = (double*)malloc(sizeof(double)*n);
	cudaMalloc(&c_d, sizeof(double)*n);
	cudaMalloc(&d_d, sizeof(double)*n);
	cudaMalloc(&x_d, sizeof(double)*n);

        cuFileDriverOpen();
        fpc = open("./double_a.bin", O_RDONLY | O_DIRECT);
        fpd = open("./double_b.bin", O_RDONLY | O_DIRECT);
        fpx = open("./double_z.bin", O_RDWR | O_DIRECT);
        cf_desc_c.handle.fd = fpc;
        cf_desc_d.handle.fd = fpd;
        cf_desc_x.handle.fd = fpx;
        cf_desc_c.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cf_desc_d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cf_desc_x.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        cuFileHandleRegister(&cf_handle_c, &cf_desc_c);
        cuFileHandleRegister(&cf_handle_d, &cf_desc_d);
        cuFileHandleRegister(&cf_handle_x, &cf_desc_x);
        cuFileBufRegister((double*)c_d, sizeof(double)*n, 0);
        cuFileBufRegister((double*)d_d, sizeof(double)*n, 0);
        cuFileBufRegister((double*)x_d, sizeof(double)*n, 0);

	cuFileRead(cf_handle_c, (double*)c_d, sizeof(double)*n, 0, 0);
	cuFileRead(cf_handle_d, (double*)d_d, sizeof(double)*n, 0, 0);
	cuFileRead(cf_handle_x, (double*)x_d, sizeof(double)*n, 0, 0);
	/* cudaMemcpy(x_d, x, sizeof(double)*n, cudaMemcpyHostToDevice); */

        int blocksize = 512;
        int gridsize = (n+(blocksize-1))/blocksize;
        dim3 dimGrid(gridsize,1);
        dim3 dimBlock(blocksize,1,1);
        vector_sqrt<<<dimGrid,dimBlock>>>(c_d,d_d,x_d);

	cuFileWrite(cf_handle_x, (double*)x_d, sizeof(double)*n, 0, 0);
	/* cudaMemcpy(x, x_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;++i) {
		printf("output: %8.3lf\n", x[i]);
	} */
	/* printf("dimGrid:%d, dimBlock:%d\n", gridsize, blocksize); */

	cuFileBufDeregister((double*)c_d);
	cuFileBufDeregister((double*)d_d);
	cuFileBufDeregister((double*)x_d);

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);
	cudaFree(c_d);
	cudaFree(d_d);
	cudaFree(x_d);

	close(fpc);
	close(fpd);
	close(fpx);
	cuFileDriverClose();
}
