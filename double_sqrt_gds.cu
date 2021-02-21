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
	FILE *fpa,*fpb,*fpc;
	int fpx, fpy, fpz;
	double *a,*b,*c;
	double *x_d,*y_d,*z_d;
	int n;
        CUfileDescr_t cf_desc_x;
        CUfileDescr_t cf_desc_y;
        CUfileDescr_t cf_desc_z;
        CUfileHandle_t cf_handle_x;
        CUfileHandle_t cf_handle_y;
        CUfileHandle_t cf_handle_z;
	if(argc < 2) {
		n = N;
	} else {
		n = atoi(argv[1]);
	}
	a = (double*)malloc(sizeof(double)*n);
	b = (double*)malloc(sizeof(double)*n);
	c = (double*)malloc(sizeof(double)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
		c[i] = 0.0;
	}

	fpa = fopen("./double_a.bin", "w");
	fpb = fopen("./double_b.bin", "w");
	fpc = fopen("./double_c.bin", "w");
	fwrite(a, sizeof(double), n, fpa);
	fwrite(b, sizeof(double), n, fpb);
	fwrite(c, sizeof(double), n, fpc);
	fclose(fpa);
	fclose(fpb);
	fclose(fpc);

	cudaMalloc(&x_d, sizeof(double)*n);
	cudaMalloc(&y_d, sizeof(double)*n);
	cudaMalloc(&z_d, sizeof(double)*n);

        cuFileDriverOpen();
        fpx = open("./double_a.bin", O_RDONLY | O_DIRECT);
        fpy = open("./double_b.bin", O_RDONLY | O_DIRECT);
        fpz = open("./double_c.bin", O_RDWR | O_DIRECT);
        cf_desc_x.handle.fd = fpx;
        cf_desc_y.handle.fd = fpy;
        cf_desc_z.handle.fd = fpz;
        cf_desc_x.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cf_desc_y.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cf_desc_z.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        cuFileHandleRegister(&cf_handle_x, &cf_desc_x);
        cuFileHandleRegister(&cf_handle_y, &cf_desc_y);
        cuFileHandleRegister(&cf_handle_z, &cf_desc_z);
        cuFileBufRegister((double*)x_d, sizeof(double)*n, 0);
        cuFileBufRegister((double*)y_d, sizeof(double)*n, 0);
        cuFileBufRegister((double*)z_d, sizeof(double)*n, 0);

	cuFileRead(cf_handle_x, (double*)x_d, sizeof(double)*n, 0, 0);
	cuFileRead(cf_handle_y, (double*)y_d, sizeof(double)*n, 0, 0);
	cuFileRead(cf_handle_z, (double*)z_d, sizeof(double)*n, 0, 0);
	/* cudaMemcpy(z_d, z, sizeof(double)*n, cudaMemcpyHostToDevice); */

        int blocksize = 512;
        int gridsize = (n+(blocksize-1))/blocksize;
        dim3 dimGrid(gridsize,1);
        dim3 dimBlock(blocksize,1,1);
        vector_sqrt<<<dimGrid,dimBlock>>>(x_d,y_d,z_d);

	cuFileWrite(cf_handle_z, (double*)z_d, sizeof(double)*n, 0, 0);
	/* cudaMemcpy(c, z_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;++i) {
		printf("output: %8.3lf\n", c[i]);
	} */
	/* printf("dimGrid:%d, dimBlock:%d\n", gridsize, blocksize); */

	cuFileBufDeregister((double*)x_d);
	cuFileBufDeregister((double*)y_d);
	cuFileBufDeregister((double*)z_d);

	free(a);
	free(b);
	free(c);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	close(fpx);
	close(fpy);
	close(fpz);
	cuFileDriverClose();
}
