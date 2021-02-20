#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KB(x) ((x)*1024L)
#define N 8

void vector_sqrt(double *s, double *t, double *u, int n) {
	for(int i=0;i<n;i++) {
		u[i] = sqrt(s[i]*s[i] + t[i]*t[i]);
	}
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpx,*fpy,*fpz;
	double *a,*b,*c,*x,*y,*z;
	int n;
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

	fpa = fopen("./double_a.bin", "wr");
	fpb = fopen("./double_b.bin", "wr");
	fpc = fopen("./double_c.bin", "wr");
	fwrite(a, sizeof(double), n, fpa);
	fwrite(b, sizeof(double), n, fpb);
	fwrite(c, sizeof(double), n, fpc);
	fclose(fpa);
	fclose(fpb);
	fclose(fpc);

	x = (double*)malloc(sizeof(double)*n);
	y = (double*)malloc(sizeof(double)*n);
	z = (double*)malloc(sizeof(double)*n);
	fpx = fopen("./double_a.bin", "r");
	fpy = fopen("./double_b.bin", "r");
	fpz = fopen("./double_c.bin", "rw");
	fread(x, sizeof(double), n, fpx);
	fread(y, sizeof(double), n, fpy);
	fread(z, sizeof(double), n, fpz);
	vector_sqrt(x,y,z,n);
	for(int i=0;i<n;++i) {
		printf("output: %8.3lf\n", z[i]);
	}

	free(a);
	free(b);
	free(c);
	free(x);
	free(y);
	free(z);

	fclose(fpx);
	fclose(fpy);
	fclose(fpz);
}
