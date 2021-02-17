#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KB(x) ((x)*1024L)
#define N 8

void vector_sqrt(float *s, float *t, float *u, int n) {
	for(int i=0;i<n;i++) {
		u[i] = sqrt(s[i]*s[i] + t[i]*t[i]);
	}
}

int main(int argc, char *argv[])
{
	FILE *fpa,*fpb,*fpc,*fpd;
	float *a,*b,*c,*d,*x;
	int n;
	if(argc < 2) {
		n = N;
	} else {
	n = atoi(argv[1]);
	}
	a = (float*)malloc(sizeof(float)*n);
	b = (float*)malloc(sizeof(float)*n);
	for(int i=0;i<n;++i) {
		a[i] = 3.0;
		b[i] = 4.0;
	}

	fpa = fopen("./float_a.bin", "wr");
	fpb = fopen("./float_b.bin", "wr");
	fwrite(a, sizeof(float), n, fpa);
	fwrite(b, sizeof(float), n, fpb);
	fclose(fpa);
	fclose(fpb);

	c = (float*)malloc(sizeof(float)*n);
	d = (float*)malloc(sizeof(float)*n);
	x = (float*)malloc(sizeof(float)*n);
	fpc = fopen("./float_a.bin", "r");
	fpd = fopen("./float_b.bin", "r");
	fread(c, sizeof(float), n, fpc);
	fread(d, sizeof(float), n, fpd);
	vector_sqrt(c,d,x,n);
	for(int i=0;i<n;++i) {
		printf("output: %5.3e\n", x[i]);
	}

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);

	fclose(fpc);
	fclose(fpd);
}
