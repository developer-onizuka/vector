#include <stdio.h>
#include <stdlib.h>

#define KB(x) ((x)*1024L)
#define N 8 

void main()
{
	FILE *fp;
	FILE *fq;
	float *a;
	float *b;
	a = (float*)malloc(sizeof(float)*N);
	for(int i=0;i<N;++i) {
		a[i] = 1.0;
	}
	b = (float*)malloc(sizeof(float)*N);

	fp = fopen("./float.txt", "wr");
	fwrite(a, sizeof(float), N, fp);
	fclose(fp);

	fq = fopen("./float.txt", "r");
	fread(b, sizeof(float), N, fp);
	fclose(fq);

	for(int j=0;j<N;++j) {
		printf("%5.2f, %5.3e\n", a[j], a[j]);
	}
}
