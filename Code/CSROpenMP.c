#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct 
{
	unsigned int nz;   
	unsigned int rows; 
	unsigned int cols;      
	unsigned int *col; 
	unsigned int *ptr; 
	double *val;  	
} Matrix;

typedef struct 
{
	unsigned int nz;   
	unsigned int rows; 
	unsigned int cols;   
	unsigned int *row; 
	unsigned int *col; 
	double *val;   
} Mtr;

void save_output(double *v, unsigned int size);
int load_matrix(char* filename, Mtr *mm);
int read_size(FILE *f, int *M, int *N, int *nz);
void free_matrix(Mtr *m);
void Sort(unsigned int *A, unsigned int *A2, double *A3, unsigned int n);

int main(int argc, char *argv[]){
	unsigned int i, iterations=8;
	Matrix csr;
	char* filename;
	double *x, *y;
	
    if (argc < 4) 
	{
		printf("Usage: %s Matrix Iterations Threads\n", argv[0]);
		exit(0);
	}
		
    iterations = atoi(argv[2]);
	filename = argv[1];
	
	csr.val = NULL;
	csr.col = NULL;
	csr.ptr = NULL;
	csr.nz = csr.rows = csr.cols = 0;
	
	printf("Loading the input matrix \"%s\"\n", filename);
	Mtr temp;
	temp.val = NULL;
	temp.row = NULL;
	temp.col = NULL;
	temp.nz = temp.rows = temp.cols = 0;
	load_matrix(filename, &temp);
	unsigned int tot = 0;
	Mtr mm;
	
	csr.val = (double *) malloc(temp.nz * sizeof(double));
	csr.col = (unsigned int *) malloc(temp.nz * sizeof(unsigned int));
	csr.ptr = (unsigned int *) malloc(((temp.rows)+1) * sizeof(unsigned int));
	csr.nz = temp.nz;
	csr.rows = temp.rows;
	csr.cols = temp.cols;
	
	mm.val = (double *) malloc(temp.nz * sizeof(double));
	mm.row = (unsigned int *) malloc(temp.nz * sizeof(unsigned int));
	mm.col = (unsigned int *) malloc(temp.nz * sizeof(unsigned int));
	mm.nz = temp.nz;
	mm.rows = temp.rows;
	mm.cols = temp.cols;
	
	for(i = 0; i < temp.nz; i++)
		mm.val[i] = temp.val[i]; 
	for(i = 0; i < temp.nz; i++)
		mm.row[i] = temp.row[i]; 
	for(i = 0; i < temp.nz; i++)
		mm.col[i] = temp.col[i]; 
	Sort(mm.row, mm.col, mm.val, mm.nz);
	
	csr.ptr[0] = tot;
	for(i = 0; i < mm.rows; i++){
		while(tot < mm.nz && mm.row[tot] == i){
			csr.val[tot] = mm.val[tot];
			csr.col[tot] = mm.col[tot];
			tot++;
		}
		csr.ptr[i+1] = tot;
	}
	
	free_matrix(&mm);
	free_matrix(&temp);

	x = (double *)malloc(csr.cols * sizeof(double));
	y = (double *)malloc(csr.rows * sizeof(double));
	for(i = 0; i < csr.cols; i++)
	{
		x[i] = 1;
	}

	printf("Computing the results for %u iterations (Parallel Version)...\n", iterations);

	int nthreads, tid;
	double start, end; 
	unsigned int il,jl,endl,*col1,*ptr1;
	double *val1;
    start = omp_get_wtime();
	omp_set_num_threads(atoi(argv[3]));
    #pragma omp parallel shared(csr,y,x) private(tid,il,jl,endl,val1,col1,ptr1) 
    {
		tid = omp_get_thread_num();
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
		#pragma omp for schedule(dynamic)
		for(il=0; il < iterations; il++){
			val1 = csr.val;
			col1 = csr.col;
			ptr1 = csr.ptr;
			end = 0;
			for(il = 0; il < csr.rows; il++)
			{
				y[il] = 0.0;
				jl = endl;
				endl = ptr1[il+1];
				for( ; jl < endl; jl++)
				{
					y[il] += val1[jl] * x[col1[jl]];
				}
			}
		}
	}
	end = omp_get_wtime(); 
    printf("The execution time is: %f us\n", (end - start)*1000000);
	save_output(y, csr.rows);
	
	free(x);
	free(y);
	if(csr.val != NULL) free(csr.val);
	if(csr.col != NULL) free(csr.col);
	if(csr.ptr != NULL) free(csr.ptr);
	csr.val = NULL;
	csr.col = NULL;
	csr.ptr = NULL;
	csr.nz = csr.rows = csr.cols = 0;
	exit(1);
}

void save_output(double *v, unsigned int size)
{
	unsigned int i;
	FILE *result;
	result = fopen("result.txt", "w");
	for (i = 0; i < size; i++)
		fprintf(result, "%e \n", v[i]);
	fclose(result);
}

void Sort(unsigned int *el1, unsigned int *el2, double *el3, unsigned int n) 
{
	unsigned int swaps = 0,width,i,*t;
	unsigned int *el4 = (unsigned int *)malloc(n * sizeof(unsigned int));
	unsigned int *el5 = (unsigned int *)malloc(n * sizeof(unsigned int));
	double *el6 = (double *)malloc(n * sizeof(double)), *td;
	for (width = 1; width < n; width = 2 * width)
	{
		for (i = 0; i < n; i = i + 2 * width)
		{
			unsigned int megemin1 = (i+width<n)?i+width:n;
			unsigned int megemin2 = (i+2*width<n)?i+2*width:n;
			unsigned int i0 = i, i1 = megemin1, j;
		
			for (j = i; j < megemin2; j++)
			{
				if (i0 < megemin1 && (i1 >= megemin2 || el1[i0] <= el1[i1]))
				{
					el4[j] = el1[i0];
					el5[j] = el2[i0];
					el6[j] = el3[i0];
					i0++;
				} 
				else 
				{
					el4[j] = el1[i1];
					el5[j] = el2[i1];
					el6[j] = el3[i1];
					i1++;
				}
			}
		}
		t = el4; 
		el4 = el1; 
		el1 = t;
		t = el5; 
		el5 = el2; 
		el2 = t;
		td = el6; 
		el6 = el3; 
		el3 = td;
		swaps++;
   }
	if(swaps%2 == 1)
	{
		t = el4; 
		el4 = el1; 
		el1 = t;
		t = el5; 
		el5 = el2; 
		el2 = t;
		td = el6; 
		el6 = el3; 
		el3 = td;
		for(i = 0; i < n; i++)
			el1[i] = el4[i];
		for(i = 0; i < n; i++)
			el2[i] = el5[i];
		for(i = 0; i < n; i++)
            el3[i] = el6[i];
	}
	free(el4);
	free(el5);
	free(el6);
}

int load_matrix(char* filename, Mtr *mm)
{
	typedef char type1[4];
	FILE *file;
	type1 code;
	int m, n, nz, i;
	unsigned int dc = 0;
	
	file = fopen(filename, "r");
	if(file == NULL){
		fprintf(stderr, "can't open the input file \"%s\"\n", filename);
		exit(0);
	}
	
	read_size(file, &m, &n, &nz);
		
	mm->rows = m;
	mm->cols = n;
	mm->nz = nz;
	
	mm->val = (double *) malloc(nz * sizeof(double));
	mm->row = (unsigned int *) malloc(nz * sizeof(unsigned int));
	mm->col = (unsigned int *) malloc(nz * sizeof(unsigned int));
	
	for(i = 1; i < nz+1; i++)
	{
	    fscanf(file, "%d %d %lg\n", &(mm->row[i]), &(mm->col[i]), &(mm->val[i]));
		mm->row[i]--;
		mm->col[i]--;
		if(mm->row[i] == mm->col[i]) { ++dc; }
	}
	
	fclose(file);
	return 1;
}

int read_size(FILE *f, int *M, int *N, int *nz)
{
    char line[1025];
    *M = *N = *nz = 0;
    do 
    {
        if (fgets(line,1025,f) == NULL) 
            return 0;
    }while (line[0] == '%');
    sscanf(line, "%d %d %d", M, N, nz);

    return 0;
}

void free_matrix(Mtr *m){
	if(m->val != NULL) free(m->val);
	if(m->row != NULL) free(m->row);
	if(m->col != NULL) free(m->col);
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->nz = m->rows = m->cols = 0;
}