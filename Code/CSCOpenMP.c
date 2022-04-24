#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct 
{
	unsigned int nz;   
	unsigned int rows; 
	unsigned int cols;      
	unsigned int *row; 
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
int load_matrix(char* filename, Matrix *m);
int read_size(FILE *f, int *M, int *N, int *nz);

int main(int argc, char *argv[])
{
	unsigned int i, iterations=8;
	Matrix m;
	char* filename;
	double *x, *y;

	if (argc < 4) 
	{
		printf("Usage: %s Matrix Iterations Threads\n", argv[0]);
		exit(1);
	}
		
    iterations = atoi(argv[2]);
	filename = argv[1];
	
	m.val = NULL;
	m.row = NULL;
	m.ptr = NULL;
	m.nz = m.rows = m.cols = 0;
	m.val = NULL;
	m.row = NULL;
	m.ptr = NULL;
	m.nz = m.rows = m.cols = 0;
	
	printf("Loading the input matrix \"%s\"\n", filename);
	load_matrix(filename, &m);
	
	x = (double *)malloc(m.cols * sizeof(double));
	y = (double *)malloc(m.rows * sizeof(double));
	for(i = 0; i < m.cols; i++)
	{
		x[i] = 1;
	}

	printf("Computing the results for %u iterations (Parallel Version)...\n", iterations);


    int nthreads, tid;
	double start, end, *value; 
    unsigned int *row, *ptr, il, jl;
	start = omp_get_wtime();
	omp_set_num_threads(atoi(argv[3]));
	#pragma omp parallel shared(m,y,x) private(tid,value,row,ptr,il,jl)
	{
		tid = omp_get_thread_num();
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }

		#pragma omp for schedule(dynamic)
		for(i=0; i < iterations; i++)
		{
			value = m.val;
			row = m.row;
			ptr = m.ptr;
			for(il = 0; il < m.rows; il++)
			{
				y[il] = 0.0;
			}
			
			for(jl = 0; jl < m.cols; jl++)
			{
				for(il = ptr[jl] ; il < ptr[jl+1]; il++)
				{
					y[row[il]] += value[il] * x[jl];
				}
			}
		}
	}
	end = omp_get_wtime(); 
    printf("The execution time is: %f us\n", (end - start)*1000000);
	save_output(y, m.rows);

	// Free resources
	free(x);
	free(y);
	if(m.val != NULL) free(m.val);
	if(m.row != NULL) free(m.row);
	if(m.ptr != NULL) free(m.ptr);
	m.val = NULL;
	m.row = NULL;
	m.ptr = NULL;
	m.nz = m.rows = m.cols = 0;
	
	exit(EXIT_SUCCESS);
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

int load_matrix(char* filename, Matrix *m)
{
	int sym;
	Mtr temp;
	temp.val = NULL;
	temp.row = NULL;
	temp.col = NULL;
	temp.nz = temp.rows = temp.cols = 0;

	typedef char typecode[4];
	FILE *file;
	typecode code;
	int mt,n,nz,i,ival;
	unsigned int dc = 0;
	
	file = fopen(filename, "r");
	if(file == NULL)
	{
		fprintf(stderr, "Can't open the input file \"%s\"\n", filename);
		exit(EXIT_FAILURE);
	}
	
	read_size(file, &mt, &n, &nz);
		
	temp.rows = mt;
	temp.cols = n;
	temp.nz = nz;
	
	temp.val = (double *) malloc(nz * sizeof(double));
	temp.row = (unsigned int *) malloc(nz * sizeof(unsigned int));
	temp.col = (unsigned int *) malloc(nz * sizeof(unsigned int));
	
	for(i = 1; i < nz+1; i++)
	{
	    fscanf(file, "%d %d %lg\n", &(temp.row[i]), &(temp.col[i]), &(temp.val[i]));
		temp.row[i]--;
		temp.col[i]--;
		if(temp.row[i] == temp.col[i]) { ++dc; }
	}
	fclose(file);

	unsigned int i1;
	unsigned int tot = 0;
	m->val = (double *) malloc(temp.nz * sizeof(double));
	m->row = (unsigned int *) malloc(temp.nz * sizeof(unsigned int));
	m->ptr = (unsigned int *) malloc(((temp.rows)+1) * sizeof(unsigned int));
	m->nz = temp.nz;
	m->rows = temp.rows;
	m->cols = temp.cols;
		
	m->ptr[0] = tot;
	for(i1 = 0; i1 < temp.cols; i1++)
	{
		while(tot < temp.nz && temp.col[tot] == i1)
		{
			m->val[tot] = temp.val[tot];
			m->row[tot] = temp.row[tot];
			tot++;
		}
		m->ptr[i1+1] = tot;
	}
	if(temp.val != NULL) free(temp.val);
	if(temp.row != NULL) free(temp.row);
	if(temp.col != NULL) free(temp.col);
	temp.val = NULL;
	temp.row = NULL;
	temp.col = NULL;
	temp.nz = temp.rows = temp.cols = 0;
	return sym;
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
