#include <iostream>
#include <fstream>
#include <vector>
#include "mmio.h"
#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>



using namespace std;

void check_results(double *res_par, double* res_seq, int M) {
	for (int i = 0; i < M; i++) {
		double diff = abs(res_par[i] - res_seq[i]) / max(abs(res_par[i]), abs(res_seq[i]));

		if (diff > 1e-5) {
			cout << "- Check completed incorrectly" << endl;
			return;
		}
	}

	cout << "- Check completed correctly" << endl;
}



void save_vector(const string& filename, const double* vec, int size) {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot open file " << filename << endl;
		return;
	}

	file << "%%MatrixMarket matrix array real general\n";
	file << size << " 1\n";

	for (int i = 0; i < size; ++i)
		file << vec[i] << "\n";

	file.close();
	cout << "Vector saved to " << filename << endl;
}



int main() {

	// Считывание матрицы //

	const string fpath = "Hardesty3.mtx";

	int M   = 0;
	int N   = 0;
	int nnz = 0;

	int    *I;
	int    *J;
	double *val;

	int rcode = mm_read_unsymmetric_sparse(fpath.c_str(), &M, &N, &nnz, &val, &I, &J);
	cout << "M = " << M << " N = " << N << " nnz = " << nnz << endl;

	for (int i = 0; i < 10; i++)
		cout << "elem " << i << " row = " << I[i] << " column = " << J[i] << " val = " << val[i] << endl;



	// Подсчёт количества ненулевых строк //

	int count_0 = 0;

	for (int i = 0; i < nnz; i++)
		if (val[i] != 0)
			count_0++;



	// Перевод матрицы из формата "COO" в "CSR" // 

	double *values   = new double[count_0];
	int *col_indices = new int[count_0];
	int *row_ptr     = new int[M + 1] {0};

	int* position = new int[M] {0};

	for (int i = 0; i < nnz; i++)
		if (val[i] != 0)
			row_ptr[I[i] + 1]++;

	for (int i = 0; i < M; i++)
		row_ptr[i + 1] += row_ptr[i];

	for (int i = 0; i < nnz; i++) {
		if (val[i] != 0) {
			int row = I[i];
			int pos = row_ptr[row] + position[row];
			values[pos]      = val[i];
			col_indices[pos] = J[i];
			position[row]++;
		}
	}

	if (row_ptr[M] != count_0) {
		cerr << "Error in filling row_ptr" << endl;
		return 1;
	}

	cout << "\n\n\nCSR-matrix" << endl;

	for (int i = 0; i < 10; i++)
		cout << values[i] << " ";
	cout << endl;

	for (int i = 0; i < 10; i++)
		cout << col_indices[i] << " ";
	cout << endl;

	for (int i = 0; i < 10; i++)
		cout << row_ptr[i] << " ";
	cout << endl;





	// --- Последовательная программа --- //

	// Создание переменных для замера времени //

	double start_time, end_time;



	// Создание единичного и результирующего векторов // 

	double *vec     = new double[N];
	double *res_seq = new double[M] {0};

	for (int i = 0; i < N; i++)
		vec[i] = 1;



	// Перемножение матрицы на вектор // 

	start_time = omp_get_wtime();
	for (int i = 0; i < M; i++)
		for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
			res_seq[i] += values[j] * vec[col_indices[j]];
	end_time   = omp_get_wtime();



	// Вывод результатов //

	cout << "\n\n\nSEQUENTIAL PROGRAM" << endl;
	cout << "- The resulting vector: ";

	for (int i = 0; i < 10; i++)
		cout << res_seq[i] << " ";
	cout << endl;

	cout << "- Time: " << (end_time - start_time) * 1000 << " (ms)" << endl;

	



	// --- Параллельная программа --- //

	// Создание массива //

	double* res_par = new double[M] {0};



	// Перемножение матрицы на вектор // 

	start_time = omp_get_wtime();
#pragma omp parallel num_threads(8) 
	{
#pragma omp for
		for (int i = 0; i < M; i++) {
			double sum = 0;

			for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
				sum += values[j] * vec[col_indices[j]];

			res_par[i] = sum;
		}
	}
	end_time   = omp_get_wtime();



	// Вывод результатов //

	cout << "\n\n\nPARALLEL PROGRAM" << endl;
	cout << "- The resulting vector: ";

	for (int i = 0; i < 10; i++)
		cout << res_par[i] << " ";
	cout << endl;

	cout << "- Time: " << (end_time - start_time) * 1000 << " (ms)" << endl;



	// Проверка корректности и сохранение вектора //

	check_results(res_par, res_seq, M);

	save_vector("Altynguzina_AD___Hardesty3___Without_MKL.mtx", res_par, M);





	// --- Параллельная программа с использованием "MKL" --- //

	// Создание массива //

	double* res_mkl = new double[M] {0};



	// Описатель матрицы //

	sparse_matrix_t mat;
	mkl_sparse_d_create_csr(
		&mat,                      
		SPARSE_INDEX_BASE_ZERO,  
		M, 
		N,
		row_ptr,  
		row_ptr + 1,             
		col_indices,             
		values                   
	);



	// Тип матрицы //

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;



	// Коэффициенты перед матрицей и вектором //

	double alpha = 1.0; 
	double beta  = 0.0;  



	// Перемножение матрицы на вектор // 

	start_time = omp_get_wtime();
	sparse_status_t status = mkl_sparse_d_mv(
		SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		mat,
		descr,
		vec,
		beta,
		res_mkl
	);
	end_time   = omp_get_wtime();



	// Вывод результатов //

	cout << "\n\n\nPARALLEL PROGRAM (MKL)" << endl;
	cout << "- The resulting vector: ";

	for (int i = 0; i < 10; i++)
		cout << res_mkl[i] << " ";
	cout << endl;

	cout << "- Time: " << (end_time - start_time) * 1000 << " (ms)" << endl;



	// Проверка корректности и сохранение вектора //

	check_results(res_mkl, res_seq, M);

	save_vector("Altynguzina_AD___Hardesty3___With_MKL.mtx", res_mkl, M);



	

	// --- Освобождение памяти --- // 

	delete[] I;
	delete[] J;
	delete[] val;

	delete[] values;
	delete[] col_indices;
	delete[] row_ptr;

	delete[] position;

	delete[] vec;
	delete[] res_seq;


	
	delete[] res_par;



	delete[] res_mkl;

	mkl_sparse_destroy(mat);



	cout << "\n\n\n\n";

	return 0;
}
