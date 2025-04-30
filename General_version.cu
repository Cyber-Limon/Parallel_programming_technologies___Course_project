#define block_size 256
#include <iostream>
#include <fstream>
#include <vector>
#include "mmio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusparse.h>



using namespace std;

__global__ void SpMV(double* values_cu, int* col_indices_cu, int* row_ptr_cu, double* vec_cu, double* res_cu, int M) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;

	if (row < M) {
		double sum = 0;

		int row_start = row_ptr_cu[row];
		int row_stop = row_ptr_cu[row + 1];

		for (int j = row_start; j < row_stop; j++)
			sum += values_cu[j] * vec_cu[col_indices_cu[j]];

		res_cu[row] = sum;
	}
}



void check_results(double* res_cu, double* res_cpp, int M) {
	for (int i = 0; i < M; i++) {
		double diff = abs(res_cu[i] - res_cpp[i]) / max(abs(res_cu[i]), abs(res_cpp[i]));

		if (diff > 1e-8) {
			printf("- Check completed incorrectly\n");
			return;
		}
	}

	printf("- Check completed correctly\n");
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
	
	const string fpath = "Serena.mtx";

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
	int count = 0;

	for (int i = 0; i < nnz; i++) {
		if (val[i] != 0) {
			values[count]      = val[i];
			col_indices[count] = J[i];
			row_ptr[I[i] + 1]++;
			count++;
		}
	}
	
	for (int i = 0; i < M; i++) 
		row_ptr[i + 1] += row_ptr[i];

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

	// Создание единичного и результирующего векторов // 

	double* vec     = new double[N];
	double* res_cpp = new double[M] {0};

	for (int i = 0; i < N; i++)
		vec[i] = 1;



	// Перемножение матрицы на вектор // 

	double start_time = clock();
	for (int i = 0; i < M; i++)
		for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
			res_cpp[i] += values[j] * vec[col_indices[j]];
	double end_time = clock();

	cout << "\n\n\nSEQUENTIAL PROGRAM" << endl;
	cout << "- The resulting vector: ";

	for (int i = 0; i < 10; i++)
		cout << res_cpp[i] << " ";
	cout << endl;

	cout << "- Time: " << (end_time - start_time) / CLOCKS_PER_SEC * 1000 << endl;





	// --- Параллельная программа --- //

	// Создание переменных для замера времени //

	float time_transaction1 = 0;
	float time_transaction2 = 0;
	float time_calculation  = 0;
	cudaEvent_t start_transaction1, stop_transaction1, start_transaction2, stop_transaction2, start_calculation, stop_calculation;
	cudaEventCreate(&start_transaction1);
	cudaEventCreate(&stop_transaction1 );
	cudaEventCreate(&start_transaction2);
	cudaEventCreate(&stop_transaction2 );
	cudaEventCreate(&start_calculation );
	cudaEventCreate(&stop_calculation  );



	// Создание массивов //

	double *res = new double[M];
	double *values_cu, *vec_cu, *res_cu;
	int    *col_indices_cu, *row_ptr_cu;
	int    size_double = sizeof(double);
	int    size_int    = sizeof(int);



	// Выделение памяти //

	cudaMalloc((void**)&values_cu,      count_0 * size_double);
	cudaMalloc((void**)&col_indices_cu, count_0 * size_int   );
	cudaMalloc((void**)&row_ptr_cu,     (M + 1) * size_int   );
	cudaMalloc((void**)&vec_cu,         N       * size_double);
	cudaMalloc((void**)&res_cu,         M       * size_double);



	// Инициализация результирующего массива нулями //

	cudaMemset(res_cu, 0, M * size_double);



	// Копирование с "CPU" на "GPU" //

	cudaEventRecord(start_transaction1, 0);
	cudaMemcpy(values_cu,      values,      count_0 * size_double, cudaMemcpyHostToDevice);
	cudaMemcpy(col_indices_cu, col_indices, count_0 * size_int,    cudaMemcpyHostToDevice);
	cudaMemcpy(row_ptr_cu,     row_ptr,     (M + 1) * size_int,    cudaMemcpyHostToDevice);
	cudaMemcpy(vec_cu,         vec,         N       * size_double, cudaMemcpyHostToDevice);
	cudaEventRecord(stop_transaction1,  0);



	// Перемножение матрицы на вектор //

	cudaEventRecord(start_calculation, 0);
	SpMV <<< (M + block_size - 1) / block_size, block_size >>> (values_cu, col_indices_cu, row_ptr_cu, vec_cu, res_cu, M);
	cudaEventRecord(stop_calculation,  0);



	// Копирование с "GPU" на "CPU" //

	cudaEventRecord(start_transaction2, 0);
	cudaMemcpy(res, res_cu, M * size_double, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_transaction2,  0);



	// Синхронизация и проверка корректности //

	cudaDeviceSynchronize();

	cout << "\n\n\nPARALLEL PROGRAM" << endl;

	check_results(res, res_cpp, M);



	// Время выполнения //

	cudaEventElapsedTime(&time_transaction1, start_transaction1, stop_transaction1);
	cudaEventElapsedTime(&time_calculation,  start_calculation,  stop_calculation );
	cudaEventElapsedTime(&time_transaction2, start_transaction2, stop_transaction2);

	printf("- Time copying from CPU to GPU: %.2f \n",   time_transaction1);
	printf("- Time calculation:             %.2f \n",   time_calculation );
	printf("- Time copying from GPU to CPU: %.2f \n",   time_transaction2);

	

	// Сохранение вектора //

	save_vector("Sharigin_MS___Serena___Without_cuSPARSE.mtx", res, M);





	// --- Параллельная программа с использованием "cuSPARSE" --- //

	// Создание переменных для замера времени //

	float time_transaction_SP = 0;
	float time_calculation_SP = 0;
	cudaEvent_t start_transaction_SP, stop_transaction_SP, start_calculation_SP, stop_calculation_SP;
	cudaEventCreate(&start_transaction_SP);
	cudaEventCreate(&stop_transaction_SP );
	cudaEventCreate(&start_calculation_SP);
	cudaEventCreate(&stop_calculation_SP );



	// Создание дескриптора //

	cusparseHandle_t handle;
	cusparseCreate(&handle);



	// Дескрипторы матрицы и векторов (единичного и результирующего) //

	cusparseSpMatDescr_t mat;
	cusparseDnVecDescr_t vec_vec, vec_res;
		
	cusparseCreateCsr(
		&mat,                     
		M, N, count_0,                 
		row_ptr_cu,                
		col_indices_cu,            
		values_cu,                 
		CUSPARSE_INDEX_32I,        
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO,  
		CUDA_R_64F                 
	);
	
	cusparseCreateDnVec(&vec_vec, N, vec_cu, CUDA_R_64F);
	cusparseCreateDnVec(&vec_res, M, res_cu, CUDA_R_64F);



	// Коэффициенты перед матрицей и вектором //

	double alpha = 1.0;  
	double beta  = 0.0;   



	// Выбор алгоритма //

	cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG1;



	// Буфер //

	size_t bufferSize = 0;
	void* Buffer = nullptr;

	cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, mat, vec_vec, &beta, vec_res, CUDA_R_64F, alg, &bufferSize
	);

	cudaMalloc(&Buffer, bufferSize);



	// Перемножение матрицы на вектор //

	cudaEventRecord(start_calculation_SP, 0);
	cusparseSpMV(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, mat, vec_vec, &beta, vec_res, CUDA_R_64F, alg, Buffer
	);
	cudaEventRecord(stop_calculation_SP,  0);



	// Копирование с "GPU" на "CPU" //

	cudaEventRecord(start_transaction_SP, 0);
	cudaMemcpy(res, res_cu, M * sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_transaction_SP,  0);



	// Проверка корректности //

	cout << "\n\n\nPARALLEL PROGRAM (cuSPARSE)" << endl;

	check_results(res, res_cpp, M);



	// Время выполнения //

	cudaEventElapsedTime(&time_calculation_SP, start_calculation_SP, stop_calculation_SP);
	cudaEventElapsedTime(&time_transaction_SP, start_transaction_SP, stop_transaction_SP);

	printf("- Time copying from CPU to GPU: %.2f \n", time_transaction1  );
	printf("- Time calculation:             %.2f \n", time_calculation_SP);
	printf("- Time copying from GPU to CPU: %.2f \n", time_transaction_SP);
		


	// Сохранение вектора //

	save_vector("Sharigin_MS___Serena___With_cuSPARSE.mtx", res, M);





	// --- Освобождение памяти --- // 

	delete[] I;
	delete[] J;
	delete[] val;

	delete[] values;
	delete[] col_indices;
	delete[] row_ptr;

	delete[] vec;
	delete[] res_cpp;



	cudaEventDestroy(start_transaction1);
	cudaEventDestroy(stop_transaction1 );
	cudaEventDestroy(start_transaction2);
	cudaEventDestroy(stop_transaction2 );
	cudaEventDestroy(start_calculation );
	cudaEventDestroy(stop_calculation  );

	cudaFree(values_cu);
	cudaFree(col_indices_cu);
	cudaFree(row_ptr_cu);
	cudaFree(vec_cu);
	cudaFree(res_cu);

	delete[] res;



	cusparseDestroy(handle);

	cusparseDestroySpMat(mat);
	cusparseDestroyDnVec(vec_vec);
	cusparseDestroyDnVec(vec_res);
	
	if (Buffer) cudaFree(Buffer);
	
	
	
	cout << "\n\n\n\n";
	
	return 0;
}
