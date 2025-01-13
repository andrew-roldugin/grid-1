#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#define INF 1000000  // Большое значение для обозначения бесконечности

// CUDA kernel для выполнения одного шага алгоритма Флойда-Уоршелла
__global__ void floydWarshallKernel(int* d_matrix, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Индекс строки
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Индекс столбца

    if (i < n && j < n) {
        int viaK = d_matrix[i * n + k] + d_matrix[k * n + j];
        if (viaK < d_matrix[i * n + j]) {
            d_matrix[i * n + j] = viaK;
        }
    }
}

// Функция для чтения матрицы смежности из файла
std::vector<std::vector<int>> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл");
    }

    int n;
    file >> n;  // Считываем размер матрицы
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> matrix[i][j];
            if (matrix[i][j] == 0 && i != j) {
                matrix[i][j] = INF;  // Заменяем 0 на бесконечность для отсутствующих рёбер
            }
        }
    }

    file.close();
    return matrix;
}

// Функция для записи матрицы в файл
void writeMatrix(const std::string& filename, const std::vector<std::vector<int>>& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл");
    }

    int n = matrix.size();
    file << n << "\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            if (val >= INF) {
                file << "INF ";
            } else {
                file << val << " ";
            }
        }
        file << "\n";
    }

    file.close();
}

int main() {
    // Считываем матрицу смежности из файла
    std::string inputFile = "input.txt";
    std::string outputFile = "output.txt";
    std::vector<std::vector<int>> matrix = readMatrix(inputFile);
    int n = matrix.size();

    // Преобразуем матрицу в одномерный массив для передачи на GPU
    std::vector<int> flatMatrix(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flatMatrix[i * n + j] = matrix[i][j];
        }
    }

    // Выделяем память на устройстве (GPU)
    int* d_matrix;
    cudaMalloc(&d_matrix, n * n * sizeof(int));
    cudaMemcpy(d_matrix, flatMatrix.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Определяем размеры блоков и сетки
    dim3 blockSize(16, 16);  // Размер блока 16x16
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Запускаем алгоритм Флойда-Уоршелла
    for (int k = 0; k < n; ++k) {
        floydWarshallKernel<<<gridSize, blockSize>>>(d_matrix, k, n);
        cudaDeviceSynchronize();  // Синхронизация устройства
    }

    // Копируем результат обратно на хост (CPU)
    cudaMemcpy(flatMatrix.data(), d_matrix, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождаем память на устройстве
    cudaFree(d_matrix);

    // Преобразуем одномерный массив обратно в матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = flatMatrix[i * n + j];
        }
    }

    // Записываем результат в файл
    writeMatrix(outputFile, matrix);

    std::cout << "Результаты сохранены в файл: " << outputFile << std::endl;
    return 0;
}
