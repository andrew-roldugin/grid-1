#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MATRIX_SIZE 100
#define NUM_ITERATIONS 100
#define MAX_LINE_LENGTH 10000

// Параметры для моделирования процесса:
#define ALPHA 0.25    // Коэффициент теплопроводности
#define DT    0.1    // Шаг по времени для источников

#define DELTA_X 1.0 // Шаг по пространству
#define DELTA_Y 1.0 // Шаг по пространству

void evolve_field(const char *sources, int local_rows, int start_row, double *const *local_current, double **local_new,
                  int iter);

void calc_bounds(const char *sources, int local_rows, int start_row, double *const *local_current, double **local_new,
                 int iter);

// Функция для выделения двумерного массива
double **allocate_2D_array(int rows, int cols) {
    double *data = (double *) malloc(rows * cols * sizeof(double));
    if (!data) {
        perror("Не удалось выделить память для матрицы");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double **array = (double **) malloc(rows * sizeof(double *));
    if (!array) {
        perror("Не удалось выделить память для указателей матрицы");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

// Функция для освобождения двумерного массива
void free_2D_array(double **array, int rows) {
    free(array[0]);
    free(array);
}

// Функция для чтения матрицы из файла
double **read_matrix(const char *filename, char *sources) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Не удалось открыть входной файл");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double **matrix = allocate_2D_array(MATRIX_SIZE, MATRIX_SIZE);

    char line[MAX_LINE_LENGTH];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < MATRIX_SIZE) {
        // Пропускаем комментарии
        if (line[0] == '#') continue;

        char *token = strtok(line, " \t\n");
        int col = 0;
        while (token && col < MATRIX_SIZE) {
            if (token[0] == '-') {
                sources[row * MATRIX_SIZE + col] = '-';
                matrix[row][col] = atof(token + 1);
            } else if (token[0] == '+') {
                sources[row * MATRIX_SIZE + col] = '+';
                matrix[row][col] = atof(token + 1);
            } else if (token[0] == '~') {
                sources[row * MATRIX_SIZE + col] = '~';
                matrix[row][col] = atof(token + 1);
            } else {
                sources[row * MATRIX_SIZE + col] = 'N'; // N - обычный узел
                matrix[row][col] = atof(token);
            }
            token = strtok(NULL, " \t\n");
            col++;
        }
        row++;
    }

    fclose(file);
    return matrix;
}

// Функция для записи матрицы в файл
void write_matrix(const char *filename, double *matrix) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Не удалось открыть выходной файл");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            fprintf(file, "%.12f ", matrix[i * MATRIX_SIZE + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Функция для записи матрицы в файл
void write_matrix2(const char *filename, double **matrix) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Не удалось открыть выходной файл");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            fprintf(file, "%.2f ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    FILE *file2 = fopen("matrices", "w+");

    int rank, size;
    double **full_matrix = NULL;
    char *sources = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Использование: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Имя файла с входными данными
    const char *input_file = argv[1];

    // Корневой процесс читает матрицу
    if (rank == 0) {
        sources = (char *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(char));
        if (!sources) {
            perror("Не удалось выделить память для источников");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        full_matrix = read_matrix(input_file, sources);
    }

    // Распространяем массив источников
    if (rank != 0) {
        sources = (char *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(char));
        if (!sources) {
            perror("Не удалось выделить память для источников");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(sources, MATRIX_SIZE * MATRIX_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("worker %d: sent sources matrix\n", rank);
    fflush(stdout);

    // Распространяем матрицу
    if (rank != 0) {
        full_matrix = allocate_2D_array(MATRIX_SIZE, MATRIX_SIZE);
    }
    // Преобразуем двумерный массив в одномерный для передачи
    MPI_Bcast(&(full_matrix[0][0]), MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("worker %d: sent main matrix\n", rank);
    fflush(stdout);

    // Определяем количество строк на процесс
    int rows_per_proc = MATRIX_SIZE / size;
    int remainder = MATRIX_SIZE % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Определяем смещение
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    printf("Worker %d takes %d rows (from %d)\n", rank, local_rows, start_row + 1);
    fflush(stdout);

    // Выделяем память для локальной части матрицы с дополнительными строками для обмена (ghost rows)
    double **local_current = allocate_2D_array(local_rows + 2, MATRIX_SIZE);
    double **local_new = allocate_2D_array(local_rows, MATRIX_SIZE);

    // Инициализируем локальную текущую матрицу с граничными строками
    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (i == 0 || i == local_rows + 1) {
                // Граничные условия сверху и снизу: фиксированные температуры
                if ((start_row + i - 1) < 0 || (start_row + i - 1) >= MATRIX_SIZE) {
                    local_current[i][j] = 0.0;
                } else if (i == 0 && start_row == 0) {
                    // Верхняя граница
                    local_current[i][j] = full_matrix[0][j];
                } else if (i == local_rows + 1 && (start_row + i - 1) == MATRIX_SIZE - 1) {
                    // Нижняя граница
                    local_current[i][j] = full_matrix[MATRIX_SIZE - 1][j];
                } else {
                    local_current[i][j] = 0.0; // Внутренние граничные строки для обмена
                }
            } else {
                int global_row = start_row + i - 1;
                local_current[i][j] = full_matrix[global_row][j];
            }
        }
    }

    printf("Worker %d | before evo\n", rank);
    fflush(stdout);

    // Итерации эволюции
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        MPI_Status status;

        // Обмен верхней строкой
        if (rank > 0) {
            // Отправляем первую строку текущего процесса соседу сверху с тегом 0
            // И одновременно получаем нижнюю границу соседу сверху с тегом 1
            MPI_Sendrecv(local_current[1], MATRIX_SIZE, MPI_DOUBLE, rank - 1, 0,
                         local_current[0], MATRIX_SIZE, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
        } else {
            // Для самого верхнего процесса устанавливаем граничную строку равной фиксированному значению
            for (int j = 0; j < MATRIX_SIZE; j++) {
                local_current[0][j] = 0.0; // Или любое другое фиксированное значение
            }
        }
        printf("Worker %d | ifter if_1\n", rank);
        fflush(stdout);

        // Обмен нижней строкой
        if (rank < size - 1) {
            // Отправляем последнюю строку текущего процесса соседу снизу с тегом 1
            // И одновременно получаем верхнюю границу соседу снизу с тегом 0
            MPI_Sendrecv(local_current[local_rows], MATRIX_SIZE, MPI_DOUBLE, rank + 1, 1,
                         local_current[local_rows + 1], MATRIX_SIZE, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        } else {
            // Для самого нижнего процесса устанавливаем граничную строку равной фиксированному значению
            for (int j = 0; j < MATRIX_SIZE; j++) {
                local_current[local_rows + 1][j] = 0.0; // Или любое другое фиксированное значение
            }
        }

        printf("Worker %d | ifter if_2\n", rank);
        fflush(stdout);

        printf("Worker %d | start updating field\n", rank);
        fflush(stdout);

        if (!file2) {
            perror("Не удалось открыть выходной файл");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < local_rows + 2; i++) {
            fprintf(file2, "Worker %d ", rank);
            for (int j = 0; j < MATRIX_SIZE; j++) {
                fprintf(file2, "%f ", local_current[i][j]);
            }
            fprintf(file2, "\n");
        }
        fprintf(file2, "\n");
        fflush(file2);

        // Вычисление новых температур
        evolve_field(sources, local_rows, start_row, local_current, local_new, iter);
        calc_bounds(sources, local_rows, start_row, local_current, local_new, iter);

        // Обновляем текущую матрицу
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                local_current[i][j] = local_new[i - 1][j];
            }
        }
    }

    printf("Worker %d | assembling...\n", rank);
    fflush(stdout);

    double *final_matrix = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

// Подготавливаем массивы для сбора данных
    if (rank == 0) {
        final_matrix = (double *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        recvcounts = (int *) malloc(size * sizeof(int));
        displs = (int *) malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * MATRIX_SIZE;
            displs[i] = offset * MATRIX_SIZE;
            offset += rows;
        }
    }
    printf("Worker %d | debug_1...\n", rank);
    fflush(stdout);

// Подготовка локальных данных для отправки
    double *send_buffer = (double *) malloc(local_rows * MATRIX_SIZE * sizeof(double));
    for (int i = 0; i < local_rows; i++) {
        memcpy(&send_buffer[i * MATRIX_SIZE], local_new[i], MATRIX_SIZE * sizeof(double));
    }
    printf("Worker %d | debug_2...\n", rank);
    fflush(stdout);


// Собираем все части матрицы
    MPI_Gatherv(send_buffer, local_rows * MATRIX_SIZE, MPI_DOUBLE,
                final_matrix, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    printf("Worker %d | debug_3...\n", rank);
    fflush(stdout);

    // Корневой процесс записывает итоговую матрицу
    if (rank == 0) {

        write_matrix("output.txt", final_matrix);
        free(final_matrix);
        free(recvcounts);
        free(displs);
        fclose(file2);

    }

    free_2D_array(local_new, local_rows);
    free(send_buffer);

    MPI_Finalize();
    return 0;
}

void evolve_field(const char *sources, int local_rows, int start_row, double *const *local_current, double **local_new,
                  int iter) {
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j < MATRIX_SIZE - 1; j++) {
            // Среднее значение соседних ячеек
            double up = local_current[i - 1][j];
            double down = local_current[i + 1][j];
            double left = local_current[i][j - 1];
            double right = local_current[i][j + 1];
            double center = local_current[i][j];

            // Применение источников тепла
            char source = sources[(start_row + i - 1) * MATRIX_SIZE + j];
            double source_value = 0.0;
            if (source == '-') {
                source_value = -0.1 * center; // Поглощение тепла
            } else if (source == '+') {
                source_value = 0.1 * (100.0 - center); // Выделение тепла
            } else if (source == '~') {
                // Переменный источник: синусоидальный сигнал
                source_value = 0.05 * sin(iter * 0.1);
            }

            // Обновление температуры
//                double newValue = DT * (ALPHA * ((up - 2.0 * center + down) / (DELTA_X * DELTA_X)
//                                                 + (left - 2.0 * center + right) / (DELTA_Y * DELTA_Y)) + source_value);
//                local_new[i - 1][j] = newValue;
            local_new[i - 1][j] = 0.25 * (up + down + left + right) + source_value;
        }
    }
}

void calc_bounds(const char *sources, int local_rows, int start_row, double *const *local_current, double **local_new,
                 int iter) {
    // Обработка граничных столбцов (левая и правая границы остаются фиксированными)
    for (int i = 1; i <= local_rows; i++) {
        int j = 0;
        double up = local_current[i - 1][j];
        double down = local_current[i + 1][j];
        double left = local_current[i][j]; // изолированная левая граница
        double right = local_current[i][j + 1];
        double center = local_current[i][j];

        char source = sources[(start_row + i - 1) * MATRIX_SIZE + j];
        double source_value = 0.0;
        if (source == '-') {
            source_value = -0.1 * center;
        } else if (source == '+') {
            source_value = 0.1 * (100.0 - center);
        } else if (source == '~') {
            source_value = 0.05 * sin(iter * 0.1);
        }

        local_new[i - 1][j] = 0.25 * (up + down + left + right) + source_value;;

        j = MATRIX_SIZE - 1;
        up = local_current[i - 1][j];
        down = local_current[i + 1][j];
        left = local_current[i][j - 1];
        right = local_current[i][j]; // изолированная правая граница
        center = local_current[i][j];

        source = sources[(start_row + i - 1) * MATRIX_SIZE + j];
        source_value = 0.0;
        if (source == '-') {
            source_value = -0.1 * center;
        } else if (source == '+') {
            source_value = 0.1 * (100.0 - center);
        } else if (source == '~') {
            source_value = 0.05 * sin(iter * 0.1);
        }

        local_new[i - 1][j] = 0.25 * (up + down + left + right) + source_value;
    }
}
