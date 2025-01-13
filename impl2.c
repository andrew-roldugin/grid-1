#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Параметры для моделирования процесса:
#define ALPHA 0.25    // Коэффициент теплопроводности
#define DT    0.1    // Шаг по времени для источников

#define DX 1 // Шаг по пространству
#define DY 1 // Шаг по пространству

// Функция для выделения двумерного массива
double **allocate_2D_array(const int rows, const int cols) {
    double **mtx = (double **) malloc(rows * sizeof(double *));
    if (!mtx) {
        perror("Не удалось выделить память для матрицы");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        mtx[i] = (double *) malloc(cols * sizeof(double));
    }

    return mtx;
}

// Функция для чтения начальных условий из файла (вызывается только на процессе rank=0)
void read_initial_conditions(const char *filename, double ***temperature, double ***Q, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Не удалось открыть файл");
    }
    char line[256];
    int r = 0, c = 0;
    *cols = 0;

    // Сначала определим размер матрицы
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        c = 0;

        char *context;
#ifdef _WIN32
        char *token = strtok_s(line, " \t\n", &context);
#else
        char* token = strtok(line, " \t\n");
#endif

        while (token) {
            c++;
#ifdef _WIN32
            token = strtok_s(NULL, " \t\n", &context);
#else
            token = strtok(NULL, " \t\n");
#endif
        }
        // установим актуальное кол-во столбцов
        if (c > *cols) {
            *cols = c;
        }
        r++;
    }
    *rows = r;

    // Выделяем память (глобально, на rank=0)
    *temperature = allocate_2D_array(*rows, *cols);
    *Q = allocate_2D_array(*rows, *cols);
    if (!(*temperature) || !(*Q)) {
        perror("Не удалось выделить память под матрицы теплового поля и источников");
    }

    // Возвращаемся в начало файла и считываем данные
    rewind(file);
    r = 0;
    // цикл по строкам
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        c = 0;

        char *context;
#ifdef _WIN32
        char *token = strtok_s(line, " \t\n", &context);
#else
        char* token = strtok(line, " \t\n");
#endif

        // цикл по столбцам
        while (token && c < *cols) {
            if (token[0] == '+') {
//                *Q[r][c] = atof(token + 1); // Положительный источник
                (*temperature)[r][c] = 0.0;
            } else if (token[0] == '-') {
//                *Q[r][c] = -atof(token + 1); // Отрицательный источник
                (*temperature)[r][c] = 0.0;
            } else {
//                // Нет источника, это температура
//                *Q[r][c] = 0.0;
                (*temperature)[r][c] = atof(token);
            }
#ifdef _WIN32
            token = strtok_s(NULL, " \t\n", &context);
#else
            token = strtok(NULL, " \t\n");
#endif
            c++;
        }
        r++;
    }
    fclose(file);
}

// Функция для записи матрицы в файл
void write_matrix(double **matrix, const int rows, const int cols, FILE *file) {
    if (!file) {
        perror("Не удалось открыть выходной файл");
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.8f ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
}

// Функция для моделирования процесса теплообмена
void evolve_field(double **temperature_field, int rows, int cols, double time) {
    int steps = (int)(time / DT);

    double ** next = allocate_2D_array(rows, cols);

    for (int step = 0; step < steps; step++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                // Уравнение теплопроводности (явная схема)
                next[i][j] = temperature_field[i][j] + ALPHA * DT * (
                        (temperature_field[i + 1][j] - 2 * temperature_field[i][j] + temperature_field[i - 1][j]) / (DX * DX) +
                        (temperature_field[i][j + 1] - 2 * temperature_field[i][j] + temperature_field[i][j - 1]) / (DY * DY)
                );
            }
        }

        // Обновляем текущую матрицу
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                temperature_field[i][j] = next[i][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rows, cols;
    double **full_matrix = NULL;
    double **sources = NULL;
    double time_start, time_finish;

    const char *input_file = argv[1];
    const int time = atoi(argv[2]);
    read_initial_conditions(input_file, &full_matrix, &sources, &rows, &cols);
    evolve_field(full_matrix, rows, cols, time);

    FILE *file = fopen("output-05_01.txt", "w");
    write_matrix(full_matrix, rows, cols, file);

    return 0;
}