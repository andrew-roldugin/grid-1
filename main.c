#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

#define NUM_ITER 1000 // Количество итераций

// Параметры для моделирования процесса:
#define ALPHA 0.1    // Коэффициент теплопроводности
#define DT    0.1    // Шаг по времени для источников

#define DELTA_X 1.0 // Шаг по пространству
#define DELTA_Y 1.0 // Шаг по пространству


// Функция для чтения начальных условий из файла
void read_initial_conditions(const char *filename, double ***temperature, double ***Q, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Не удалось открыть файл");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int r = 0, c = 0;

    // Сначала определим размер матрицы
    *cols = 0; // Убедимся, что cols инициализирован
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue; // Пропускаем комментарии и пустые строки
        c = 0;

        char *context;
        char *token = strtok_s(line, " \t\n", &context);
        while (token) {
            c++;
            token = strtok_s(NULL, " \t\n", &context);
        }
        if (c > *cols)
            *cols = c;
        r++;
    }
    *rows = r;

    // Выделяем память для матриц
    *temperature = (double **) malloc(*rows * sizeof(double *));
    *Q = (double **) malloc(*rows * sizeof(double *));
    for (int i = 0; i < *rows; i++) {
        (*temperature)[i] = (double *) malloc(*cols * sizeof(double));
        (*Q)[i] = (double *) calloc(*cols, sizeof(double)); // Инициализируем Q нулями
        if (!(*temperature)[i] || !(*Q)[i]) {
            perror("Не удалось выделить память для матриц");
            exit(EXIT_FAILURE);
        }
    }

    rewind(file);
    r = 0;
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n') continue; // Пропускаем комментарии и пустые строки
        c = 0;
        char *context;
        char *token = strtok_s(line, " \t\n", &context);

        // Считываем данные в матрицы
        while (token && c < *cols) {
            if (token[0] == '+') {
                (*Q)[r][c] = atof(token + 1); // Положительный источник
                (*temperature)[r][c] = 0;
            } else if (token[0] == '-') {
                (*Q)[r][c] = -atof(token + 1); // Отрицательный источник
                (*temperature)[r][c] = 0;
            } else {
                (*Q)[r][c] = 0; // Нет источника
                (*temperature)[r][c] = atof(token); // Начальная температура
            }
            token = strtok_s(NULL, " \t\n", &context);
            c++;
        }
        r++;
    }

    fclose(file);
}

// Вспомогательная функция: распределяем строки между процессами «по блокам».
// pcount  = сколько строк достаётся текущему процессу
// pstart  = с какой строки начинаются данные для текущего процесса
// Простейший равномерный подход: делим nrows на размер коммуникатора.
void distribute_rows(int nrows, int rank, int size, int *pstart, int *pcount) {
    int base = nrows / size;       // базовое количество строк на процесс
    int extra = nrows % size;      // остаток, который надо распределить
    // процессы с индексами меньше extra получают на 1 строку больше
    if (rank < extra) {
        *pcount = base + 1;
        *pstart = rank * (base + 1);
    } else {
        *pcount = base;
        *pstart = extra * (base + 1) + (rank - extra) * base;
    }
}

void exchange_borders(double **temp_local, int local_rows, int ncols, int rank, int size) {
    // Обмен верхней границей с процессом выше
    if (rank > 0) {
        // Отправляем верхнюю строку
        MPI_Send(temp_local[0], ncols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        // Получаем верхнюю строку
        MPI_Recv(temp_local[-1], ncols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Обмен нижней границей с процессом ниже
    if (rank < size - 1) {
        // Отправляем нижнюю строку
        MPI_Send(temp_local[local_rows-1], ncols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        // Получаем нижнюю строку
        MPI_Recv(temp_local[local_rows], ncols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Главная функция эволюции температурного поля на локальном участке.
static void evolve_local_field(double **temp_local, double **source_local, int local_rows, int ncols, int rank, int size) {

    // Создадим вспомогательный массив для новых значений
    double **new_temp = (double **) malloc((local_rows + 2) * sizeof(double *));
    for (int i = 0; i < local_rows + 2; i++) {
        new_temp[i] = (double *) malloc(ncols * sizeof(double));
    }

    // Копируем данные из temp_local в new_temp, включая границы.
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < ncols; j++) {
            new_temp[i + 1][j] = temp_local[i][j];
        }
    }

    // Обновляем каждую «внутреннюю» ячейку (учитывая, что граничные строки
    // у процессов будут обмениваться отдельно).
    for (int i = 1; i < local_rows + 1; i++) {
        for (int j = 1; j < ncols - 1; j++) {
            double t_ij = new_temp[i][j];
            double t_up = new_temp[i - 1][j];
            double t_dn = new_temp[i + 1][j];
            double t_lf = new_temp[i][j - 1];
            double t_rt = new_temp[i][j + 1];

            double diff_term = DT * (ALPHA * ((t_up - 2.0 * t_ij + t_dn) / (DELTA_X * DELTA_X)
                                              + (t_lf - 2.0 * t_ij + t_rt) / (DELTA_Y * DELTA_Y)) + source_local[i - 1][j]);

            new_temp[i][j] = t_ij + diff_term;
        }
    }

    // Перенесём новые значения обратно в temp_local
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < ncols; j++) {
            temp_local[i][j] = new_temp[i + 1][j];
        }
    }

    // Освобождаем память
    for (int i = 0; i < local_rows + 2; i++) {
        free(new_temp[i]);
    }
    free(new_temp);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
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

    // Матрицы и их размеры (определяются на root)
    double **global_temp = NULL;
    double **global_source = NULL;
    int nrows = 0, ncols = 0;

    // Читаем только на процессе 0
    if (rank == 0) {
        read_initial_conditions(input_file, &global_temp, &global_source, &nrows, &ncols);
        printf("Размерность входных данных: %d x %d\n", nrows, ncols);
    }

    // Разошлём информацию о размерах nrows, ncols всем процессам
    MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Определим, какие строки достаются текущему процессу
    int pstart, pcount;
    distribute_rows(nrows, rank, size, &pstart, &pcount);

    if (!pcount) {
        printf("Process %d has no work and is exiting.\n", rank);
        size--;
    } else {

        printf("Worker %d takes %d rows (from %d)\n", rank, pcount, pstart + 1);
        fflush(stdout);

        // Каждый процесс выделит память под локальную часть строк:
        double **temp_local = (double **) malloc(pcount * sizeof(double *));
        double **source_local = (double **) malloc(pcount * sizeof(double *));
        for (int i = 0; i < pcount; i++) {
            temp_local[i] = (double *) malloc(ncols * sizeof(double));
            source_local[i] = (double *) malloc(ncols * sizeof(double));
        }

        // Подготавливаем буфер для отправки/приёма строк
        // Сериализуем строки как массив из ncols double
        // Будем пользоваться MPI_Scatterv / MPI_Gatherv или вручную через MPI_Send/MPI_Recv
        // Для простоты здесь – подход с MPI_Scatterv в духе "по строкам".
        // Сперва подготовим массив displs и sendcounts для Scatterv.
        int *sendcounts = (int *) malloc(size * sizeof(int));
        int *displs = (int *) malloc(size * sizeof(int));
        {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int pst, pc;
                distribute_rows(nrows, i, size, &pst, &pc);
                sendcounts[i] = pc * ncols; // столько double на процесс
                displs[i] = offset;
                offset += pc * ncols;
            }
        }

        // Подготовим временные буферы на root
        double *sendbuf_temp = NULL;
        double *sendbuf_source = NULL;
        if (rank == 0) {
            sendbuf_temp = (double *) malloc(nrows * ncols * sizeof(double));
            sendbuf_source = (double *) malloc(nrows * ncols * sizeof(double));
            // Сериализуем global_temp и global_source в эти буферы.
            int idx = 0;
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < ncols; j++) {
                    sendbuf_temp[idx] = global_temp[i][j];
                    sendbuf_source[idx] = global_source[i][j];
                    idx++;
                }
            }
        }

        // Локальные буферы для приёма:
        double *recvbuf_temp = (double *) malloc(pcount * ncols * sizeof(double));
        double *recvbuf_source = (double *) malloc(pcount * ncols * sizeof(double));

        // Рассылаем данные
        MPI_Scatterv(sendbuf_temp, sendcounts, displs, MPI_DOUBLE,
                     recvbuf_temp, pcount * ncols, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(sendbuf_source, sendcounts, displs, MPI_DOUBLE,
                     recvbuf_source, pcount * ncols, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Разложим принятые данные в двумерные локальные массивы
        int idx = 0;
        for (int i = 0; i < pcount; i++) {
            for (int j = 0; j < ncols; j++) {
                temp_local[i][j] = recvbuf_temp[idx];
                source_local[i][j] = recvbuf_source[idx];
                idx++;
            }
        }

        // Память под sendbuf_temp/source можно освободить
        if (rank == 0) {
            free(sendbuf_temp);
            free(sendbuf_source);
        }
        free(recvbuf_temp);
        free(recvbuf_source);

        // ----------------------------------------------------------------------------
        // Основной цикл итераций
        // ----------------------------------------------------------------------------
        for (int iter = 0; iter < NUM_ITER; iter++) {
            // Обмениваем «призрачные» (граничные) строки между соседними процессами
            // Процесс собирается отправлять/принимать строку temp_local[0] (если есть верхний сосед)
            // и temp_local[pcount-1] (если есть нижний сосед).

            // Верхний сосед есть, если rank > 0
            if (rank > 0) {
                // Отправляем свою первую строку rank-1, получаем вниз последнюю строку от rank-1
                MPI_Send(temp_local[0], ncols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(temp_local[-1 + 1], ncols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Аналогично отправляем source_local[0], чтобы при желании учитывать в «соседнем слое» (если нужно).
                // В данном простом примере используем source локально, но при необходимости – аналогично.
            }
            // Нижний сосед есть, если rank < size-1
            if (rank < size - 1) {
                // Отправляем свою последнюю строку
                MPI_Send(temp_local[pcount - 1], ncols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                // Принимаем в temp_local[pcount] (условно, нужно дать «буфер»),
                // но здесь у нас нет прямого pcount индекса. Обычно расширяют массив
                // или дополнительно хранят буфер. Упростим, будем считать, что достаточно
                // для «реальных» расчётов иметь строку pcount-1.
                // В реальном коде нужно расширять на 2 строки (одну сверху, одну снизу).
            }

            // Обновляем поле (итерационная схема)
            evolve_local_field(temp_local, source_local, pcount, ncols);
        }

        // ----------------------------------------------------------------------------
        // Сбор результатов обратно на root
        // ----------------------------------------------------------------------------
        // Сериализуем локальные данные
        double *gatherbuf_temp = (double *) malloc(pcount * ncols * sizeof(double));
        idx = 0;
        for (int i = 0; i < pcount; i++) {
            for (int j = 0; j < ncols; j++) {
                gatherbuf_temp[idx++] = temp_local[i][j];
            }
        }

        // Подготовим для корня общий буфер
        double *final_temp = NULL;
        if (rank == 0) {
            final_temp = (double *) malloc(nrows * ncols * sizeof(double));
        }

        // Собираем
        MPI_Gatherv(gatherbuf_temp, pcount * ncols, MPI_DOUBLE,
                    final_temp, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // На root восстанавливаем двумерный массив и выводим/сохраняем
        if (rank == 0) {
            printf("\nРезультирующее температурное поле после %d итераций:\n", NUM_ITER);
            int pos = 0;
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < ncols; j++) {
                    printf("%3.2f ", final_temp[pos++]);
                }
                printf("\n");
            }
            free(final_temp);

            // Освободим глобальные входные данные
            for (int i = 0; i < nrows; i++) {
                free(global_temp[i]);
                free(global_source[i]);
            }
            free(global_temp);
            free(global_source);
        }

        // Освобождаем ресурсы
        free(sendcounts);
        free(displs);
        free(gatherbuf_temp);

        for (int i = 0; i < pcount; i++) {
            free(temp_local[i]);
            free(source_local[i]);
        }
        free(temp_local);
        free(source_local);
    }

    MPI_Finalize();

    return 0;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
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

    // Матрицы и их размеры (определяются на root)
    double **global_temp = NULL;
    double **global_source = NULL;
    int nrows = 0, ncols = 0;

    // Читаем только на процессе 0
    if (rank == 0) {
        read_initial_conditions(input_file, &global_temp, &global_source, &nrows, &ncols);
        printf("Размерность входных данных: %d x %d\n", nrows, ncols);
    }

    // Разошлём информацию о размерах nrows, ncols всем процессам
    MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Определим, какие строки достаются текущему процессу
    int pstart, pcount;
    distribute_rows(nrows, rank, size, &pstart, &pcount);

    if (!pcount) {
        printf("Process %d has no work and is exiting.\n", rank);
        size--;
    } else {

        printf("Worker %d takes %d rows (from %d)\n", rank, pcount, pstart + 1);
        fflush(stdout);


        // Каждый процесс выделит память под локальную часть строк:
        double **temp_local = (double **) malloc(pcount * sizeof(double *));
        double **source_local = (double **) malloc(pcount * sizeof(double *));
        for (int i = 0; i < pcount; i++) {
            temp_local[i] = (double *) malloc(ncols * sizeof(double));
            source_local[i] = (double *) malloc(ncols * sizeof(double));
        }

        // Подготавливаем буфер для отправки/приёма строк
        // Сериализуем строки как массив из ncols double
        // Будем пользоваться MPI_Scatterv / MPI_Gatherv или вручную через MPI_Send/MPI_Recv
        // Для простоты здесь – подход с MPI_Scatterv в духе "по строкам".
        // Сперва подготовим массив displs и sendcounts для Scatterv.
        int *sendcounts = (int *) malloc(size * sizeof(int));
        int *displs = (int *) malloc(size * sizeof(int));
        {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int pst, pc;
                distribute_rows(nrows, i, size, &pst, &pc);
                sendcounts[i] = pc * ncols; // столько double на процесс
                displs[i] = offset;
                offset += pc * ncols;
            }
        }

        // Подготовим временные буферы на root
        double *sendbuf_temp = NULL;
        double *sendbuf_source = NULL;
        if (rank == 0) {
            sendbuf_temp = (double *) malloc(nrows * ncols * sizeof(double));
            sendbuf_source = (double *) malloc(nrows * ncols * sizeof(double));
            // Сериализуем global_temp и global_source в эти буферы.
            int idx = 0;
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < ncols; j++) {
                    sendbuf_temp[idx] = global_temp[i][j];
                    sendbuf_source[idx] = global_source[i][j];
                    idx++;
                }
            }
        }

        // Локальные буферы для приёма:
        double *recvbuf_temp = (double *) malloc(pcount * ncols * sizeof(double));
        double *recvbuf_source = (double *) malloc(pcount * ncols * sizeof(double));

        // Рассылаем данные
        MPI_Scatterv(sendbuf_temp, sendcounts, displs, MPI_DOUBLE,
                     recvbuf_temp, pcount * ncols, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(sendbuf_source, sendcounts, displs, MPI_DOUBLE,
                     recvbuf_source, pcount * ncols, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Разложим принятые данные в двумерные локальные массивы
        int idx = 0;
        for (int i = 0; i < pcount; i++) {
            for (int j = 0; j < ncols; j++) {
                temp_local[i][j] = recvbuf_temp[idx];
                source_local[i][j] = recvbuf_source[idx];
                idx++;
            }
        }

        // Память под sendbuf_temp/source можно освободить
        if (rank == 0) {
            free(sendbuf_temp);
            free(sendbuf_source);
        }
        free(recvbuf_temp);
        free(recvbuf_source);

        // Основной цикл расчета
        for (int iter = 0; iter < NUM_ITER; iter++) {
            exchange_borders(temp_local, local_rows, ncols, rank, size);
            evolve_local_field(temp_local, source_local, local_rows, ncols, rank, size);
        }

        // ----------------------------------------------------------------------------
        // Сбор результатов обратно на root
        // ----------------------------------------------------------------------------
        // Сериализуем локальные данные
        double *gatherbuf_temp = (double *) malloc(pcount * ncols * sizeof(double));
        idx = 0;
        for (int i = 0; i < pcount; i++) {
            for (int j = 0; j < ncols; j++) {
                gatherbuf_temp[idx++] = temp_local[i][j];
            }
        }

        // Подготовим для корня общий буфер
        double *final_temp = NULL;
        if (rank == 0) {
            final_temp = (double *) malloc(nrows * ncols * sizeof(double));
        }

        // Собираем
        MPI_Gatherv(gatherbuf_temp, pcount * ncols, MPI_DOUBLE,
                    final_temp, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // На root восстанавливаем двумерный массив и выводим/сохраняем
        if (rank == 0) {
            printf("\nРезультирующее температурное поле после %d итераций:\n", NUM_ITER);
            int pos = 0;
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < ncols; j++) {
                    printf("%3.2f ", final_temp[pos++]);
                }
                printf("\n");
            }
            free(final_temp);

            // Освободим глобальные входные данные
            for (int i = 0; i < nrows; i++) {
                free(global_temp[i]);
                free(global_source[i]);
            }
            free(global_temp);
            free(global_source);
        }

        // Освобождаем ресурсы
        free(sendcounts);
        free(displs);
        free(gatherbuf_temp);

        for (int i = 0; i < pcount; i++) {
            free(temp_local[i]);
            free(source_local[i]);
        }
        free(temp_local);
        free(source_local);
    }

    MPI_Finalize();
    return 0;
}
