cmake_minimum_required(VERSION 3.26)
project(task1 C)

set(CMAKE_C_STANDARD 11)

# Поиск библиотеки MPI
find_package(MPI REQUIRED)

# Добавление пути до библиотеки MPI
include_directories("C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include")
include_directories("C:\\msys64\\build\\include")

add_executable(task1
#        main.c
        lab1.c
)

# Линковка с библиотекой MPI
target_link_libraries(task1 "C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64\\msmpi.lib")
target_link_libraries(task1 "C:\\msys64\\build\\lib\\libgmp.dll.a")