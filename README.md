[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/EH4Oowbe)
# Concurrency and Parallelism 23/24 - Project

## Structure

- **cmake**: helper files for the project's CMake configuration.
- **dataset**: set of images for you to test. Your performance analysis should include more images
than the ones included here.
- **include**: project header files.
- **libwb**: header and source files for the WB library that oers some utility functions to handle
images.
- **report**: template for the project's report. You must place your final report in this folder.
- **src**: project source files.
- **test**: project test files that make use of the Google Test framework.

## Installation requirements

C++ compiler and a profiler.

cmake is advised

## Compilation and Execution
To compile import in a IDE that supports cmake or run the following sequence of command in a terminal:


```
mkdir build
cd build
cmake ..
make
```

To run the executable in the IDE, simply 
select the target from the target list and edit the 
configuration to provide the arguments.

From the command line, type:

```
./project input-image.ppm n_iterations output_image.ppm
```

To run the tests in the IDE,  simply
select the target from the target list.

From the command line, type:

```
test/histogram_eq_test
```

or 
```
test/template_test
```