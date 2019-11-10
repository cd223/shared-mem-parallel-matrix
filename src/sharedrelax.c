#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/*
    Barrier to stop a thread continuing execution until all other threads have
    'hit' the barrier. ("Superstep Programming"). Needed in order to prevent a
    thread carrying onto the next iteration before others have finished with
    the relaxation process on the subsection of the matrix they work on.
    
    Barriers were chosen as synchronisation was needed across threads and 
    threads likely to not be running in lockstep.
*/
pthread_barrier_t posix_barrier;

//  Flag indicating if precision reached on all threads. Main thread sets this.
volatile bool all_within_precision = false;

/* 
    Array (size = number of threads) for each thread to indicate if more 
    iterations needed to reach precision. Each pthread sets its own element.
 */
bool *next_iteration_needed;

/*
    Structs for optional program arguments and info each thread needs to
    relax the subpart of the matrix it has been assigned.
*/
struct prog_opts_t
{
    int mat_dimension;
    int num_threads;
    double rel_precision;
    bool verbose_mode;
    bool info_mode;
} opts;

typedef struct thread_info
{
    int id;                 // Friendly identifier
    int target_row;         // Each thread starts at different co-ordinate
    int target_col;
    double** mat_in;        // Each thread keeps reference to whole matrices
    double** mat_out;
    int num_cells_to_relax; // Kept even as possible between threads 
} thread_info_t;

/* 
    Check entered thread count does not exceed the number of inner cells in
    the matrix, i.e. (d-2)^2 where d is the matrix dimension.
*/
void validate_thread_count(void) 
{
    int inner_cell_count = (opts.mat_dimension - 2) * (opts.mat_dimension - 2);
    if (opts.num_threads > inner_cell_count) {
    	printf("WARN: Thread count entered (%d) exceeds number of inner cells"
                " (%d). Resetting thread count to 1.\n",
                opts.num_threads, inner_cell_count);
    	opts.num_threads = 1; // Reset to default value of 1 if so.
    }
}

/* Parse values of optional command line arguments. */
void parse_opts(int argc, char **argv)
{
    static const char *options = "d:t:p:vi";
    int opt = getopt(argc, argv, options);
    while(opt != -1) {
        switch( opt ) {
            case 'd': 
                opts.mat_dimension = atoi(optarg);  // -d [int]
                break;
            case 't':
                opts.num_threads = atoi(optarg);    // -t [int]
                break;
            case 'p':
                opts.rel_precision = atof(optarg);  // -p [double]
                break;
            case 'v':
                opts.verbose_mode = true;           // -v
                break;
            case 'i':
                opts.info_mode = true;              // -i
                break;
            default:
                break;
        }
        opt = getopt(argc, argv, options);
    }
    validate_thread_count(); // Check value of -t is valid
}

/* Join pthreads with main thread, checking for errors. */
int join_threads(pthread_t *threads, int t_count) 
{
    for (int i = 0; i < t_count; ++i) {
        if (pthread_join(threads[i], NULL))
        {
            printf("ERROR: Joining of pthread %d failed.", i);
            return 1;
        }
	}
    return 0;
}

/* Synchroniza pthreads, checking for errors. */
int sync_threads() 
{
    int ret_val = 0;
    ret_val = pthread_barrier_wait(&posix_barrier); // Sync threads

    if (ret_val != PTHREAD_BARRIER_SERIAL_THREAD && ret_val != 0) {
        printf("ERROR: Error synchronising threads");
        return 1; // error with barrier wait
    }
    return 0;
}

/*
    Initialise contents of the given matrices to be identical.
    Fixed value of 1.0 at edges, random values in inner cells.
*/
void initialise_matrices(double **mat_in, double **mat_out) 
{
    int mat_size = opts.mat_dimension;
    // srand((unsigned)time(NULL)); // Can be (un)commented to set a random seed

    for (int row=0; row<mat_size; ++row) {
        for (int col=0; col<mat_size; ++col) {
            if (row == 0 || col == 0 || row == (mat_size-1) || 
                col == (mat_size-1))
            {
				mat_in[row][col] = (double)1;
				mat_out[row][col] = (double)1;
			} else {
                double rand_val = ((double) rand()/(double) RAND_MAX);
				mat_in[row][col] = rand_val;
				mat_out[row][col] = rand_val;
			}
        }
    }
}

/* Display values of arguments to be used in the program. */
void display_opts()
{
    printf("\nINFO: Arguments to be used in program:\n"
            "   - Matrix Dimensions (-d):       %dx%d\n"
            "   - Thread Count (-t):            %d\n"
            "   - Relaxation Precision (-p):    %f\n"
            "   - Verbose Mode (-v):            %s\n"
            "   - Info Mode (-i):               %s\n",
            opts.mat_dimension, opts.mat_dimension, opts.num_threads,
            opts.rel_precision, opts.verbose_mode ? "YES" : "NO",
            opts.info_mode ? "YES" : "NO");
}

/* Display details about matrix/problem size and breakdown across threads. */
void display_problem_breakdown(int inner_cell_count,
                               int cells_per_thread,
                               int cells_left,
                               int t_count,
                               thread_info_t *threads_info)
{
    printf("\nINFO: Problem size for n x n matrix, using t threads "
                    "where n=%d, t=%d:\n"
            "   - Inner cell count, I = (n-2)^2    =   %d\n"
            "   - Cells per thread, c = I / t      =   %d\n"
            "   - Cells left over,  R = I `mod` t  =   %d\n\n",
            opts.mat_dimension, t_count, inner_cell_count, 
            cells_per_thread, cells_left);

    for(int t = 0; t < t_count; ++t) {
        printf("\nINFO: Thread %d given %d cells to relax, "
               "starting at (%d, %d).\n",
            threads_info[t].id, threads_info[t].num_cells_to_relax,
            threads_info[t].target_row, threads_info[t].target_col);
    }
}

/* Display matrix produced at each step by relaxation process. */
void display_matrix(double **matrix, int iteration_num) {
    int mat_size = opts.mat_dimension;

    // Positive non-zero iteration_num means matrix relaxed
    if(iteration_num == -1) {
        printf("\nINFO: ___Initial Matrix_______\n");
    } else if(iteration_num > 0) {
        printf("\nINFO: ___Relaxed Matrix (%d iterations)_______\n",
            iteration_num);
    } else {
        printf("\nINFO: ___Iteration Complete_______\n");
    }
    for (int row = 0; row < mat_size; ++row) {
        for (int col = 0; col < mat_size; ++col) {
            printf("%f ", matrix[row][col]);        
        }
        printf("\n");
    }
    printf("\n");
}

/* Extract program variables from command line and set default values. */
void extract_program_opts(int argc, char **argv)
{
    // Set optional arguments to default values in case of invalid input
    opts.mat_dimension = 50;
    opts.num_threads = 1;
    opts.rel_precision = 0.1;
    opts.verbose_mode = 0;
    opts.info_mode = 0;
    parse_opts(argc, argv);
    
    if(opts.verbose_mode) {
        display_opts();
    }
}

/* Compute each "target" cell location - where each thread starts relaxation */
void configure_target_cells(thread_info_t *threads_info, int inner_matrix_size,
                            int t_count)
{
    int step_size = 0, row_counter = 0, col_counter = 0, cell_counter = 0;

	for (int i=0; i < t_count; i++) {
        step_size = threads_info[i].num_cells_to_relax;

        // Set target cell position for this thread
		threads_info[i].target_row = row_counter + 1; 
		threads_info[i].target_col = col_counter + 1;

        // Calculate last cell before next thread's target position
		cell_counter = col_counter + step_size;
        row_counter += cell_counter / inner_matrix_size;
        col_counter = cell_counter % inner_matrix_size;
	}
}

/*
    Each thread is allocated a sub-part of the main matrix to relax; the size
    of which is dependent on the total problem size (total cells to relax).
*/
void divide_matrix_between_threads(thread_info_t *threads_info,
                                   double **mat_in,
                                   double **mat_out, 
                                   int t_count)
{
    // Calculate total cells to relax if dividing equally amongst threads 
    int inner_matrix_size = opts.mat_dimension - 2;
    int inner_cell_count = (inner_matrix_size) * (inner_matrix_size);
    int cells_per_thread = inner_cell_count / t_count;

    // Non-zero remainder if total cells don't divide by number of threads
    int cells_left = inner_cell_count % t_count;
    int thread_id = 0;

    // Assign number of cells to relax equally before distributing remainder
    for(int t = 0; t < t_count; ++t) {
        threads_info[t].id = t;
        threads_info[t].mat_in = mat_in; // Each thread has refs to matrices
        threads_info[t].mat_out = mat_out;
        threads_info[t].num_cells_to_relax = cells_per_thread;
    }

    // Divide remaining cells (if any) equally as possible
    if(cells_left) { 
        while(cells_left > 0) {
            threads_info[thread_id].num_cells_to_relax++;
            thread_id = (thread_id == t_count) ? 0 : thread_id + 1;
 			cells_left--;
        }
    }

    // Compute where in matrix each thread will start to relax
    configure_target_cells(threads_info, inner_matrix_size, t_count);

    if (opts.verbose_mode) {
        display_problem_breakdown(inner_cell_count, cells_per_thread,
                                  (inner_cell_count % t_count), t_count,
                                  threads_info);
    }
}

/* 
    Swaps the references to two matrices in order to operate on the most recent
    matrix. This is because mat_out was written to in the last iteration when
    mat_in was operated on, so references are swapped to ensure the previous
    output matrix forms the next iteration's input matrix.
*/
void swap_matrix_refs(double ***mat_in, double ***mat_out)
{
    double **temp = *mat_in; // Store temporary ref to mat_in before swap
    *mat_in = *mat_out;
    *mat_out = temp;
}

/* 
    Returns value indicating whether corresponding cells in mat_in and mat_out
    fall within precision range.
        | 0 = Cells not within precision, 1 = Cells within precision 
*/
bool is_within_precision(double **mat_in, double **mat_out, int row, int col)
{
    double difference = fabs(mat_in[row][col]-mat_out[row][col]);
	return (difference < opts.rel_precision);
}

/*
    Check precision reached across all threads.
        | 1 = all elements set to false (precision reached across all threads)
        | 0 = one or more elements set to true (more iterations needed) 
*/
int is_global_precision_reached(int t_count) 
{
    for (int t = 0; t < t_count; t++) {
		if (next_iteration_needed[t]) {
			return 0;
		}
	}
	return 1;
}

/*  Cell relaxation - calculates average of a cell's 4 neighbours. */
double cell_relax(double **matrix, int row, int col)
{
	double total = matrix[row][col-1]     // Cell below
                 + matrix[row][col+1]     // Cell above 
                 + matrix[row-1][col]     // Cell to left
                 + matrix[row+1][col];    // Cell to right 
    return total / (double) 4;
}

/* 
    Submatrix relaxation - relaxes part of the matrix a thread is assigned.
    Uses t_info_ptr to determine starting cell, how many cells to relax, etc.
*/
void submatrix_relax(thread_info_t *t_info)
{
    int cells_to_relax = t_info->num_cells_to_relax;
    int starting_row = t_info->target_row;
    int starting_col = t_info->target_col;
    int mat_size = opts.mat_dimension;

    if(opts.verbose_mode) {
        printf("\nINFO: Thread %d starting execution at (%d,%d)...\n",
                t_info->id, t_info->target_row, t_info->target_col);
    }

    bool within_precision = true; // Boolean check if cells within precision
    int row = 0, col = 0;

	while(true)
    {
        within_precision = true; // Reset to default = true
        row = starting_row;
        col = starting_col;

        // Relax number of cells given to this thread
        for (int n=0; n < cells_to_relax; n++)
        {
            // Relax current cell and check if within desired precision
            t_info->mat_out[row][col] = cell_relax(t_info->mat_in, row, col);

            within_precision = (within_precision && 
                                is_within_precision(t_info->mat_in,
                                                    t_info->mat_out,
                                                    row, col));
			
            // Wrap around to next row if current row finished
			col++;
            if (col == mat_size-1) {
				col = 1;
                row++;
			}
		}

        /* If precision not reached on this thread, another iteration is needed
           across all threads so update array accordingly */
        next_iteration_needed[t_info->id] = (!within_precision) ? true : false;

        // Wait until other threads have finished their iterations
        if (sync_threads()) {
            return; // error syncing threads 
        }
        if (sync_threads()) {
            return; // error syncing threads 
        }

        /* Check if all threads reached precision (this flag is updated on main
           thread). If so, this thread can exit */
        if (all_within_precision) {
            pthread_exit(NULL);
        }

        // Otherwise, another iteration needed - Swap matrix refs and loop
        if ((opts.verbose_mode || opts.info_mode) && t_info->id==0) {
            // Only need 1 thread to display - all hold refs to same matrices
            display_matrix(t_info->mat_out, 0);
        }
        swap_matrix_refs(&t_info->mat_in, &t_info->mat_out);
	}
}

/* Create and start pthreads, checking for errors. */
int create_threads(pthread_t *threads, thread_info_t *threads_info, 
                    int t_count) 
{
    for (int i=0; i<t_count; ++i)
    {
        if (pthread_create(&threads[i], NULL, (void*(*)(void*))submatrix_relax,
                           (void*)&threads_info[i]))
        {
            printf("ERROR: Creation of pthread %d failed.", i);
            return 1;
        }
    }
    return 0;
}

/* 
    Matrix relaxation - main loop to relax the matrix. Returns either 0 or 1.
        | 0 - Matrix relaxed without error, 1 - Error occurred
*/
int matrix_relax(pthread_t *threads, thread_info_t *threads_info, int t_count)
{
    // Barrier initialised for t+1 threads = main thread plus created pthreads
    if (pthread_barrier_init(&posix_barrier, NULL, t_count+1)) {
        printf("ERROR: Initialisation of pthread_barrier failed.");
        return 1; // error with barrier init
    }

    // Create pthreads and pass info required to relax subpart of matrix
    if (create_threads(threads, threads_info, t_count)) {
        return 1; // error creating threads 
    }

    int iteration_num = 0;
    
    // Main loop - repeat whilst precision not reached across all threads
    while (!all_within_precision) 
    {
        if (sync_threads()) {
            return 1; // error syncing threads 
        }

        ++iteration_num;

        if (is_global_precision_reached(t_count)) { 
            all_within_precision = true; // No more iterations needed, exit
        }

        if (sync_threads()) {
            return 1; // error syncing threads 
        }
    }

    if (join_threads(threads, t_count)) {
        return 1;  // error joining threads
    }

    printf("INFO: Matrix RELAXED after %d iterations...\n", iteration_num);
    if (opts.verbose_mode || opts.info_mode)
    {
        display_matrix(threads_info[0].mat_out, iteration_num);
    }

    // Destroy barrier when relaxed
    if (pthread_barrier_destroy(&posix_barrier)) {
        printf("ERROR: Destroying of barrier failed.");
        return 1; // error with barrier destroy
    }
    return 0;
}

/* 
    Relaxation solver can expect in argv:
        -d      = [int] dimension of matrix
        -t      = [int] number of threads
        -p      = [double] precision to work to
        -v / -i = verbose / info mode enabled 
*/
int main(int argc, char **argv)
{
    extract_program_opts(argc, argv);
    int mat_size = opts.mat_dimension;
    int t_count = opts.num_threads;

    // Allocate memory for matrices and threads
    size_t matrix_size = mat_size * sizeof(double*);
    size_t buffer_size = mat_size * mat_size * sizeof(double);
    size_t thread_info_size = t_count * sizeof(thread_info_t);
    size_t threads_size = t_count * sizeof(pthread_t);
    size_t iteration_check_array_size = t_count * sizeof(bool);
    double **mat_in = malloc(matrix_size);
	double **mat_out = malloc(matrix_size);
    double *in_buffer = malloc(buffer_size);
    double *out_buffer = malloc(buffer_size);
    thread_info_t *threads_info = malloc(thread_info_size);
    pthread_t *threads = malloc(threads_size);
    next_iteration_needed = malloc(iteration_check_array_size);

    if (mat_in == NULL || mat_out == NULL || 
        in_buffer == NULL || out_buffer == NULL) {
		printf("ERROR: Allocation of memory to matrices failed.");
        return 1;
    } else if (threads_info==NULL || threads==NULL) {
        printf("ERROR: Allocation of memory for threads failed.");
        return 1;
    }
    for (int i=0; i < mat_size; ++i) {
        mat_in[i] = in_buffer + mat_size*i;
        mat_out[i] = out_buffer + mat_size*i;
    }

    // Initialise matrices with identical content
    initialise_matrices(mat_in, mat_out);

    // Divide problem between threads so each works on different part of matrix 
    divide_matrix_between_threads(threads_info, mat_in, mat_out, t_count);
    
    // Display initial matrices
    if (opts.verbose_mode || opts.info_mode) {
        display_matrix(mat_in, -1);
    }

    // Relax matrix, checking for errors
    if(matrix_relax(threads, threads_info, t_count)) {
        printf("ERROR: Matrix relaxation failed.");
        return 1;
    }

    // Free memory allocated to matrices and threads
    free(mat_in);
    free(mat_out);
    free(in_buffer);
    free(out_buffer);
    free(threads_info);
    free(threads);
    free(next_iteration_needed);
    return 0;
}