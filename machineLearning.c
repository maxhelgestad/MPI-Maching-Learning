/* assert */
#include <assert.h>

/* fabs */
#include <math.h>

/* MPI API */
#include <mpi.h>

/* printf, fopen, fclose, fscanf, scanf */
#include <stdio.h>

/* EXIT_SUCCESS, malloc, calloc, free, qsort */
#include <stdlib.h>

#define MPI_SIZE_T MPI_UNSIGNED_LONG

struct distance_metric {
  size_t viewer_id;
  double distance;
};

static int
cmp(void const *ap, void const *bp)
{
  struct distance_metric const a = *(struct distance_metric*)ap;
  struct distance_metric const b = *(struct distance_metric*)bp;

  return a.distance < b.distance ? -1 : 1;
}

int
main(int argc, char * argv[])
{
  int ret, p, rank;
  size_t n, m, k;
  double * rating;

  /* Initialize MPI environment. */
  ret = MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS == ret);

  /* Get size of world communicator. */
  ret = MPI_Comm_size(MPI_COMM_WORLD, &p);
  assert(ret == MPI_SUCCESS);

  /* Get my rank. */
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(ret == MPI_SUCCESS);

  /* Validate command line arguments. */
  assert(2 == argc);

  /* Read input --- only if your rank 0. */
  if (0 == rank) {
    /* ... */
    char const * const fn = argv[1];

    /* Validate input. */
    assert(fn);

    /* Open file. */
    FILE * const fp = fopen(fn, "r");
    assert(fp);

    /* Read file. */
    fscanf(fp, "%zu %zu", &n, &m);

    /* Allocate memory. */
    rating = malloc(n * m * sizeof(*rating));

    /* Check for success. */
    assert(rating);

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        fscanf(fp, "%lf", &rating[i * m + j]);
      }
    }

    /* Close file. */
    ret = fclose(fp);
    assert(!ret);
  }

  /* Send number of viewers and movies to rest of processes. */
  if (0 == rank) {
    for (int r = 1; r < p; r++) {
      ret = MPI_Send(&n, 1, MPI_SIZE_T, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
      ret = MPI_Send(&m, 1, MPI_SIZE_T, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
      ret = MPI_Recv(&n, 1, MPI_SIZE_T, 0, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
      assert(MPI_SUCCESS == ret);
      ret = MPI_Recv(&m, 1, MPI_SIZE_T, 0, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
      assert(MPI_SUCCESS == ret);
  }

  /* Compute base number of viewers. */
  size_t const base = 1 + ((n - 1) / p); // ceil(n / p)

  /* Compute local number of viewers. */
  size_t const ln = (rank + 1) * base > n ? n - rank * base : base;

  /* Send viewer data to rest of processes. */
  if (0 == rank) {
    for (int r = 1; r < p; r++) {
      size_t const rn = (r + 1) * base > n ? n - r * base : base;
      ret = MPI_Send(rating + r * base * m, rn * m, MPI_DOUBLE, r, 0,
        MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
    /* Allocate memory. */
    rating = malloc(ln * m * sizeof(*rating));

    /* Check for success. */
    assert(rating);

    ret = MPI_Recv(rating, ln * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ret);
  }

  /* Allocate more memory. */
  double * const urating = malloc((m - 1) * sizeof(*urating));

  /* Check for success. */
  assert(urating);

  /* Get user input and send it to rest of processes. */
  if (0 == rank) {
    for (size_t j = 0; j < m - 1; j++) {
      printf("Enter your rating for movie %zu: ", j + 1);
      fflush(stdout);
      scanf("%lf", &urating[j]);
    }

    for (int r = 1; r < p; r++) {
      ret = MPI_Send(urating, m - 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
    ret = MPI_Recv(urating, m - 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ret);
  }

  /* Allocate more memory. */
  struct distance_metric * const distance = calloc(ln * m, sizeof(*distance));

  /* Check for success. */
  assert(distance);

  /* Compute distances. */
  for (size_t i = 0; i < ln; i++) {
    distance[i].viewer_id = i;
    for (size_t j = 0; j < m - 1; j++) {
      distance[i].distance += fabs(urating[j] - rating[i * m + j]);
    }
  }

	/* Send to rank 0 */
  ret = MPI_Send(distance, ln * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);

	/* rank 0 recieves */
	if (rank == 0) {
		struct distance_metric * const distances = malloc(n * m * sizeof(*distances));
		for (int r = 0; r < p; r++) {			
			ret = MPI_Recv(distances, ln * m, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		assert(MPI_SUCCESS == ret);
		
  	/* Sort distances. */
  	qsort(distances, n, sizeof(*distances), cmp);

  	/* Get user input. */
  	printf("Enter the number of similar viewers to report: ");
  	scanf("%zu", &k);

  	/* Output k viewers who are least different from the user. */
  	printf("Viewer ID   Movie five   Distance\n");
  	printf("---------------------------------\n");

  	for (size_t i = 0; i < k; i++) {
    	printf("%9zu   %10.1lf   %8.1lf\n", distances[i].viewer_id + 1,
      	rating[distances[i].viewer_id * m + 4], distances[i].distance);
  	}

  	printf("---------------------------------\n");

  	/* Compute the average to make the prediction. */
  	double sum = 0.0;
  	for (size_t i = 0; i < k; i++) {
    	sum += rating[distances[i].viewer_id * m + 4];
  	}

  	/* Output prediction. */
  	printf("The predicted rating for movie five is %.1lf.\n", sum / k);

  	free(distances);
	}

	free(rating);
	free(urating);
	free(distance);

	ret = MPI_Finalize();
	assert(MPI_SUCCESS == ret);

	return EXIT_SUCCESS;
}
