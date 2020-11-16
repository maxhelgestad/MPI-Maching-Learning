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

  MPI_Bcast(&n, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);

  /* Compute base number of viewers. */
  size_t const base = 1 + ((n - 1) / p); // ceil(n / p)

  /* Compute local number of viewers. */
  size_t const ln = (rank + 1) * base > n ? n - rank * base : base;

  /* Send viewer data to rest of processes. */
  if (0 != rank) {
    /* Allocate memory. */
    rating = malloc(ln * m * sizeof(*rating));
    assert(rating);
  }
  
  int * const sendcount = malloc(p * sizeof(*sendcount));
  int * const displs = malloc(p * sizeof(*displs));
  
  for (size_t r = 0; r < p; r++) {
  	size_t const rn = (r + 1) * base > n ? n - r * base : base;
  	sendcount[r] = rn * m;
  	displs[r] = r * base * m;
  }

  ret = MPI_Scatterv(rating, sendcount, displs, MPI_DOUBLE, rating, ln * m, 	MPI_DOUBLE, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);

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
	}
	
	/* Send to all processes */
	MPI_Bcast(urating, m-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Allocate more memory. */
  double * distance;
  
  if (rank == 0) {
  	distance = calloc(n, sizeof(*distance));
  } else {
  	distance = calloc(ln, sizeof(*distance));
  }

  /* Check for success. */
  assert(distance);

  /* Compute distances. */
  for (size_t i = 0; i < ln; i++) {
    for (size_t j = 0; j < m - 1; j++) {
      distance[i] += fabs(urating[j] - rating[i * m + j]);
    }
  }
#if 0
  if (rank != 0) {
  	ret = MPI_Send(distance, ln, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  	assert(MPI_SUCCESS == ret);
  } else {
  		for (int r = 1; r < p; r++) {	
			size_t const rn = (r + 1) * base > n ? n - r * base : base;
			ret = MPI_Recv(distance + r * base, rn, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			assert(MPI_SUCCESS == ret);
		}
  }
#endif
  
//#if 0
  int * const recvcount = malloc(p * sizeof(*recvcount));
  int * const displs2 = malloc(p * sizeof(*displs2));
  for (size_t r = 0; r < p; r++) {
  	size_t const rn = (r + 1) * base > n ? n - r * base : base;
  	recvcount[r] = rn;
  	displs2[r] = r * base;
  }
  
	ret = MPI_Gatherv(distance, ln, MPI_DOUBLE, distance, recvcount, displs2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	assert(MPI_SUCCESS == ret);
//#endif

	/* rank 0 recieves */
	if (rank == 0) {
		struct distance_metric * const distances = malloc(n * m * sizeof(*distances));
		for (int i = 0; i < n * m; i++) {
			distances[i].viewer_id = i;
			distances[i].distance = distance[i];
		}
  	/* Sort distances. */
  	qsort(distances, n, sizeof(*distances), cmp);

  	/* Get user input. */
  	printf("Enter the number of similar viewers to report: ");
    fflush(stdout);
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
	} else {

	}

	free(rating);
	free(urating);
	free(distance);
	free(sendcount);
	free(displs);
	//free(recvcount);
	//free(displs2);

	ret = MPI_Finalize();
	assert(MPI_SUCCESS == ret);

	return EXIT_SUCCESS;
}
