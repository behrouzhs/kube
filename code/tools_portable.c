#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "tools_portable.h"

#ifdef _WIN32
	double time_gettime()
	{
		return ((double)((double)clock() / (double)CLOCKS_PER_SEC));
	}

	double time_duration(double begin)
	{
		return (time_gettime() - begin);
	}

	LOCK_MUTEX* lock_create_multi(int no_lock)
	{
		int i;
		LOCK_MUTEX *lock = (LOCK_MUTEX*)malloc(sizeof(LOCK_MUTEX) * no_lock);
		if (lock == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
		for (i = 0; i < no_lock; i++)
			InitializeCriticalSection(&lock[i]);
		return lock;
	}

	void lock_destroy_multi(LOCK_MUTEX *lock, int no_lock)
	{
		int i;
		for (i = 0; i < no_lock; i++)
			DeleteCriticalSection(&lock[i]);
		free(lock); lock = NULL;
	}

	void lock_acquire(LOCK_MUTEX *lock)
	{
		EnterCriticalSection(lock);
	}

	int lock_try_acquire(LOCK_MUTEX *lock)
	{
		return (TryEnterCriticalSection(lock) ? 0 : 1);
	}

	void lock_release(LOCK_MUTEX *lock)
	{
		LeaveCriticalSection(lock);
	}
#else
	double time_gettime()
	{
		struct timespec now;
		clock_gettime(CLOCK_REALTIME, &now);
		return ((double)((double)now.tv_sec + ((double)now.tv_nsec * 1.0e-9)));
	}

	double time_duration(double begin)
	{
		struct timespec now;
		clock_gettime(CLOCK_REALTIME, &now);
		double d_now = (double)((double)now.tv_sec + ((double)now.tv_nsec * 1.0e-9));
		return (d_now - begin);
	}

	LOCK_MUTEX* lock_create_multi(int no_lock)
	{
		int i;
		LOCK_MUTEX *lock = (LOCK_MUTEX*)malloc(sizeof(LOCK_MUTEX) * no_lock);
		if (lock == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
		for (i = 0; i < no_lock; i++)
			pthread_mutex_init(&lock[i], NULL);
		return lock;
	}

	void lock_destroy_multi(LOCK_MUTEX *lock, int no_lock)
	{
		int i;
		for (i = 0; i < no_lock; i++)
			pthread_mutex_destroy(&lock[i]);
		free(lock); lock = NULL;
	}

	void lock_acquire(LOCK_MUTEX *lock)
	{
		pthread_mutex_lock(lock);
	}

	int lock_try_acquire(LOCK_MUTEX *lock)
	{
		return pthread_mutex_trylock(lock);
	}

	void lock_release(LOCK_MUTEX *lock)
	{
		pthread_mutex_unlock(lock);
	}
#endif
