#pragma once

#ifndef TOOLS_PORTABLE
#define TOOLS_PORTABLE

#ifdef _WIN32
	#include <Windows.h>
	#define LOCK_MUTEX CRITICAL_SECTION
#else
	#include <pthread.h>
	#define LOCK_MUTEX pthread_mutex_t
#endif

double time_gettime(void);
double time_duration(double begin);

LOCK_MUTEX* lock_create_multi(int no_lock);
void lock_destroy_multi(LOCK_MUTEX *lock, int no_lock);
void lock_acquire(LOCK_MUTEX *lock);
int lock_try_acquire(LOCK_MUTEX *lock);
void lock_release(LOCK_MUTEX *lock);

#endif
