#ifndef	PROFILING_H_
#define	PROFILING_H_

	#include <stdlib.h>

	#ifdef __APPLE__
		#include <sys/time.h>
	#elif	_WIN32
		#include <Windows.h>
	#else
	#endif

// definitions
typedef struct timing_str
{
	#ifdef __APPLE__
		uint64_t processingTime;
		uint64_t communicationTime;
		uint64_t initializationTime;
		uint64_t cleanupTime;
	#elif _WIN32
		DWORDLONG processingTime;
		DWORDLONG communicationTime;
		DWORDLONG initializationTime;
		DWORDLONG cleanupTime;
	#else
	#endif
} timing;

// variables
	#ifdef __APPLE__
		#define profileDecls struct timeval startTime, endTime;
	#elif	_WIN32
		#define profileDecls LARGE_INTEGER startCount; \
								LARGE_INTEGER endCount; \
								LARGE_INTEGER counterConversion; \
								LONGLONG totalTime; \
								LONGLONG * elapsedTimes;
	#else
	#endif


	#ifdef __APPLE__
		#define	startTimer()	(gettimeofday( &startTime, NULL ))
		#define	stopTimer()	(gettimeofday( &endTime, NULL ))
		#define	calculateTime()	( (1000*1000) * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) )
	#elif _WIN32
		#define	startTimer()	(QueryPerformanceCounter(&startCount))
		#define	stopTimer()		QueryPerformanceCounter(&endCount); QueryPerformanceFrequency( &counterConversion );
		#define calculateTime()	(1000*1000*((double)(endCount.QuadPart - startCount.QuadPart))/(double)counterConversion.QuadPart)
	#else
	#endif

#endif
