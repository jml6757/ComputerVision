
// guard
#ifndef GPU_CACHE_H_
#define GPU_CHACE_H_

// inclusions
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <CL/cl.h>

// definitions
#define	DEFAULT_CACHE_SIZE	10

class GPUCache
{
private:
	// variables
	// start with the naive approach
	cl_mem * cacheDataTable;
	int * cacheIndexTable;
	int * cacheOwnerTable;
	uint32_t cacheSize;
	uint32_t currentCacheIndex;
	// space for A matrix
	uint32_t lastStartJ;
	uint32_t lastEndJ;
	cl_mem singleSavedData;
	// space for G
	cl_mem g_data;
	int g_data_allocated;
	// space for y (that's labels, folks)
	cl_mem y_data;
	int y_data_allocated;
	
public:

	// variables
	uint32_t maxIndex;
	// profiling
	uint32_t cacheMisses;
	uint32_t cacheInvalidations;

	// methods
	
	// constructor
	GPUCache( uint32_t inputMaxIndex, uint32_t inputCacheSize = DEFAULT_CACHE_SIZE )
	{
		// variables
		uint32_t i;
		
		// function body
		lastStartJ = 0;
		lastEndJ = 0;
		g_data_allocated = 0;
		y_data_allocated = 0;
		cacheMisses = 0;
		cacheInvalidations = 0;
		// remember the cache size
		cacheSize = inputCacheSize;
		maxIndex = inputMaxIndex;
		// allocate memory for the index table
		cacheIndexTable = (int*) malloc( sizeof(int) * maxIndex );
		if ( NULL == cacheIndexTable )
		{
			fprintf( stderr, "Error: Failed to allocate GPU cache index table. Exiting\n" );
			exit( -1 );
		}
		// set all the cache indices to an initial, invalid value
		for ( i = 0; i < maxIndex; i++ )
		{
			cacheIndexTable[ i ] = -1;
		}
		// allocate memory for the cache table
		cacheDataTable = (cl_mem*) malloc( sizeof(cl_mem) * cacheSize );
		if ( NULL == cacheDataTable )
		{
			fprintf( stderr, "Error: Failed to allocate GPU cache data table. Exiting\n" );
			exit( -1 );
		}
		currentCacheIndex = 0;
		// set all the slots as unowned
		cacheOwnerTable = (int*) malloc( sizeof(int) * cacheSize );
		for ( i = 0; i < cacheSize; i++ )
		{
			cacheOwnerTable[ i ] = -1;
		}
		
		
		// clean up
	}
	
	~GPUCache()
	{
	
		// debugging
		//fprintf( stdout, "Deconstructing GPU cache\n" );
	
		// variables
		uint32_t i;
		
		// function body
		for ( i = 0; i < maxIndex; i++ )
		{
			if ( -1 != cacheIndexTable[ i ] )
			{
				//fprintf( stdout, "Releasing\n" );
				clReleaseMemObject( cacheDataTable[ cacheIndexTable[ i ] ] );
				//fprintf( stdout, "Finished releasing\n" );
			}
		}
		
		if ( 0 != lastStartJ || 0 != lastEndJ )
		{
			//fprintf( stdout, "Release single saved data\n" );
			clReleaseMemObject( singleSavedData );
			//fprintf( stdout, "Finished Release single saved data\n" );
		}
		
		if ( g_data_allocated )
		{
			//fprintf( stdout, "Releasing G\n" );
			clReleaseMemObject( g_data );
			//fprintf( stdout, "Finished releasing G\n" );
		}
		
		if ( y_data_allocated )
		{
			//fprintf( stdout, "Releasing y\n" );
			clReleaseMemObject( y_data );
			//fprintf( stdout, "Finished Releasing y\n" );
		}
		
		// clean up
		// debugging
		//fprintf( stdout, "Finished Deconstructing GPU cache\n" );
		
	}
	
	inline void InvalidateCache()
	{
		// variables
		uint32_t i;
		
		// function body
		cacheInvalidations++;
		for ( i = 0; i < maxIndex; i++ )
		{
			if ( -1 != cacheIndexTable[ i ] )
			{
				clReleaseMemObject( cacheDataTable[ cacheIndexTable[ i ] ] );
				cacheIndexTable[ i ] = -1;
			}
		}
		if ( 0 != lastStartJ || 0 != lastEndJ )
		{
			clReleaseMemObject( singleSavedData );
			lastStartJ = lastEndJ = 0;
		}
		
		for ( i = 0; i < cacheSize; i++ )
		{
			cacheOwnerTable[ i ] = -1;
		}
		
		currentCacheIndex = 0;
		
		// clean up
	}
	
	// main interface
	inline int CheckCache( uint32_t index, cl_mem * output )
	{
		//fprintf( stdout, "CHECKING CACHE: %x\n", this );
	
		// variables
		uint32_t cacheDataIndex;
		
		// function body
		if ( index > maxIndex )
		{
			return -1;
		}
		// if the value is not in the cache
		if ( -1 == cacheIndexTable[ index ]  )
		{
			// report that
			// debugging
			//fprintf( stdout, "GPU Cache miss\n" );
			cacheMisses++;
			return -1;
		}
		// other wise
		else
		{
			// debugging
			//fprintf( stdout, "GPU Cache hit\n" );
			// set output accordingly
			cacheDataIndex = cacheIndexTable[ index ];
			*output = cacheDataTable[ cacheDataIndex ];
		}
		
		
		// clean up
		return 0;
	}
	
	inline int CacheData( uint32_t index, cl_mem & input )
	{
	
		// debugging
		//fprintf( stdout, "CACHING: %x\n", this );
		// variables
		uint32_t currentOwner;
		
		// function body
		// if there is valid memory in the table at the current index
		currentOwner = cacheOwnerTable[ currentCacheIndex ];
		if ( -1 != currentOwner && index != currentOwner )
		{
			// find out who it belongs to
			// mark the owner as uncached
			cacheIndexTable[ currentOwner ] = -1;
			// free the cached memory
			clReleaseMemObject( cacheDataTable[ currentCacheIndex ] );
		}
		// save the input to memory
		cacheDataTable[ currentCacheIndex ] = input;
		// mark the owner of the current slot as the current index
		cacheOwnerTable[ currentCacheIndex ] = index;
		cacheIndexTable[ index ] = currentCacheIndex;
		// increment the current index
		currentCacheIndex = ( currentCacheIndex + 1 ) % cacheSize;
		
		// clean up
		return 0;
	}
	
	// maintain validity
	inline int SwapIndices( uint32_t i, uint32_t j )
	{
		// variables
		int blockNumberI;
		int blockNumberJ;
		
		// function body
		// get the two indices into the data table
		blockNumberI = cacheIndexTable[ i ];
		blockNumberJ = cacheIndexTable[ j ];
		// swap the ownership of those two blocks
		{
			if ( -1 != blockNumberI )
			{
				cacheOwnerTable[ blockNumberI ] = j;
			}
			if ( -1 != blockNumberJ )
			{
				cacheOwnerTable[ blockNumberJ ] = i;
			}
		}
		// swap the indices into the data table
		cacheIndexTable[ i ] = blockNumberJ;
		cacheIndexTable[ j ] = blockNumberI;
		
		// kill large block
		/*if ( 0 != lastStartJ || 0 != lastEndJ )
		{
			clReleaseMemObject( singleSavedData );
			lastStartJ = lastEndJ = 0;
		}*/
		
		// clean up
		return 0;
	}
	
	// new interface
	inline int CheckCache( uint32_t startJ, uint32_t endJ, cl_mem * output )
	{
		// variables
		
		// function body
		if ( lastStartJ == startJ && lastEndJ == endJ )
		{
			*output = singleSavedData;
		}
		else
		{
			return -1;
		}
		
		// clean up
		return 0;
	}
	
	inline int CacheData( uint32_t startJ, uint32_t endJ, cl_mem & input )
	{
		// variables
		
		// function body
		if ( 0 != lastStartJ || 0 != lastEndJ )
		{
			clReleaseMemObject( singleSavedData );
		}
		lastStartJ = startJ;
		lastEndJ = endJ;
		singleSavedData = input;
		
		// clean up
		return 0;
	}
	
	inline int GetGSpace( cl_mem * output )
	{
		// variables
		
		// function body
		if ( g_data_allocated )
		{
			// return it
			*output = g_data;
			return 0;
		}
		
		// clean up
		return -1;
	}
	
	inline int SaveGSpace( cl_mem & input )
	{
		// variables
		
		// debugging
		fprintf( stdout, "Saving G space\n" );
		
		// function body
		// if there is any space allocated
		if ( g_data_allocated )
		{
			// deallocate it
			clReleaseMemObject( g_data );
		}
		// remember this new space
		g_data = input;
		g_data_allocated = 1;
		
		// clean up
		return 0;
	}
	
	inline int GetYSpace( cl_mem * output )
	{
		// variables
		
		// function body
		if ( y_data_allocated )
		{
			// return it
			*output = y_data;
			return 0;
		}
		
		// clean up
		return -1;
	}
	
	inline int SaveYSpace( cl_mem & input )
	{
		// variables
		
		// function body
		// if there is any space allocated
		if ( y_data_allocated )
		{
			// deallocate it
			clReleaseMemObject( y_data );
		}
		// remember this new space
		y_data = input;
		y_data_allocated = 1;
		
		// clean up
		return 0;
	}
	
	inline int GetASpace( cl_mem * output )
	{
		// variables
		
		// function body
		if ( 0 == lastStartJ && 0 == lastEndJ )
		{
			return -1;
		}
		*output = singleSavedData;
		
		// clean up
		return 0;
	}
	
};

#endif
