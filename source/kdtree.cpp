#include "kdtree.h"

template<typename T, int dim>
Kdtree<T, dim>::Kdtree(std::vector<TVec>& vertices, int thread_num) :
   NodeNum( 0 ), MaxThreadNum( 0 ), MaxSubmitDepth( -1 )
{
   std::vector<T*> coordinates;
   coordinates.reserve( vertices.size() );
   for (auto& v : vertices) coordinates.emplace_back( glm::value_ptr( v ) );

   prepareMultiThreading( thread_num );
   create( coordinates );
}

template<typename T, int dim>
void Kdtree<T, dim>::prepareMultiThreading(int thread_num)
{
   // Calculate the number of child threads to be the number of threads minus 1,
   // then calculate the maximum tree depth at which to launch a child thread.
   // Truncate this depth such that the total number of threads, including the master thread, is an integer power of 2,
   // hence simplifying the launching of child threads by restricting them to only the < branch of the tree for some
   // depth in the tree.
   int n = 0;
   if (thread_num > 0) {
      while (thread_num > 0) {
         ++n;
         thread_num >>= 1;
      }
      thread_num = 1 << (n - 1);
   }
   else thread_num = 0;

   if (thread_num < 2) MaxSubmitDepth = -1; // The sentinel value -1 specifies no child threads.
   else if (thread_num == 2) MaxSubmitDepth = 0;
   else {
      MaxSubmitDepth = static_cast<int>(std::floor( std::log( static_cast<double>(thread_num - 1)) / std::log( 2.0 ) ));
   }
   MaxThreadNum = thread_num;
   std::cout << " >> Max number of threads = " << MaxThreadNum << "\n >> Max submit depth = " << MaxSubmitDepth << "\n\n";
}

template<typename T, int dim>
void Kdtree<T, dim>::sortReferenceAscending(
   T** const reference,
   T** const buffer,
   int low,
   int high,
   int axis,
   int max_submit_depth,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;

      // Is a child thread available to subdivide the lower half of the reference array?
      if (max_submit_depth < 0 || max_submit_depth < depth) {

         // No, recursively subdivide the lower/upper halves of the reference array with the current thread
         sortBufferAscending( reference, buffer, low, mid, axis, max_submit_depth, depth + 1 );
         sortBufferDescending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );

         // Compare the results in the buffer in ascending order and merge them into the reference array in ascending order.
         for (int i = low, j = high, k = low; k <= high; ++k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) < 0 ? buffer[i++] : buffer[j--];
         }
      }
      else {
         // Yes, a child thread is available, so recursively subdivide the lower half of the reference array
         // with a child thread and return the result in the buffer in ascending order.
         auto sort_future = std::async(
            std::launch::async, sortBufferAscending, reference, buffer, low, mid, axis, max_submit_depth, depth + 1
         );

         // And simultaneously, recursively subdivide the upper half of the reference array with the current thread
         // and return the result in the buffer in descending order.
         sortBufferDescending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );

         // Wait for the child thread to finish execution.
         try { sort_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with sort future in sortReferenceAscending()\n" );
         }

         // Compare the results in buffer in ascending order with a child thread
         // and merge them into the lower half of the reference array in ascending order.
         auto merge_future = std::async(
            std::launch::async,
            [&]
            {
               for (int i = low, j = high, k = low; k <= mid; ++k) {
                  reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) <= 0 ? buffer[i++] : buffer[j--];
               }
            }
         );

         // And simultaneously compare the results in the buffer in descending order with the current thread
         // and merge them into the upper half of the reference array in ascending order.
         for (int i = mid, j = mid + 1, k = high; k > mid; --k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) > 0 ? buffer[i--] : buffer[j++];
         }

         // Wait for the child thread to finish execution.
         try { merge_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with merge future in sortReferenceAscending()\n" );
         }
      }
   }
   else {
      // sort in ascending order and leaves the result in the reference array.
      for (int i = low + 1; i <= high; ++i) {
         int j;
         T* const t = reference[i];
         for (j = i; j > low && compareSuperKey( reference[j - 1], t, axis ) > 0; --j) reference[j] = reference[j - 1];
         reference[j] = t;
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortReferenceDescending(
   T** const reference,
   T** const buffer,
   int low,
   int high,
   int axis,
   int max_submit_depth,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (max_submit_depth < 0 || max_submit_depth < depth) {
         sortBufferDescending( reference, buffer, low, mid, axis, max_submit_depth, depth + 1 );
         sortBufferAscending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) > 0 ? buffer[i++] : buffer[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortBufferDescending, reference, buffer, low, mid, axis, max_submit_depth, depth + 1
         );

         sortBufferAscending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );

         try { sort_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with sort future in sortReferenceDescending()\n" );
         }

         auto merge_future = std::async(
            std::launch::async,
            [&]
            {
               for (int i = low, j = high, k = low; k <= mid; ++k) {
                  reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) >= 0 ? buffer[i++] : buffer[j--];
               }
            }
         );

         for (int i = mid, j = mid + 1, k = high; k > mid; --k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) < 0 ? buffer[i--] : buffer[j++];
         }

         try { merge_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with merge future in sortReferenceDescending()\n" );
         }
      }
   }
   else {
      for (int i = low + 1; i <= high; ++i) {
         int j;
         T* const t = reference[i];
         for (j = i; j > low && compareSuperKey( reference[j - 1], t, axis ) < 0; --j) reference[j] = reference[j - 1];
         reference[j] = t;
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortBufferAscending(
   T** const reference,
   T** const buffer,
   int low,
   int high,
   int axis,
   int max_submit_depth,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (max_submit_depth < 0 || max_submit_depth < depth) {
         sortReferenceAscending( reference, buffer, low, mid, axis, max_submit_depth, depth + 1 );
         sortReferenceDescending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) < 0 ? reference[i++] : reference[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortReferenceAscending, reference, buffer, low, mid, axis, max_submit_depth, depth + 1
         );

         sortReferenceDescending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );

         try { sort_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with sort future in sortBufferAscending()\n" );
         }

         auto merge_future = std::async(
            std::launch::async,
            [&]
            {
               for (int i = low, j = high, k = low; k <= mid; ++k) {
                  buffer[k] = compareSuperKey( reference[i], reference[j], axis ) <= 0 ? reference[i++] : reference[j--];
               }
            }
         );

         for (int i = mid, j = mid + 1, k = high; k > mid; --k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) > 0 ? reference[i--] : reference[j++];
         }

         try { merge_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with merge future in sortBufferAscending()\n" );
         }
      }
   }
   else {
      int i, j;
      buffer[high] = reference[high];
      for (j = high - 1; j >= low; --j) {
         for (i = j; i < high; ++i) {
            if (compareSuperKey( reference[j], reference[i + 1], axis ) > 0) buffer[i] = buffer[i + 1];
            else break;
         }
         buffer[i] = reference[j];
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortBufferDescending(
   T** const reference,
   T** const buffer,
   int low,
   int high,
   int axis,
   int max_submit_depth,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (max_submit_depth < 0 || max_submit_depth < depth) {
         sortReferenceDescending( reference, buffer, low, mid, axis, max_submit_depth, depth + 1 );
         sortReferenceAscending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) > 0 ? reference[i++] : reference[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortReferenceDescending, reference, buffer, low, mid, axis, max_submit_depth, depth + 1
         );

         sortReferenceAscending( reference, buffer, mid + 1, high, axis, max_submit_depth, depth + 1 );

         try { sort_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with sort future in sortBufferDescending()\n" );
         }

         auto merge_future = std::async(
            std::launch::async,
            [&]
            {
               for (int i = low, j = high, k = low; k <= mid; ++k) {
                  buffer[k] = compareSuperKey( reference[i], reference[j], axis ) >= 0 ? reference[i++] : reference[j--];
               }
            }
         );

         for (int i = mid, j = mid + 1, k = high; k > mid; --k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) < 0 ? reference[i--] : reference[j++];
         }

         try { merge_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with merge future in sortBufferDescending()\n" );
         }
      }
   }
   else {
      int i, j;
      buffer[high] = reference[high];
      for (j = high - 1; j >= low; --j) {
         for (i = j; i < high; ++i) {
            if (compareSuperKey( reference[j], reference[i + 1], axis ) < 0) buffer[i] = buffer[i + 1];
            else break;
         }
         buffer[i] = reference[j];
      }
   }
}

template<typename T, int dim>
int Kdtree<T, dim>::removeDuplicates(T** const reference, int leading_dim_for_super_key, int size)
{
   int end = 0;
	for (int j = 1; j < size; ++j) {
      const T compare = compareSuperKey( reference[j], reference[end], leading_dim_for_super_key );
      if (compare < 0) {
         std::ostringstream buffer;
         buffer << "sort failure: compareSuperKey( reference[" << j << "], reference[" << j - 1 << "], ("
            << leading_dim_for_super_key << ") = " << compare  << "\n";
		   throw std::runtime_error( buffer.str() );
      }
      else if (compare > 0) reference[++end] = reference[j];
      else delete [] reference[j];
	}
	return end;
}

template<typename T, int dim>
std::shared_ptr<typename Kdtree<T, dim>::KdtreeNode> Kdtree<T, dim>::build(
   T*** const references,
   const std::vector<std::vector<int>>& permutation,
   int start,
   int end,
   int max_submit_depth,
   int depth
)
{
   std::shared_ptr<KdtreeNode> node;

   // The partition permutes as x, y, z, w... and specifies the most significant key.
   const int axis = permutation[depth][permutation[0].size() - 1];

   // Obtain the reference array that corresponds to the most significant key.
   T** const reference = references[permutation[depth][dim]];

   if (end == start) {
      // Only one reference was passed to this function, so add it to the tree.
      node = std::make_shared<KdtreeNode>( reference[end] );
   }
   else if (end == start + 1) {
      // Two references were passed to this function in sorted order, so store the start element at this level of
      // the tree and store the end element as the > child.
      node = std::make_shared<KdtreeNode>( reference[start] );
      node->RightChild = std::make_shared<KdtreeNode>( reference[end] );
   }
   else if (end == start + 2) {
      // Three references were passed to this function in sorted order, so store the median element at this level of
      // the tree, store the start element as the < child and store the end element as the > child.
      node = std::make_shared<KdtreeNode>( reference[start + 1] );
      node->LeftChild = std::make_shared<KdtreeNode>( reference[start] );
      node->RightChild = std::make_shared<KdtreeNode>( reference[end] );
	}
   else if (end > start + 2) {
      // More than three references were passed to this function, so the median element of references[0] is chosen
      // as the tuple about which the other reference arrays will be partitioned.
      // Avoid overflow when computing the median.
      const int median = start + (end - start) / 2;

      // Store the median element of references[0] in a new kdNode.
      node = std::make_shared<KdtreeNode>( reference[median] );

      // Build both branches with child threads at as many levels of the tree as possible.
      // Create the child threads as high in the tree as possible.
      // Are child threads available to build both branches of the tree?
      if (max_submit_depth < 0 || max_submit_depth < depth) {
         // No, child threads are not available, so one thread will be used.
         // Initialize start_index = 1 so that the 'for' loop that partitions the reference arrays will partition
         // a number of arrays equal to dim.
         int start_index = 1;

         // If depth < dim - 1, copy references[permut[dim]] to references[permut[0]] where permut is the permutation
         // vector for this level of the tree.
         // Sort the two halves of references[permut[0]] with p + 1 as the most significant key of the super key.
         // Use as the temporary array references[permut[1]] because that array is not used for partitioning.
         // Partition a number of reference arrays equal to the tree depth because those reference arrays are already sorted.
         if (depth < dim - 1) {
            start_index = dim - depth;
            T** const target = references[permutation[depth][0]];
            T** const buffer = references[permutation[depth][1]];
            for (int i = start; i <= end; ++i) target[i] = reference[i];

            // Sort the lower half of references[permut[0]] with the current thread.
            sortReferenceAscending( target, buffer, start, median - 1, axis + 1, max_submit_depth, depth );
            // Sort the upper half of references[permut[0]] with the current thread.
            sortReferenceAscending( target, buffer, median + 1, end, axis + 1, max_submit_depth, depth );
         }

         // Partition the reference arrays specified by 'startIndex' in a priori sorted order by comparing super keys.
         // Store the result from references[permut[i]]] in references[permut[i - 1]] where permut is the permutation
         // vector for this level of the tree, thus permuting the reference arrays.
         // Skip the element of references[permut[i]] that equals the tuple that is stored in the new KdNode.
         T* const tuple = node->Tuple;
         for (int i = start_index; i < dim; ++i) {
            // Specify the source and destination reference arrays.
            T** const src = references[permutation[depth][i]];
            T** const dst = references[permutation[depth][i - 1]];

            // Fill the lower and upper halves of one reference array
            // in ascending order with the current thread.
            for (int j = start, lower = start - 1, upper = median; j <= end; ++j) {
               T* const src_j = src[j];
               const T compare = compareSuperKey( src_j, tuple, axis );
               if (compare < 0) dst[++lower] = src_j;
               else if (compare > 0) dst[++upper] = src_j;
            }
         }

         // Recursively build the < branch of the tree with the current thread.
         node->LeftChild = build( references, permutation, start, median - 1, max_submit_depth, depth + 1 );

         // Then recursively build the > branch of the tree with the current thread.
         node->RightChild = build( references, permutation, median + 1, end, max_submit_depth, depth + 1 );
      }
      else {
         // Yes, child threads are available, so two threads will be used.
         // Initialize end_index = 0 so that the 'for' loop that partitions the reference arrays will partition
         // a number of arrays equal to dim.
         int start_index = 1;

         // If depth < dim-1, copy references[permut[dim]] to references[permut[0]] where permut is the permutation
         // vector for this level of the tree.
         // Sort the two halves of references[permut[0]] with p+1 as the most significant key of the super key.
         // Use as the temporary array references[permut[1]] because that array is not used for partitioning.
         // Partition a number of reference arrays equal to the tree depth because those reference arrays are already sorted.
         if (depth < dim - 1) {
            start_index = dim - depth;
            T** const target = references[permutation[depth][0]];
            T** const buffer = references[permutation[depth][1]];

            // Copy and sort the lower half of references[permut[0]] with a child thread.
            auto copy_future = std::async(
               std::launch::async,
               [&]
               {
                  for (int i = start; i <= median - 1; ++i) target[i] = reference[i];
                  sortReferenceAscending( target, buffer, start, median - 1, axis + 1, max_submit_depth, depth );
               }
            );

            // Copy and sort the upper half of references[permut[0]] with the current thread.
            for (int i = median + 1; i <= end; ++i) target[i] = reference[i];
            sortReferenceAscending( target, buffer, median + 1, end, axis + 1, max_submit_depth, depth );

            // Wait for the child thread to finish execution.
            try { copy_future.get(); }
            catch (const std::exception& e) {
               throw std::runtime_error( "error with copy future in build()\n" );
            }
         }

         // Create a copy of the node->tuple array so that the current thread
         // and the child thread do not contend for read access to this array.
         T* point = new T[dim];
         const T* tuple = node->Tuple;
         for (int i = 0; i < dim; ++i) point[i] = tuple[i];

         // Partition the reference arrays specified by 'start_index' in a priori sorted order by comparing super keys.
         // Store the result from references[permut[i]]] in references[permut[i - 1]] where permut is the permutation
         // vector for this level of the tree, thus permuting the reference arrays.
         // Skip the element of references[permut[i]] that equals the tuple that is stored in the new KdNode.
         for (int i = start_index; i < dim; ++i) {
            // Specify the source and destination reference arrays.
            T** const src = references[permutation[depth][i]];
            T** const dst = references[permutation[depth][i - 1]];

            // Two threads may be used to partition the reference arrays, analogous to
            // the use of two threads to merge the results for the merge sort algorithm.
            // Fill one reference array in ascending order with a child thread.
            auto partition_future = std::async( std::launch::async,
               [&]
               {
                  for (int lower = start - 1, upper = median, j = start; j <= median; ++j) {
                     T* const src_j = src[j];
                     const T compare = compareSuperKey( src_j, point, axis );
                     if (compare < 0) dst[++lower] = src_j;
                     else if (compare > 0) dst[++upper] = src_j;
                  }
               }
            );

            // Simultaneously fill the same reference array in descending order with the current thread.
            for (int lower = median, upper = end + 1, k = end; k > median; --k) {
               T* const src_k = src[k];
               const T compare = compareSuperKey( src_k, tuple, axis );
               if (compare < 0) dst[--lower] = src_k;
               else if (compare > 0) dst[--upper] = src_k;
            }

            // Wait for the child thread to finish execution.
            try { partition_future.get(); }
            catch (const std::exception& e) {
               throw std::runtime_error( "error with partition future in build()\n" );
            }
         }
         delete [] point;

         // Recursively build the < branch of the tree with a child thread.
         auto build_future = std::async(
            std::launch::async,
            build, references, std::ref( permutation ), start, median - 1, max_submit_depth, depth + 1
         );

         // And simultaneously build the > branch of the tree with the current thread.
         node->RightChild = build( references, permutation, median + 1, end, max_submit_depth, depth + 1 );

         // Wait for the child thread to finish execution.
         try { node->LeftChild = build_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with build future in build()\n" );
         }
      }
   }
   else if (end < start) {
      std::ostringstream buffer;
      buffer << "error at depth = " << depth << " : end = " << end << "  <  start = " << start << " in build()\n";
      throw std::runtime_error( buffer.str() );
   }

	// Return the pointer to the root of the k-d tree.
	return node;
}

template<typename T, int dim>
void Kdtree<T, dim>::createPermutation(std::vector<int>& permutation, int coordinates_num)
{
   int max_depth = 1;
   int size = coordinates_num;
   while (size > 0) {
      ++max_depth;
      size >>= 1;
   }

   permutation.resize( max_depth );
   for (size_t i = 0; i < permutation.size(); ++i) permutation[i] = i % dim;
}

template<typename T, int dim>
int Kdtree<T, dim>::verify(KdtreeNode* node, const std::vector<int>& permutation, int max_submit_depth, int depth) const
{
   int count = 1;
	if (node->Tuple == nullptr) throw std::runtime_error( "point is null in verify()\n" );

	// The partition cycles as x, y, z, w...
	const int axis = permutation[depth];
   if (node->LeftChild != nullptr) {
      if (node->LeftChild->Tuple[axis] > node->Tuple[axis]) {
         throw std::runtime_error( "child is > node in verify()\n" );
      }
      if (compareSuperKey( node->LeftChild->Tuple, node->Tuple, axis ) >= 0) {
         throw std::runtime_error( "child is >= node in verify()\n" );
      }
	}
	if (node->RightChild != nullptr) {
      if (node->RightChild->Tuple[axis] < node->Tuple[axis]) {
         throw std::runtime_error( "child is < node in verify()\n" );
      }
      if (compareSuperKey( node->RightChild->Tuple, node->Tuple, axis ) <= 0) {
         throw std::runtime_error( "child is <= node in verify()\n" );
      }
	}

   // Verify the < branch with a child thread at as many levels of the tree as possible.
   // Create the child thread as high in the tree as possible for greater utilization.

   // Is a child thread available to verify the < branch?
   if (max_submit_depth < 0 || max_submit_depth < depth) {
      // No, so verify the < branch with the current thread.
      if (node->LeftChild != nullptr) {
         count += verify( node->LeftChild.get(), permutation, max_submit_depth, depth + 1 );
      }

      // Then verify the > branch with the current thread.
      if (node->RightChild != nullptr) {
         count += verify( node->RightChild.get(), permutation, max_submit_depth, depth + 1 );
      }
   }
   else {
      // Yes, so verify the < branch with a child thread.
      // Note that a lambda is required because this verify() function is not static.
      // The use of std::ref may be unnecessary in view of the [&] lambda argument specification.
      std::future<int> verify_future;
      if (node->LeftChild != nullptr) {
         verify_future = std::async(
            std::launch::async,
            [&]{ return verify( node->LeftChild.get(), std::ref( permutation ), max_submit_depth, depth + 1 ); }
         );
      }

      // And simultaneously verify the > branch with the current thread.
      int right_count = 0;
      if (node->RightChild != nullptr) {
         right_count = verify( node->RightChild.get(), permutation, max_submit_depth, depth + 1 );
      }

      // Wait for the child thread to finish execution.
      int left_count = 0;
      if (node->LeftChild != nullptr) {
         try { left_count = verify_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with verify future in verify()\n" );
         }
      }

      // Sum the counts returned by the child and current threads.
      count += left_count + right_count;
   }
	return count;
}

template<typename T, int dim>
void Kdtree<T, dim>::create(std::vector<T*>& coordinates)
{
   T*** const references = new T**[dim + 1];
   references[0] = coordinates.data();
   const auto size = static_cast<int>(coordinates.size());
   for (int i = 1; i <= dim; ++i) references[i] = new T*[size];

   auto start_time = std::chrono::system_clock::now();
   sortReferenceAscending( references[0], references[dim], 0, size - 1, 0, MaxSubmitDepth, 0 );
   auto end_time = std::chrono::system_clock::now();
   const auto sort_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   start_time = std::chrono::system_clock::now();
   const int end = removeDuplicates( references[0], 0, size );
   end_time = std::chrono::system_clock::now();
   const auto remove_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   start_time = std::chrono::system_clock::now();
   int max_depth = 1, s = size;
   while (s > 0) {
      max_depth++;
      s >>= 1;
   }

   std::vector<int> indices(dim + 2);
   for (int i = 0; i <= dim; ++i) indices[i] = i;

   std::vector<std::vector<int>> permutation(max_depth, std::vector<int>(dim + 2));
   for (size_t i = 0; i < permutation.size(); ++i) {
      indices[dim + 1] = i % dim;
      std::swap( indices[0], indices[dim] );
      permutation[i] = indices;
      std::swap( indices[dim - 1], indices[dim] );
   }

   Root = build( references, permutation, 0, end, MaxSubmitDepth, 0 );
   end_time = std::chrono::system_clock::now();
   const auto build_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   start_time = std::chrono::system_clock::now();
   std::vector<int> verify_permutation;
   createPermutation( verify_permutation, size );
   NodeNum = verify( Root.get(), verify_permutation, MaxSubmitDepth, 0 );
   end_time = std::chrono::system_clock::now();
   const auto verify_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << "\n>> Number of nodes = " << NodeNum << "\n";

   std::cout <<
      " >> Total Time = " << std::fixed << std::setprecision( 2 ) << sort_time + remove_time + build_time + verify_time
      << "\n >> Sort Time = " << sort_time << "\n >> Remove Time = " << remove_time
      << "\n >> Build Time = " << build_time << "\n >> Verify Time = " << verify_time << "\n\n";

   for (int i = 1; i <= dim; ++i) delete [] references[i];
   delete [] references;
}

template<typename T, int dim>
std::list<typename Kdtree<T, dim>::KdtreeNode*> Kdtree<T, dim>::search(
   KdtreeNode* node,
   const TVec& query,
   T radius,
   int depth
)
{
   // The partition cycles as x, y, z, w...
   const int axis = depth % dim;

	// If the distance from the query node to the node is within the radius in all k dimensions, add the node to a list.
   bool inside = true;
   std::list<KdtreeNode*> found;
	for (int i = 0; i < dim; ++i) {
      if (std::abs( query[i] - node->Tuple[i]) > radius) {
         inside = false;
         break;
      }
	}
	if (inside) found.emplace_back( node ); // The push_back function expects a KdNode for a call by reference.

	// Search the < branch of the k-d tree if the partition coordinate of the query point minus the radius is
   // <= the partition coordinate of the node. The < branch must be searched when the radius equals the partition
   // coordinate because the super key may assign a point to either branch of the tree if the sorting or partition
   // coordinate, which forms the most significant portion of the super key, shows equality.
	if (node->LeftChild != nullptr && (query[axis] - radius) <= node->Tuple[axis]) {
	    std::list<KdtreeNode*> left = search( node->LeftChild.get(), query, radius, depth + 1 );
	    found.splice( found.end(), left ); // Can't substitute search() for left.
	}

	// Search the > branch of the k-d tree if the partition coordinate of the query point plus the radius is
   // >= the partition coordinate of the k-d node. The < branch must be searched when the radius equals the partition
   // coordinate because the super key may assign a point to either branch of the tree if the sorting or partition
   // coordinate, which forms the most significant portion of the super key, shows equality.
	if (node->RightChild != nullptr && (query[axis] + radius) >= node->Tuple[axis]) {
	    std::list<KdtreeNode*> right = search( node->RightChild.get(), query, radius, depth + 1 );
	    found.splice( found.end(), right ); // Can't substitute search() for right.
	}

	return found;
}

template<typename T, int dim>
void Kdtree<T, dim>::print(KdtreeNode* node, int depth) const
{
   if (node->RightChild != nullptr) print( node->RightChild.get(), depth + 1 );

	for (int i = 0; i < depth; ++i) std::cout << "       ";

   const T* tuple = node->Tuple;
   std::cout << "(" << tuple[0] << ",";
   for (int i = 1; i < dim - 1; ++i) std::cout << tuple[i] << ",";
   std::cout << tuple[dim - 1] << ")\n";

	if (node->LeftChild != nullptr) print( node->LeftChild.get(), depth + 1 );
}