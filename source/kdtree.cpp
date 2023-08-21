#include "kdtree.h"

template<typename T, int dim>
Kdtree<T, dim>::Kdtree(const std::vector<TVec>& vertices, int thread_num) : NodeNum( 0 )
{
   std::vector<const T*> coordinates;
   coordinates.reserve( vertices.size() );
   for (const auto& v : vertices) coordinates.emplace_back( glm::value_ptr( v ) );

   prepareMultiThreading( thread_num );
   create( coordinates );
}

template<typename T, int dim>
void Kdtree<T, dim>::prepareMultiThreading(int thread_num)
{
   int n = 0;
   if (thread_num > 0) {
      while (thread_num > 0) {
         ++n;
         thread_num >>= 1;
      }
      thread_num = 1 << (n - 1);
   }
   else thread_num = 0;

   if (thread_num < 2) MaxSubmitDepth = -1;
   else if (thread_num == 2) MaxSubmitDepth = 0;
   else MaxSubmitDepth = static_cast<int>(std::floor( std::log2( static_cast<double>(thread_num - 1) ) ));
   MaxThreadNum = thread_num;
   std::cout << " >> Max number of threads = " << MaxThreadNum << "\n >> Max submit depth = " << MaxSubmitDepth << "\n";
}

template<typename T, int dim>
void Kdtree<T, dim>::sortReferenceAscending(
   const T** reference,
   const T** buffer,
   int low,
   int high,
   int axis,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (!isThreadAvailable( depth )) {
         sortBufferAscending( reference, buffer, low, mid, axis, depth + 1 );
         sortBufferDescending( reference, buffer, mid + 1, high, axis, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) < 0 ? buffer[i++] : buffer[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortBufferAscending, reference, buffer, low, mid, axis, depth + 1
         );

         sortBufferDescending( reference, buffer, mid + 1, high, axis, depth + 1 );

         try { sort_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with sort future in sortReferenceAscending()\n" );
         }

         auto merge_future = std::async(
            std::launch::async,
            [&]
            {
               for (int i = low, j = high, k = low; k <= mid; ++k) {
                  reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) <= 0 ? buffer[i++] : buffer[j--];
               }
            }
         );

         for (int i = mid, j = mid + 1, k = high; k > mid; --k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) > 0 ? buffer[i--] : buffer[j++];
         }

         try { merge_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with merge future in sortReferenceAscending()\n" );
         }
      }
   }
   else {
      for (int i = low + 1; i <= high; ++i) {
         int j;
         const T* t = reference[i];
         for (j = i; j > low && compareSuperKey( reference[j - 1], t, axis ) > 0; --j) reference[j] = reference[j - 1];
         reference[j] = t;
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortReferenceDescending(
   const T** reference,
   const T** buffer,
   int low,
   int high,
   int axis,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (!isThreadAvailable( depth )) {
         sortBufferDescending( reference, buffer, low, mid, axis, depth + 1 );
         sortBufferAscending( reference, buffer, mid + 1, high, axis, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) > 0 ? buffer[i++] : buffer[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortBufferDescending, reference, buffer, low, mid, axis, depth + 1
         );

         sortBufferAscending( reference, buffer, mid + 1, high, axis, depth + 1 );

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
         const T* t = reference[i];
         for (j = i; j > low && compareSuperKey( reference[j - 1], t, axis ) < 0; --j) reference[j] = reference[j - 1];
         reference[j] = t;
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortBufferAscending(
   const T** reference,
   const T** buffer,
   int low,
   int high,
   int axis,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (!isThreadAvailable( depth )) {
         sortReferenceAscending( reference, buffer, low, mid, axis, depth + 1 );
         sortReferenceDescending( reference, buffer, mid + 1, high, axis, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) < 0 ? reference[i++] : reference[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortReferenceAscending, reference, buffer, low, mid, axis, depth + 1
         );

         sortReferenceDescending( reference, buffer, mid + 1, high, axis, depth + 1 );

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
      buffer[high] = reference[high];
      for (int j = high - 1; j >= low; --j) {
         int i;
         for (i = j; i < high && compareSuperKey( reference[j], buffer[i + 1], axis ) > 0; ++i) {
            buffer[i] = buffer[i + 1];
         }
         buffer[i] = reference[j];
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortBufferDescending(
   const T** reference,
   const T** buffer,
   int low,
   int high,
   int axis,
   int depth
)
{
   if (high - low > InsertionSortThreshold) {
      const int mid = low + (high - low) / 2;
      if (!isThreadAvailable( depth )) {
         sortReferenceDescending( reference, buffer, low, mid, axis, depth + 1 );
         sortReferenceAscending( reference, buffer, mid + 1, high, axis, depth + 1 );
         for (int i = low, j = high, k = low; k <= high; ++k) {
            buffer[k] = compareSuperKey( reference[i], reference[j], axis ) > 0 ? reference[i++] : reference[j--];
         }
      }
      else {
         auto sort_future = std::async(
            std::launch::async, sortReferenceDescending, reference, buffer, low, mid, axis, depth + 1
         );

         sortReferenceAscending( reference, buffer, mid + 1, high, axis, depth + 1 );

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
      buffer[high] = reference[high];
      for (int j = high - 1; j >= low; --j) {
         int i;
         for (i = j; i < high && compareSuperKey( reference[j], buffer[i + 1], axis ) < 0; ++i) {
            buffer[i] = buffer[i + 1];
         }
         buffer[i] = reference[j];
      }
   }
}

template<typename T, int dim>
int Kdtree<T, dim>::removeDuplicates(const T** reference, int size)
{
   int end = 0;
	for (int j = 1; j < size; ++j) {
      const T compare = compareSuperKey( reference[j], reference[end], 0 );
      if (compare < 0) {
         std::ostringstream buffer;
         buffer << "sort error: (reference[" << j << "], reference[" << j - 1 << "]) = " << compare << "\n";
		   throw std::runtime_error( buffer.str() );
      }
      else if (compare > 0) reference[++end] = reference[j];
	}
	return end;
}

template<typename T, int dim>
std::shared_ptr<typename Kdtree<T, dim>::KdtreeNode> Kdtree<T, dim>::build(
   const T*** references,
   const std::vector<std::vector<int>>& permutation,
   int start,
   int end,
   int depth
)
{
   std::shared_ptr<KdtreeNode> node;
   const std::vector<int>& p = permutation[depth];
   const int axis = p.back();
   const T** reference = references[p[dim]];
   if (end == start) node = std::make_shared<KdtreeNode>( reference[end] );
   else if (end == start + 1) {
      node = std::make_shared<KdtreeNode>( reference[start] );
      node->RightChild = std::make_shared<KdtreeNode>( reference[end] );
   }
   else if (end == start + 2) {
      node = std::make_shared<KdtreeNode>( reference[start + 1] );
      node->LeftChild = std::make_shared<KdtreeNode>( reference[start] );
      node->RightChild = std::make_shared<KdtreeNode>( reference[end] );
	}
   else if (end > start + 2) {
      const int mid = start + (end - start) / 2;
      node = std::make_shared<KdtreeNode>( reference[mid] );
      if (!isThreadAvailable( depth )) {
         int start_index = 1;
         if (depth < dim - 1) {
            start_index = dim - depth;
            const T** target = references[p[0]];
            const T** buffer = references[p[1]];
            for (int i = start; i <= end; ++i) target[i] = reference[i];
            sortReferenceAscending( target, buffer, start, mid - 1, axis + 1, depth );
            sortReferenceAscending( target, buffer, mid + 1, end, axis + 1, depth );
         }

         const T* tuple = node->Tuple;
         for (int i = start_index; i < dim; ++i) {
            const T** src = references[p[i]];
            const T** dst = references[p[i - 1]];
            for (int j = start, left = start, right = mid + 1; j <= end; ++j) {
               const T compare = compareSuperKey( src[j], tuple, axis );
               if (compare < 0) dst[left++] = src[j];
               else if (compare > 0) dst[right++] = src[j];
            }
         }

         node->LeftChild = build( references, permutation, start, mid - 1, depth + 1 );
         node->RightChild = build( references, permutation, mid + 1, end, depth + 1 );
      }
      else {
         int start_index = 1;
         if (depth < dim - 1) {
            start_index = dim - depth;
            const T** target = references[p[0]];
            const T** buffer = references[p[1]];
            auto copy_future = std::async(
               std::launch::async,
               [&]
               {
                  for (int i = start; i <= mid - 1; ++i) target[i] = reference[i];
                  sortReferenceAscending( target, buffer, start, mid - 1, axis + 1, depth );
               }
            );

            for (int i = mid + 1; i <= end; ++i) target[i] = reference[i];
            sortReferenceAscending( target, buffer, mid + 1, end, axis + 1, depth );

            try { copy_future.get(); }
            catch (const std::exception& e) {
               throw std::runtime_error( "error with copy future in build()\n" );
            }
         }

         T* copied_tuple = new T[dim];
         const T* tuple = node->Tuple;
         for (int i = 0; i < dim; ++i) copied_tuple[i] = tuple[i];

         for (int i = start_index; i < dim; ++i) {
            const T** src = references[p[i]];
            const T** dst = references[p[i - 1]];
            auto partition_future = std::async(
               std::launch::async,
               [&]
               {
                  for (int j = start, left = start, right = mid + 1; j <= mid; ++j) {
                     const T compare = compareSuperKey( src[j], copied_tuple, axis );
                     if (compare < 0) dst[left++] = src[j];
                     else if (compare > 0) dst[right++] = src[j];
                  }
               }
            );

            for (int k = end, left = mid - 1, right = end; k > mid; --k) {
               const T compare = compareSuperKey( src[k], tuple, axis );
               if (compare < 0) dst[left--] = src[k];
               else if (compare > 0) dst[right--] = src[k];
            }

            try { partition_future.get(); }
            catch (const std::exception& e) {
               throw std::runtime_error( "error with partition future in build()\n" );
            }
         }
         delete [] copied_tuple;

         auto build_future = std::async(
            std::launch::async,
            build, references, std::ref( permutation ), start, mid - 1, depth + 1
         );

         node->RightChild = build( references, permutation, mid + 1, end, depth + 1 );

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
int Kdtree<T, dim>::verify(const KdtreeNode* node, int depth) const
{
   int count = 1;
	if (node->Tuple == nullptr) throw std::runtime_error( "point is null in verify()\n" );

	const int axis = Permutation[depth];
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

   if (!isThreadAvailable( depth )) {
      if (node->LeftChild != nullptr) count += verify( node->LeftChild.get(), depth + 1 );
      if (node->RightChild != nullptr) count += verify( node->RightChild.get(), depth + 1 );
   }
   else {
      std::future<int> verify_future;
      if (node->LeftChild != nullptr) {
         verify_future = std::async( std::launch::async, [&]{ return verify( node->LeftChild.get(), depth + 1 ); } );
      }

      int right_count = 0;
      if (node->RightChild != nullptr) right_count = verify( node->RightChild.get(), depth + 1 );

      int left_count = 0;
      if (node->LeftChild != nullptr) {
         try { left_count = verify_future.get(); }
         catch (const std::exception& e) {
            throw std::runtime_error( "error with verify future in verify()\n" );
         }
      }

      count += left_count + right_count;
   }
	return count;
}

template<typename T, int dim>
void Kdtree<T, dim>::create(std::vector<const T*>& coordinates)
{
   const T*** references = new const T**[dim + 1];
   const auto size = static_cast<int>(coordinates.size());
   for (int i = 1; i <= dim; ++i) references[i] = new const T*[size];
   references[0] = coordinates.data();

   auto start_time = std::chrono::system_clock::now();
   sortReferenceAscending( references[0], references[dim], 0, size - 1, 0, 0 );
   auto end_time = std::chrono::system_clock::now();
   const auto sort_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   start_time = std::chrono::system_clock::now();
   const int end = removeDuplicates( references[0], size );
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
   std::iota( indices.begin(), indices.end() - 1, 0 );
   std::vector<std::vector<int>> permutation(max_depth, std::vector<int>(indices.size()));
   for (int i = 0; i < max_depth; ++i) {
      indices.back() = i % dim;
      std::swap( indices[0], indices[dim] );
      permutation[i] = indices;
      std::swap( indices[dim - 1], indices[dim] );
   }
   Root = build( references, permutation, 0, end, 0 );
   end_time = std::chrono::system_clock::now();
   const auto build_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   start_time = std::chrono::system_clock::now();
   createPermutation( Permutation, size );
   NodeNum = verify( Root.get(), 0 );
   end_time = std::chrono::system_clock::now();
   const auto verify_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

   std::cout << " >> Number of nodes = " << NodeNum << "\n" << std::fixed << std::setprecision( 2 )
      << " >> Total Time = "  << sort_time + remove_time + build_time + verify_time << " sec."
      << "\n\t* Sort Time = " << sort_time << " sec.\n\t* Remove Time = " << remove_time << " sec."
      << "\n\t* Build Time = " << build_time << " sec.\n\t* Verify Time = " << verify_time << " sec.\n\n";

   for (int i = 1; i <= dim; ++i) delete [] references[i];
   delete [] references;
}

template<typename T, int dim>
bool Kdtree<T, dim>::isInside(
   const KdtreeNode* node,
   const TVec& lower,
   const TVec& upper
)
{
   bool inside = true;
   for (int i = 0; i < dim; ++i) {
      if (lower[i] > node->Tuple[i] || upper[i] < node->Tuple[i]) {
         inside = false;
         break;
      }
   }
   return inside;
}

template<typename T, int dim>
void Kdtree<T, dim>::search(
   std::list<const KdtreeNode*>& found,
   const KdtreeNode* node,
   const TVec& lower,
   const TVec& upper,
   const std::vector<int>& permutation,
   int depth
)
{
   const int axis = permutation[depth];
   if (isInside( node, lower, upper )) found.push_front( node );

   const bool search_left = node->LeftChild != nullptr &&
      (compareSuperKey( node->Tuple, glm::value_ptr( lower ), axis ) >= 0);
   const bool search_right = node->RightChild != nullptr &&
      (compareSuperKey( node->Tuple, glm::value_ptr( upper ), axis ) <= 0);
   if (search_left && search_right && isThreadAvailable( depth )) {
      std::list<const KdtreeNode*> left;
      auto search_future = std::async(
         std::launch::async,
         [&]
         {
            search(
               std::ref( left ), node->LeftChild.get(),
               std::ref( lower ), std::ref( upper ), std::ref( permutation ), depth + 1
            );
         }
      );

      std::list<const KdtreeNode*> right;
      search( right, node->RightChild.get(), lower, upper, permutation, depth + 1 );

      try { search_future.get(); }
      catch (const std::exception& e) {
         throw std::runtime_error( "error with search future in search()\n" );
      }

      found.splice( found.end(), left );
      found.splice( found.end(), right );
   }
   else {
      if (search_left) {
         std::list<const KdtreeNode*> left;
         search( left, node->LeftChild.get(), lower, upper, permutation, depth + 1 );
         found.splice( found.end(), left );
      }
      if (search_right) {
         std::list<const KdtreeNode*> right;
         search( right, node->RightChild.get(), lower, upper, permutation, depth + 1 );
         found.splice( found.end(), right );
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::findNearestNeighbors(
   Finder& finder,
   const KdtreeNode* node,
   const TVec& query,
   int depth
) const
{
   const int axis = Permutation[depth];
   if (query[axis] < node->Tuple[axis]) {
      if (node->LeftChild != nullptr) findNearestNeighbors( finder, node->LeftChild.get(), query, depth + 1 );

      const T d = node->Tuple[axis] - query[axis];
      if (d * d <= finder.getMaxSquaredDistance() || !finder.isFull()) {
         if (node->RightChild != nullptr) findNearestNeighbors( finder, node->RightChild.get(), query, depth + 1 );
         finder.add( node, query );
      }
   }
   else if (query[axis] > node->Tuple[axis]) {
      if (node->RightChild != nullptr) findNearestNeighbors( finder, node->RightChild.get(), query, depth + 1 );

      const T d = node->Tuple[axis] - query[axis];
      if (d * d <= finder.getMaxSquaredDistance() || !finder.isFull()) {
         if (node->LeftChild != nullptr) findNearestNeighbors( finder, node->LeftChild.get(), query, depth + 1 );
         finder.add( node, query );
      }
   }
   else {
      if (node->LeftChild != nullptr) findNearestNeighbors( finder, node->LeftChild.get(), query, depth + 1 );
      if (node->RightChild != nullptr) findNearestNeighbors( finder, node->RightChild.get(), query, depth + 1 );
      finder.add( node, query );
   }
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