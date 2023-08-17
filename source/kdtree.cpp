#include "kdtree.h"

template<typename T, int dim>
Kdtree<T, dim>::Kdtree(const std::vector<TVec>& vertices) : NodeNum( 0 )
{
   std::vector<const T*> coordinates;
   coordinates.reserve( vertices.size() );
   for (const auto& v : vertices) coordinates.emplace_back( glm::value_ptr( v ) );

   create( coordinates );
}

template<typename T, int dim>
void Kdtree<T, dim>::sort(std::vector<const T*>& reference, std::vector<const T*>& buffer, int low, int high, int axis)
{
   if (low < high) {
      const int mid = low + (high - low) / 2;
      sort( reference, buffer, low, mid, axis );
      sort( reference, buffer, mid + 1, high, axis );

      int i, j;
      for (i = mid + 1; i > low; --i) buffer[i - 1] = reference[i - 1];
      for (j = mid; j < high; ++j) buffer[mid + high - j] = reference[j + 1];
      for (int k = low; k <= high; ++k) {
         reference[k] = compareSuperKey( buffer[i], buffer[j], axis ) < 0 ? buffer[i++] : buffer[j--];
      }
   }
}

template<typename T, int dim>
void Kdtree<T, dim>::sortReferenceAscending(
   const T** reference,
   const T** buffer,
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
   const T** reference,
   const T** buffer,
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
int Kdtree<T, dim>::removeDuplicates(const T** reference, int leading_dim_for_super_key, int size)
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
   const T*** references,
   std::vector<const T*>& buffer,
   int start,
   int end,
   int depth
)
{
   std::shared_ptr<KdtreeNode> node;

	// The axis permutes as x, y, z, w... and addresses the referenced data.
   int axis = depth % dim;
   if (end == start) {
      // Only one reference was passed to this function, so add it to the tree.
      node = std::make_shared<KdtreeNode>( references[0][end] );
   }
   else if (end == start + 1) {
      // Two references were passed to this function in sorted order, so store the start element at this level of
      // the tree and store the end element as the > child.
      node = std::make_shared<KdtreeNode>( references[0][start] );
      node->RightChild = std::make_shared<KdtreeNode>( references[0][end] );
   }
   else if (end == start + 2) {
      // Three references were passed to this function in sorted order, so store the median element at this level of
      // the tree, store the start element as the < child and store the end element as the > child.
      node = std::make_shared<KdtreeNode>( references[0][start + 1] );
      node->LeftChild = std::make_shared<KdtreeNode>( references[0][start] );
      node->RightChild = std::make_shared<KdtreeNode>( references[0][end] );
	}
   else if (end > start + 2) {
      // More than three references were passed to this function, so the median element of references[0] is chosen
      // as the tuple about which the other reference arrays will be partitioned.
      // Avoid overflow when computing the median.
      const int median = start + (end - start) / 2;

      // Store the median element of references[0] in a new kdNode.
      node = std::make_shared<KdtreeNode>( references[0][median] );

      // Copy references[0] to the temporary array before partitioning.
      for (int i = start; i <= end; ++i) buffer[i] = references[0][i];

      // Process each of the other reference arrays in a priori sorted order and partition it by comparing super keys.
      // Store the result from references[i] in references[i-1], thus permuting the reference arrays.
      // Skip the element of references[i] that that references a point that equals the point that is stored
      // in the new k-d node.
      int lower, upper, lowerSave, upperSave;
      for (int i = 1; i < dim; ++i) {
         // Process one reference array. Compare once only.
         lower = start - 1;
         upper = median;
         for (int j = start; j <= end; ++j) {
            T compare = compareSuperKey( references[i][j], node->Tuple, axis );
            if (compare < 0) references[i - 1][++lower] = references[i][j];
            else if (compare > 0) references[i - 1][++upper] = references[i][j];
         }

         // Check the new indices for the reference array.
         if (lower < start || lower >= median) {
            std::cout << "incorrect range for lower at depth = " << depth << " : start = " << start <<
               "  lower = " << lower << "  median = " << median << "\n";
            std::exit( 1 );
         }
         if (upper <= median || upper > end) {
            std::cout << "incorrect range for upper at depth = " << depth << " : median = " << median <<
               "  upper = " << upper << "  end = " << end << "\n";
            std::exit( 1 );
         }
         if (i > 1 && lower != lowerSave) {
            std::cout << " lower = " << lower << "  !=  lowerSave = " << lowerSave << "\n";
            std::exit( 1 );
         }
         if (i > 1 && upper != upperSave) {
            std::cout << " upper = " << upper << "  !=  upperSave = " << upperSave << "\n";
            std::exit( 1 );
         }

         lowerSave = lower;
         upperSave = upper;
      }

      // Copy the temporary array to references[dim - 1] to finish permutation.
      for (int i = start; i <= end; ++i) references[dim - 1][i] = buffer[i];

      // Recursively build the < branch of the tree.
      node->LeftChild = build( references, buffer, start, lower, depth + 1 );

      // Recursively build the > branch of the tree.
      node->RightChild = build( references, buffer, median + 1, upper, depth + 1 );

	}
   else if (end < start) {
      // This is an illegal condition that should never occur, so test for it last.
      std::cout << "error has occurred at depth = " << depth << " : end = " << end
         << "  <  start = " << start << "\n";
      std::exit( 1 );
	}

	// Return the pointer to the root of the k-d tree.
	return node;
}

/*
 * Walk the k-d tree and check that the children of a node are in the correct branch of that node.
 */
template<typename T, int dim>
int Kdtree<T, dim>::verify(KdtreeNode* node, int depth) const
{
   int count = 1;
	if (node->Tuple == nullptr) {
	    std::cout << "point is null!\n";
	    std::exit( 1 );
	}

	// The partition cycles as x, y, z, w...
	const int axis = depth % dim;
   if (node->LeftChild != nullptr) {
      if (node->LeftChild->Tuple[axis] > node->Tuple[axis]) {
         std::cout << "child is > node!\n";
         std::exit( 1 );
      }
      if (compareSuperKey( node->LeftChild->Tuple, node->Tuple, axis ) >= 0) {
         std::cout << "child is >= node!\n";
         std::exit( 1 );
      }
      count += verify( node->LeftChild.get(), depth + 1 );
	}
	if (node->RightChild != nullptr) {
      if (node->RightChild->Tuple[axis] < node->Tuple[axis]) {
         std::cout << "child is < node!\n";
         std::exit( 1 );
      }
      if (compareSuperKey( node->RightChild->Tuple, node->Tuple, axis ) <= 0) {
         std::cout << "child is <= node\n";
         std::exit( 1 );
      }
      count += verify( node->RightChild.get(), depth + 1 );
	}
	return count;
}

template<typename T, int dim>
void Kdtree<T, dim>::create(std::vector<const T*>& coordinates, int max_submit_depth)
{
   const T*** references = new T**[dim + 1];
   references[0] = coordinates.data();
   const auto size = static_cast<int>(coordinates.size());
   for (int i = 1; i <= dim; ++i) references[i] = new T*[size];


   auto start_time = std::chrono::system_clock::now();
   sortReferenceAscending( references[0], references[dim], 0, size - 1, 0, max_submit_depth, 0 );
   auto end_time = std::chrono::system_clock::now();
   const auto sort_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() * 1e-9;

   start_time = std::chrono::system_clock::now();
   const int end = removeDuplicates( references[0], 0, size );
   end_time = std::chrono::system_clock::now();
   const auto remove_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() * 1e-9;

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

   Root = build( references, buffer, 0, end[0], 0 );
   end_time = std::chrono::system_clock::now();
   const auto build_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() * 1e-9;

   NodeNum = verify( Root.get(), 0 );
   std::cout << "\n>> Number of nodes = " << NodeNum << "\n";
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