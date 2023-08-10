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
void Kdtree<T, dim>::sort(
   std::vector<const T*>& reference,
   std::vector<const T*>& temporary,
   int low,
   int high,
   int partition
)
{
   if (low < high) {
      const int mid = low + (high - low) / 2;
      sort( reference, temporary, low, mid, partition );
      sort( reference, temporary, mid + 1, high, partition );

      int i, j;
      for (i = mid + 1; i > low; --i) temporary[i - 1] = reference[i - 1];
      for (j = mid; j < high; ++j) temporary[mid + (high - j)] = reference[j + 1];
      for (int k = low; k <= high; ++k) {
         reference[k] = compareSuperKey( temporary[i], temporary[j], partition ) < 0 ? temporary[i++] : temporary[j--];
      }
   }
}

template<typename T, int dim>
int Kdtree<T, dim>::removeDuplicates(std::vector<const T*>& reference, int leading_dim_for_super_key)
{
   int end = 0;
   const auto size = static_cast<int>(reference.size());
	for (int j = 1; j < size; ++j) {
      const T compare = compareSuperKey( reference[j], reference[j - 1], leading_dim_for_super_key );
      if (compare < 0) {
         std::cout << "sort failure: compareSuperKey( reference[" << j << "], reference[" << j - 1 << "], ("
            << leading_dim_for_super_key << ") = " << compare  << std::endl;
		   std::exit( 1 );
      }
      else if (compare > 0) reference[++end] = reference[j];
	}
	return end;
}

/*
 * This function builds a k-d tree by recursively partitioning the reference arrays and adding kdNodes to the tree.
 * These arrays are permuted cyclically for successive levels of the tree in order that sorting occur on x, y, z, w...
 */
template<typename T, int dim>
std::shared_ptr<typename Kdtree<T, dim>::KdtreeNode> Kdtree<T, dim>::build(
   std::vector<std::vector<const T*>>& references,
   std::vector<const T*>& temporary,
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
      for (int i = start; i <= end; ++i) temporary[i] = references[0][i];

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
               "  lower = " << lower << "  median = " << median << std::endl;
            std::exit( 1 );
         }
         if (upper <= median || upper > end) {
            std::cout << "incorrect range for upper at depth = " << depth << " : median = " << median <<
               "  upper = " << upper << "  end = " << end << std::endl;
            std::exit( 1 );
         }
         if (i > 1 && lower != lowerSave) {
            std::cout << " lower = " << lower << "  !=  lowerSave = " << lowerSave << std::endl;
            std::exit( 1 );
         }
         if (i > 1 && upper != upperSave) {
            std::cout << " upper = " << upper << "  !=  upperSave = " << upperSave << std::endl;
            std::exit( 1 );
         }

         lowerSave = lower;
         upperSave = upper;
      }

      // Copy the temporary array to references[dim - 1] to finish permutation.
      for (int i = start; i <= end; ++i) references[dim - 1][i] = temporary[i];

      // Recursively build the < branch of the tree.
      node->LeftChild = build( references, temporary, start, lower, depth + 1 );

      // Recursively build the > branch of the tree.
      node->RightChild = build( references, temporary, median + 1, upper, depth + 1 );

	}
   else if (end < start) {
      // This is an illegal condition that should never occur, so test for it last.
      std::cout << "error has occurred at depth = " << depth << " : end = " << end
         << "  <  start = " << start << std::endl;
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
	    std::cout << "point is null!" << std::endl;
	    std::exit( 1 );
	}

	// The partition cycles as x, y, z, w...
	const int axis = depth % dim;
   if (node->LeftChild != nullptr) {
      if (node->LeftChild->Tuple[axis] > node->Tuple[axis]) {
         std::cout << "child is > node!" << std::endl;
         std::exit( 1 );
      }
      if (compareSuperKey( node->LeftChild->Tuple, node->Tuple, axis ) >= 0) {
         std::cout << "child is >= node!" << std::endl;
         std::exit( 1 );
      }
      count += verify( node->LeftChild.get(), depth + 1 );
	}
	if (node->RightChild != nullptr) {
      if (node->RightChild->Tuple[axis] < node->Tuple[axis]) {
         std::cout << "child is < node!" << std::endl;
         std::exit( 1 );
      }
      if (compareSuperKey( node->RightChild->Tuple, node->Tuple, axis ) <= 0) {
         std::cout << "child is <= node" << std::endl;
         std::exit( 1 );
      }
      count += verify( node->RightChild.get(), depth + 1 );
	}
	return count;
}

template<typename T, int dim>
void Kdtree<T, dim>::create(std::vector<const T*>& coordinates)
{
   // Initialize and sort the reference arrays.
   std::vector<const T*> temporary(coordinates.size());
   std::vector<std::vector<const T*>> references(dim, std::vector<const T*>(coordinates.size()));
   for (int i = 0; i < dim; ++i) {
      references[i] = coordinates;
      sort( references[i], temporary, 0, static_cast<int>(references[i].size() - 1), i );
   }

   // Remove references to duplicate coordinates via one pass through each reference array.
   std::vector<int> end;
   for (int i = 0; i < dim; ++i) end.emplace_back( removeDuplicates( references[i], i ) );

   // Check that the same number of references was removed from each reference array.
   for (int i = 0; i < dim - 1; ++i) {
      for (int j = i + 1; j < dim; ++j) {
         if (end[i] != end[j]) {
            std::cout << "reference removal error" << std::endl;
            std::exit( 1 );
         }
      }
   }

   // Build the k-d tree.
   Root = build( references, temporary, 0, end[0], 0 );

   // Verify the k-d tree and report the number of KdNodes.
   NodeNum = verify( Root.get(), 0 );
   std::cout << std::endl << "Number of nodes = " << NodeNum << std::endl;
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
   std::cout << tuple[dim - 1] << ")";
	std::cout << std::endl;

	if (node->LeftChild != nullptr) print( node->LeftChild.get(), depth + 1 );
}

#if 0
int Kdtree::partition(
   std::vector<const glm::vec3*>& vertices,
   int vertex_num,
   int axis,
   int min_index,
   int max_index
)
{
   assert( min_index <= max_index );
   assert( 0 <= min_index && min_index < vertex_num );
   assert( 0 <= max_index && max_index < vertex_num );

   if (min_index == max_index) return min_index;

   int split_index = min_index;
   const int middle_index = (min_index + max_index) / 2;
   const auto r = min_index + static_cast<int>(getRandomValue( 0.0f, 1.0f ) * static_cast<float>(max_index - min_index));
   const float split_point = (*vertices[r])[axis];
   std::swap( vertices[r], vertices[max_index] );
   for (int i = min_index; i < max_index; ++i) {
      assert( split_index <= i );

      const float point = (*vertices[i])[axis];
      if (point < split_point || (point == split_point && split_index < middle_index)) {
         std::swap( vertices[split_index], vertices[i] );
         split_index++;
      }
   }
   std::swap( vertices[split_index], vertices[max_index] );

   const int half = vertex_num / 2;
   if (split_index == min_index) {
      return min_index >= half ? min_index : partition( vertices, vertex_num, axis, min_index + 1, max_index );
   }
   else if (split_index == max_index) {
      return max_index <= half ? max_index : partition( vertices, vertex_num, axis, min_index, max_index - 1 );
   }
   else if (split_index == half) return split_index;
   else if (split_index > half) return partition( vertices, vertex_num, axis, min_index, split_index - 1 );
   else return partition( vertices, vertex_num, axis, split_index + 1, max_index );
}

void Kdtree::insert(KdtreeNode* node, std::vector<const glm::vec3*>& vertices, const Box& box, int vertex_num)
{
   assert( node != nullptr );
   assert( node->PointNum == 0 );
   assert( node->Children[0] == nullptr && node->Children[1] == nullptr );

   if (vertex_num <= MaxPointsPerNode) {
      for (int i = 0; i < vertex_num; ++i) {
         node->Points[node->PointNum] = vertices[i];
         node->PointNum++;
      }
   }
   else {
      node->SplitDimension = box.getDominantAxis();
      const int split_index = partition( vertices, vertex_num, node->SplitDimension, 0, vertex_num - 1 );

      assert( 0 <= split_index && split_index < vertex_num );

      node->SplitCoordinate = (*vertices[split_index])[node->SplitDimension];

      assert( box.getMinPoint()[node->SplitDimension] <= node->SplitCoordinate );
      assert( node->SplitCoordinate <= box.getMaxPoint()[node->SplitDimension] );

      Box left_box(box), right_box(box);
      left_box.setMaxPoint( node->SplitCoordinate, node->SplitDimension );
      right_box.setMinPoint( node->SplitCoordinate, node->SplitDimension );

      node->Children[0] = std::make_shared<KdtreeNode>( node );
      node->Children[1] = std::make_shared<KdtreeNode>( node );

      insert( node->Children[0].get(), vertices, left_box, split_index );
      insert( node->Children[1].get(), vertices, right_box, vertex_num - split_index );
      NodeNum += 2;
   }
}
#endif