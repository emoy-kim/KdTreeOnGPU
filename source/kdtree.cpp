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
int Kdtree<T, dim>::removeDuplicates(std::vector<const T*>& reference, int leading_dim_for_super_key)
{
   int end = 0;
   const auto size = static_cast<int>(reference.size());
	for (int j = 1; j < size; ++j) {
      const T compare = compareSuperKey( reference[j], reference[j - 1], leading_dim_for_super_key );
      if (compare < 0) {
         std::cout << "sort failure: compareSuperKey( reference[" << j << "], reference[" << j - 1 << "], ("
            << leading_dim_for_super_key << ") = " << compare  << "\n";
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
void Kdtree<T, dim>::create(std::vector<const T*>& coordinates)
{
   std::vector<const T*> buffer(coordinates.size());
   std::vector<std::vector<const T*>> references(dim, std::vector<const T*>(coordinates.size()));
   for (int i = 0; i < dim; ++i) {
      references[i] = coordinates;
      sort( references[i], buffer, 0, static_cast<int>(references[i].size() - 1), i );
   }

   // Remove references to duplicate coordinates via one pass through each reference array.
   std::vector<int> end;
   for (int i = 0; i < dim; ++i) end.emplace_back( removeDuplicates( references[i], i ) );

   // Check that the same number of references was removed from each reference array.
   for (int i = 0; i < dim - 1; ++i) {
      for (int j = i + 1; j < dim; ++j) {
         if (end[i] != end[j]) {
            std::cout << "reference removal error\n";
            std::exit( 1 );
         }
      }
   }

   Root = build( references, buffer, 0, end[0], 0 );

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