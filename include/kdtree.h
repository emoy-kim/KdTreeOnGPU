#pragma once

#include "base.h"

template<typename T = float, int dim = 3>
class Kdtree final
{
public:
   struct KdtreeNode
   {
      const T* Tuple;
      std::shared_ptr<KdtreeNode> LeftChild;
      std::shared_ptr<KdtreeNode> RightChild;

      explicit KdtreeNode(const T* tuple) : Tuple( tuple ) {}
   };

   using E = std::pair<T, const KdtreeNode*>;
   using TVec = glm::vec<dim, T, glm::defaultp>;

   struct Finder
   {
      int CurrentDepth;
      int RequestedDepth;
      std::priority_queue<E> Heap;

      explicit Finder(int requested_depth) : CurrentDepth( 0 ), RequestedDepth( requested_depth ) {}

      [[nodiscard]] bool isFull() const { return CurrentDepth >= RequestedDepth; }
      [[nodiscard]] T getMaxSquaredDistance() const { return Heap.empty() ? 0 : Heap.top().first; }
      void add(const KdtreeNode* node, const TVec& query)
      {
         T squared_distance = node->Tuple[0] - query[0];
         squared_distance *= squared_distance;
         for (int i = 1; i < dim; ++i) {
            const T x = node->Tuple[i] - query[i];
            squared_distance += x * x;
         }
         if (!isFull()) {
            Heap.push( std::make_pair( squared_distance, node ) );
            CurrentDepth++;
         }
         else if (squared_distance < getMaxSquaredDistance()) {
            Heap.pop();
            Heap.push( std::make_pair( squared_distance, node ) );
         }
      }
   };

   explicit Kdtree(const std::vector<TVec>& vertices, int thread_num = 8);
   ~Kdtree() = default;

   void print(std::vector<T>& output) const { if (Root != nullptr) print( output, Root.get(), 0 ); }
   [[nodiscard]] std::list<const KdtreeNode*> search(const TVec& query, T search_radius) const
   {
      std::list<const KdtreeNode*> found;
      if (Root != nullptr) search( found, Root.get(), query - search_radius, query + search_radius, Permutation, 0 );
      return found;
   }
   [[nodiscard]] std::forward_list<E> findNearestNeighbors(const TVec& query, int neighbor_num) const
   {
      std::forward_list<E> found;
      Finder finder(neighbor_num);
      if (Root != nullptr) findNearestNeighbors( finder, Root.get(), query, 0 );
      for (; !finder.Heap.empty(); finder.Heap.pop()) found.emplace_front( finder.Heap.top() );
      return found;
   }

private:
   inline static constexpr int InsertionSortThreshold = 15;
   inline static int MaxThreadNum = 0;
   inline static int MaxSubmitDepth = -1;

   int NodeNum;
   std::vector<int> Permutation;
   std::shared_ptr<KdtreeNode> Root;

   [[nodiscard]] static bool isThreadAvailable(int depth) { return MaxSubmitDepth >= 0 && MaxSubmitDepth >= depth; }
   [[nodiscard]] static T compareSuperKey(const T* const a, const T* const b, int axis)
   {
      T difference = a[axis] - b[axis];
      for (int i = 1; difference == 0 && i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim; // A fast alternative to the modulus operator for (i + axis) < 2 * dim.
         difference = a[r] - b[r];
      }
      return difference;
   }
   [[nodiscard]] int verify(const KdtreeNode* node, int depth) const;
   void create(std::vector<const T*>& coordinates);
   [[nodiscard]] static bool isInside(
      const KdtreeNode* node,
      const TVec& lower,
      const TVec& upper
   );
   static void prepareMultiThreading(int thread_num);
   static void sortReferenceAscending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int depth
   );
   static void sortReferenceDescending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int depth
   );
   static void sortBufferAscending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int depth
   );
   static void sortBufferDescending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int depth
   );
   static int removeDuplicates(const T** reference, int size);
   static std::shared_ptr<KdtreeNode> build(
      const T*** references,
      const std::vector<std::vector<int>>& permutation,
      int start,
      int end,
      int depth
   );
   static void createPermutation(std::vector<int>& permutation, int coordinates_num);
   static void search(
      std::list<const KdtreeNode*>& found,
      const KdtreeNode* node,
      const TVec& lower,
      const TVec& upper,
      const std::vector<int>& permutation,
      int depth
   );
   void findNearestNeighbors(
      Finder& finder,
      const KdtreeNode* node,
      const TVec& query,
      int depth
   ) const;
   void print(std::vector<T>& output, KdtreeNode* node, int depth) const;
};

template class Kdtree<float, 3>;