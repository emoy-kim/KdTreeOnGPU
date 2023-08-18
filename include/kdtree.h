#pragma once

#include "base.h"

template<typename T = float, int dim = 3>
class Kdtree final
{
public:
   using TVec = glm::vec<dim, T, glm::defaultp>;

   struct KdtreeNode
   {
      T* const Tuple;
      std::shared_ptr<KdtreeNode> LeftChild;
      std::shared_ptr<KdtreeNode> RightChild;

      explicit KdtreeNode(T* const tuple) : Tuple( tuple ) {}
   };

   explicit Kdtree(std::vector<TVec>& vertices, int thread_num = 4);

   void create(std::vector<T*>& coordinates);
   void print() const
   {
      if (Root != nullptr) print( Root.get(), 0 );
   }
   [[nodiscard]] std::list<KdtreeNode*> search(const TVec& query, T radius)
   {
      return Root != nullptr ? search( Root.get(), query, radius, 0 ) : std::list<KdtreeNode*>{};
   }

private:
   inline static constexpr int InsertionSortThreshold = 15;

   int NodeNum;
   int MaxThreadNum;
   int MaxSubmitDepth;
   std::shared_ptr<KdtreeNode> Root;

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
   [[nodiscard]] int verify(
      KdtreeNode* node,
      const std::vector<int>& permutation,
      int max_submit_depth,
      int depth
   ) const;
   [[nodiscard]] static std::list<KdtreeNode*> search(KdtreeNode* node, const TVec& query, T radius, int depth);
   void prepareMultiThreading(int thread_num);
   static void sortReferenceAscending(
      T** const reference,
      T** const buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortReferenceDescending(
      T** const reference,
      T** const buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortBufferAscending(
      T** const reference,
      T** const buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortBufferDescending(
      T** const reference,
      T** const buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static int removeDuplicates(T** const reference, int leading_dim_for_super_key, int size);
   static std::shared_ptr<KdtreeNode> build(
      T*** const references,
      const std::vector<std::vector<int>>& permutation,
      int start,
      int end,
      int max_submit_depth,
      int depth
   );
   static void createPermutation(std::vector<int>& permutation, int coordinates_num);
   void print(KdtreeNode* node, int depth) const;
};

template class Kdtree<float, 3>;