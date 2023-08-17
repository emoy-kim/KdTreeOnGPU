#pragma once

#include "base.h"

template<typename T = float, int dim = 3>
class Kdtree final
{
public:
   using TVec = glm::vec<dim, T, glm::defaultp>;

   struct KdtreeNode
   {
      const T* Tuple;
      std::shared_ptr<KdtreeNode> LeftChild;
      std::shared_ptr<KdtreeNode> RightChild;

      explicit KdtreeNode(const T* tuple) : Tuple( tuple ) {}
   };

   explicit Kdtree(const std::vector<TVec>& vertices);

   void create(std::vector<const T*>& coordinates, int max_submit_depth);
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
   std::shared_ptr<KdtreeNode> Root;

   [[nodiscard]] static T compareSuperKey(const T* const a, const T* const b, int axis)
   {
      T difference = a[axis] - b[axis];
      for (int i = 1; difference == 0 && i < dim; ++i) {
         int r = i + axis;
         r = (r < dim) ? r : r - dim; // A fast alternative to the modulus operator for (i + axis) < 2 * dim.
         difference = a[r] - b[r];
      }
      return difference;
   }
   [[nodiscard]] int verify(KdtreeNode* node, int depth) const;
   [[nodiscard]] static std::list<KdtreeNode*> search(KdtreeNode* node, const TVec& query, T radius, int depth);
   static void sort(std::vector<const T*>& reference, std::vector<const T*>& buffer, int low, int high, int axis);
   static void sortReferenceAscending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortReferenceDescending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortBufferAscending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static void sortBufferDescending(
      const T** reference,
      const T** buffer,
      int low,
      int high,
      int axis,
      int max_submit_depth,
      int depth
   );
   static int removeDuplicates(const T** reference, int leading_dim_for_super_key, int size);
   static std::shared_ptr<KdtreeNode> build(
      const T*** references,
      std::vector<const T*>& buffer,
      int start,
      int end,
      int depth
   );
   void print(KdtreeNode* node, int depth) const;
};

template class Kdtree<float, 3>;