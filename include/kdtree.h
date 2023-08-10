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

   void create(std::vector<const T*>& coordinates);
   void print() const
   {
      if (Root != nullptr) print( Root.get(), 0 );
   }
   [[nodiscard]] std::list<KdtreeNode*> search(const TVec& query, T radius)
   {
      return Root != nullptr ? search( Root.get(), query, radius, 0 ) : std::list<KdtreeNode*>{};
   }

private:
   int NodeNum;
   std::shared_ptr<KdtreeNode> Root;

   [[nodiscard]] static T compareSuperKey(const T* a, const T* b, int p)
   {
      T difference = static_cast<T>(0);
      for (int i = 0; i < dim; ++i) {
          int r = i + p;
          r = (r < dim) ? r : r - dim; // A fast alternative to the modulus operator for (i + p) < 2 * dim.
          difference = a[r] - b[r];
          if (difference != static_cast<T>(0)) break;
      }
      return difference;
   }
   [[nodiscard]] int verify(KdtreeNode* node, int depth) const;
   [[nodiscard]] static std::list<KdtreeNode*> search(KdtreeNode* node, const TVec& query, T radius, int depth);
   static void sort(
      std::vector<const T*>& reference,
      std::vector<const T*>& temporary,
      int low,
      int high,
      int partition
   );
   static int removeDuplicates(std::vector<const T*>& reference, int leading_dim_for_super_key);
   static std::shared_ptr<KdtreeNode> build(
      std::vector<std::vector<const T*>>& references,
      std::vector<const T*>& temporary,
      int start,
      int end,
      int depth
   );
   void print(KdtreeNode* node, int depth) const;
};

template class Kdtree<float, 3>;