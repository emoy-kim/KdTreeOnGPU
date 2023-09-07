#pragma once

#ifdef USE_CUDA
#include <iostream>
#include <iomanip>
#include <cassert>
#include <sstream>
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#ifdef NDEBUG
#define CHECK_CUDA(cu_result) (static_cast<void>(0))
#define CHECK_KERNEL
#else
#define CHECK_CUDA(cu_result) \
do { \
   if ((cu_result) == cudaSuccess) ; \
   else { \
      std::ostringstream buffer; \
      buffer << "CUDA ERROR CODE: " << (cu_result) << "\n"; \
      throw std::runtime_error( buffer.str() ); \
   } \
} while(0)
#define CHECK_KERNEL \
do { \
   if ((cudaGetLastError()) == cudaSuccess) ; \
   else { \
      std::ostringstream buffer; \
      buffer << "CUDA KERNEL ERROR CODE: " << (cudaGetLastError()) << "\n"; \
      throw std::runtime_error( buffer.str() ); \
   } \
} while(0)
#endif

using node_type = float;

namespace cuda
{
   constexpr int WarpSize = 32;
   constexpr int ThreadNum = 512;
   constexpr int ThreadBlockNum = 32;
   constexpr int SampleStride = 128;
   constexpr int SharedSize = WarpSize * WarpSize;

   struct KdtreeNode
   {
      int Index;
      int LeftChildIndex;
      int RightChildIndex;

      KdtreeNode() : Index( -1 ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
      explicit KdtreeNode(int index) : Index( index ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
   };

   class KdtreeCUDA final
   {
   public:
      explicit KdtreeCUDA(const node_type* vertices, int size, int dim);
      ~KdtreeCUDA();

      void print() const;
      void getResult(std::vector<node_type>& output) const;

   private:
      struct SortGPU
      {
         int MaxSampleNum;
         int* LeftRanks;
         int* RightRanks;
         int* LeftLimits;
         int* RightLimits;
         int* Reference;
         node_type* Buffer;

         SortGPU() :
            MaxSampleNum( 0 ), LeftRanks( nullptr ), RightRanks( nullptr ), LeftLimits( nullptr ),
            RightLimits( nullptr ), Reference( nullptr ), Buffer( nullptr ) {}
      };

      struct CUDADevice
      {
         int ID;
         int TupleNum;
         int RootNode;
         SortGPU Sort;
         KdtreeNode* Root;
         cudaStream_t Stream;
         std::vector<int*> Reference;
         std::vector<node_type*> Buffer;
         node_type* CoordinatesDevicePtr;
         int* LeftChildNumInWarp;
         int* RightChildNumInWarp;
         int* NodeSums;
         std::array<int*, 2> MidReferences;

         CUDADevice() :
            ID( -1 ), TupleNum( 0 ), RootNode( -1 ), Sort(), Root( nullptr ), Stream( nullptr ),
            CoordinatesDevicePtr( nullptr ), LeftChildNumInWarp( nullptr ), RightChildNumInWarp( nullptr ),
            NodeSums( nullptr ), MidReferences{} {}
      };

      const node_type* const Coordinates;
      const int Dim;
      int TupleNum;
      int NodeNum;
      int RootNode;
      CUDADevice Device;

      void prepareCUDA();
      void initialize(const node_type* coordinates, int size);
      void initializeReference(int axis);
      void sortByAxis(int axis);
      [[nodiscard]] int removeDuplicates(int axis) const;
      void sort(std::vector<int>& end);
      void partitionDimension(int axis, int depth);
      void partitionDimensionFinal(int axis, int depth);
      void build();
      [[nodiscard]] int verify(int start_axis) const;
      [[nodiscard]] int verify();
      void create();
      void print(const std::vector<KdtreeNode>& kd_nodes, int index, int depth) const;
      void getResult(
         std::vector<node_type>& output,
         const std::vector<KdtreeNode>& kd_nodes,
         int index,
         int depth
      ) const;
   };
}
#endif