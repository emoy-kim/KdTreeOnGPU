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

typedef float node_type;

namespace cuda
{
   constexpr int WarpSize = 32;
   constexpr int ThreadNum = 512;
   constexpr int ThreadBlockNum = 32;
   constexpr int SampleStride = 128;
   constexpr int SharedSizeLimit = 1024;
   constexpr int MergePathBlockSize = 512;

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

   private:
      inline static int DeviceNum = 0;

      struct SortGPU
      {
         int MaxSampleNum;
         int* RanksA;
         int* RanksB;
         int* LimitsA;
         int* LimitsB;
         int* MergePath;
         int* Reference;
         node_type* Buffer;

         SortGPU() :
            MaxSampleNum( 0 ), RanksA( nullptr ), RanksB( nullptr ), LimitsA( nullptr ), LimitsB( nullptr ),
            MergePath( nullptr ), Reference( nullptr ), Buffer( nullptr ) {}
      };

      struct Device
      {
         int ID;
         int TupleNum;
         int RootNode;
         SortGPU Sort;
         KdtreeNode* Root;
         cudaStream_t Stream;
         cudaEvent_t SyncEvent;
         std::vector<int*> Reference;
         std::vector<node_type*> Buffer;
         node_type* CoordinatesDevicePtr;
         int* LeftSegmentLengths;
         int* RightSegmentLengths;
         int* NodeSums;
         std::array<int*, 2> MidReferences;

         Device() :
            ID( -1 ), TupleNum( 0 ), RootNode( -1 ), Sort(), Root( nullptr ), Stream( nullptr ), SyncEvent( nullptr ),
            CoordinatesDevicePtr( nullptr ), LeftSegmentLengths( nullptr ), RightSegmentLengths( nullptr ),
            NodeSums( nullptr ), MidReferences{} {}
      };

      const node_type* const Coordinates;
      const int Dim;
      int TupleNum;
      int NodeNum;
      int RootNode;
      std::vector<Device> Devices;

      static void setDevice(int device_id) { CHECK_CUDA( cudaSetDevice( device_id ) ); }
      void sync() const { for (auto& device : Devices) CHECK_CUDA( cudaStreamSynchronize( device.Stream ) ); }
      [[nodiscard]] static bool isP2PCapable(const cudaDeviceProp& properties)
      {
         return properties.major >= 2; // Only boards based on Fermi can support P2P
      }
      void prepareCUDA();
      void initialize(Device& device, const node_type* coordinates, int size);
      void initializeReference(Device& device, int size, int axis) const;
      void sortPartially(Device& device, int source_index, int target_index, int size, int axis) const;
      [[nodiscard]] int swapBalanced(int source_index, int start_offset, int size, int axis);
      void mergeSwap(Device& device, int source_index, int target_index, int merge_point, int size) const;
      [[nodiscard]] int removeDuplicates(
         Device& device,
         int source_index,
         int target_index,
         int size,
         int axis,
         Device* other_device = nullptr,
         int other_size = 0
      ) const;
      void fillUp(Device& device, int size) const;
      static void copyReferenceAndBuffer(Device& device, int source_index, int target_index, int size);
      static void copyReference(Device& device, int source_index, int target_index, int size);
      void sort(std::vector<int>& end);
      void partitionDimension(Device& device, int axis, int depth) const;
      void partitionDimensionFinal(Device& device, int axis, int depth) const;
      void build();
      [[nodiscard]] int verify(Device& device, int start_axis) const;
      [[nodiscard]] int verify();
      void create();
      void print(const std::vector<KdtreeNode>& kd_nodes, int index, int depth) const;
   };
}
#endif