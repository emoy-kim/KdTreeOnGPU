#pragma once

#include <iostream>
#include <cassert>
#include <sstream>
#include <array>
#include <vector>
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

      explicit KdtreeNode(int index) : Index( index ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
   };

   class KdtreeCUDA final
   {
   public:
      explicit KdtreeCUDA(const node_type* vertices, int size, int dim);
      ~KdtreeCUDA();

      void create(const node_type* coordinates, int size);

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
         SortGPU Sort;
         KdtreeNode* Root;
         cudaStream_t Stream;
         cudaEvent_t SyncEvent;
         std::vector<int*> Reference;
         std::vector<node_type*> Buffer;
         node_type* CoordinatesDevicePtr;

         Device() :
            ID( -1 ), TupleNum( 0 ), Sort(), Root( nullptr ), Stream( nullptr ), SyncEvent( nullptr ),
            CoordinatesDevicePtr( nullptr ) {}
      };

      const int Dim;
      int NodeNum;
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
      void sortPartially(
         Device& device,
         int source_index,
         int target_index,
         int start_offset,
         int size,
         int axis
      ) const;
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
      void sort(std::vector<int>& end, int size);
   };
}