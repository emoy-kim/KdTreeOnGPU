#pragma once

#include <iostream>
#include <cassert>
#include <sstream>
#include <array>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(cu_result) \
do { \
   if ((cu_result) == cudaSuccess) ; \
   else { \
      std::ostringstream buffer; \
      buffer << "CUDA ERROR CODE: " << (cu_result) << "\n"; \
      throw std::runtime_error( buffer.str() ); \
   } \
} while(0)

typedef float node_type;

namespace cuda
{
   constexpr int ThreadNum = 512;
   constexpr int ThreadBlockNum = 32;
   constexpr int SampleStride = 128;
   constexpr uint SharedSizeLimit = 1024;

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

      void create(const node_type* coordinates, int size);

   private:
      inline static int DeviceNum = 0;

      struct SortGPU
      {
         uint MaxSampleNum;
         uint* RanksA;
         uint* RanksB;
         uint* LimitsA;
         uint* LimitsB;
         int* Reference;
         node_type* Buffer;

         SortGPU() :
            MaxSampleNum( 0 ), RanksA( nullptr ), RanksB( nullptr ), LimitsA( nullptr ), LimitsB( nullptr ),
            Reference( nullptr ), Buffer( nullptr ) {}
      };

      const int Dim;
      int NodeNum;
      std::vector<int> DeviceID;
      std::vector<SortGPU> Sort;
      std::vector<KdtreeNode*> Root;
      std::vector<cudaStream_t> Streams;
      std::vector<cudaEvent_t> SyncEvents;
      std::vector<int**> References;
      std::vector<node_type**> Buffers;
      std::vector<node_type*> CoordinatesDevicePtr;

      static void setDevice(int device_id) { CHECK_CUDA( cudaSetDevice( device_id ) ); }
      void sync() const { for (int i = 0; i < DeviceNum; ++i) CHECK_CUDA( cudaStreamSynchronize( Streams[i] ) ); }
      [[nodiscard]] static bool isP2PCapable(const cudaDeviceProp& properties)
      {
         return properties.major >= 2; // Only boards based on Fermi can support P2P
      }
      void prepareCUDA();
      void initialize(const node_type* coordinates, int size, int device_id);
      void initializeReference(int size, int axis, int device_id);
      void sortPartially(int source_index, int target_index, int start_offset, int size, int axis, int device_id);
      void sort(int* end, int size);
   };
}