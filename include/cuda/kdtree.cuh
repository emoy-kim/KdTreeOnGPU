#pragma once

#include "base.h"
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

namespace cuda
{
   constexpr int ThreadNum = 512;
   constexpr int ThreadBlockNum = 32;

   template<typename T = float, int dim = 3>
   class KdtreeCUDA final
   {
   public:
      using TVec = glm::vec<dim, T, glm::defaultp>;

      struct KdtreeNode
      {
         int Index;
         int LeftChildIndex;
         int RightChildIndex;

         explicit KdtreeNode(int index) : Index( index ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
      };

      explicit KdtreeCUDA(const std::vector<TVec>& vertices);

      void create(const std::vector<TVec>& coordinates);

   private:
      inline static int DeviceNum = 0;

      int NodeNum;
      KdtreeNode* Root;
      std::vector<int> DeviceID;
      std::vector<cudaStream_t> Streams;
      std::vector<cudaEvent_t> SyncEvents;
      std::vector<T*> CoordinatesDevicePtr;

      [[nodiscard]] static bool isP2PCapable(const cudaDeviceProp& properties)
      {
         return properties.major >= 2; // Only boards based on Fermi can support P2P
      }
      void prepareCUDA();
      void setDevice(int device_id) { CHECK_CUDA( cudaSetDevice( device_id ) ); }
      void initialize(const T* coordinates, int size, int device_id);
   };

   template class KdtreeCUDA<float, 3>;
}