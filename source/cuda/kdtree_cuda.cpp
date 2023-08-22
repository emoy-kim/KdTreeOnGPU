#include "cuda/kdtree_cuda.h"

namespace cuda
{
   template<typename T, int dim>
   KdtreeCUDA<T, dim>::KdtreeCUDA(const std::vector<TVec>& vertices) : NodeNum( 0 )
   {
      std::vector<const float*> coordinates;
      coordinates.reserve( vertices.size() );
      for (const auto& v : vertices) coordinates.emplace_back( glm::value_ptr( v ) );

      if (DeviceNum == 0) prepareCUDA();
      create( coordinates );
   }

   template<typename T, int dim>
   void KdtreeCUDA<T, dim>::prepareCUDA()
   {
      CHECK_CUDA( cudaGetDeviceCount( &DeviceNum ) );
      DeviceNum = std::min( DeviceNum, 2 );

      int gpu_count = 0;
      std::array<int, 2> gpu_id{};
      cudaDeviceProp properties[2];
      for (int i = 0; i < DeviceNum; ++i) {
         CHECK_CUDA( cudaGetDeviceProperties( &properties[i], i ) );
         if (isP2PCapable( properties[i] )) gpu_id[gpu_count++] = i;
      }

      if (gpu_count == 2) {
         int can_access_peer_01, can_access_peer_10;
		   CHECK_CUDA( cudaDeviceCanAccessPeer( &can_access_peer_01, gpu_id[0], gpu_id[1] ) );
		   CHECK_CUDA( cudaDeviceCanAccessPeer( &can_access_peer_10, gpu_id[1], gpu_id[0] ) );
         if (can_access_peer_01 == 0 || can_access_peer_10 == 0) {
            CHECK_CUDA( cudaSetDevice( gpu_id[0] ) );
            DeviceNum = 1;
         }
         else {
            CHECK_CUDA( cudaSetDevice( gpu_id[0] ) );
            CHECK_CUDA( cudaDeviceEnablePeerAccess( gpu_id[1], 0 ) );
            CHECK_CUDA( cudaSetDevice( gpu_id[1] ) );
            CHECK_CUDA( cudaDeviceEnablePeerAccess( gpu_id[0], 0 ) );

            const bool has_uva = properties[gpu_id[0]].unifiedAddressing && properties[gpu_id[1]].unifiedAddressing;
            if (!has_uva) DeviceNum = 1;
         }
      }
      else DeviceNum = 1;

      for (int i = 0; i < DeviceNum; ++i) DeviceID.emplace_back( i );
   }

   template<typename T, int dim>
   void KdtreeCUDA<T, dim>::create(std::vector<const T*>& coordinates)
   {

   }
}