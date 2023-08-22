#include "cuda/kdtree.cuh"

namespace cuda
{
   template<typename T, int dim>
   KdtreeCUDA<T, dim>::KdtreeCUDA(const std::vector<TVec>& vertices) : NodeNum( 0 ), Root( nullptr )
   {
      if (DeviceNum == 0) prepareCUDA();
      create( vertices );
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

      Streams.resize( DeviceNum );
      SyncEvents.resize( DeviceNum );
      CoordinatesDevicePtr.resize( DeviceNum );
      for (int i = 0; i < DeviceNum; ++i) {
         DeviceID.emplace_back( i );

         setDevice( i );
         CHECK_CUDA( cudaStreamCreate( &Streams[i] ) );
         CHECK_CUDA( cudaEventCreate( &SyncEvents[i] ) );
      }
   }

   template<typename T, int dim>
   void KdtreeCUDA<T, dim>::initialize(const T* coordinates, int size, int device_id)
   {
      if (CoordinatesDevicePtr[device_id] != nullptr) {
         throw std::runtime_error( "coordinates device ptr already allocated!" );
      }
      if (Root != nullptr) throw std::runtime_error( "k-d tree already allocated!" );

      setDevice( device_id );
      CHECK_CUDA(
         cudaMalloc( reinterpret_cast<void**>(&CoordinatesDevicePtr[device_id]), sizeof( T ) * dim * (size + 1) )
      );
      CHECK_CUDA(
         cudaMemcpyAsync(
            CoordinatesDevicePtr[device_id], coordinates, sizeof( T ) * dim * size,
            cudaMemcpyHostToDevice, Streams[device_id]
         )
      );

      const TVec max_value(std::numeric_limits<T>::max());
      CHECK_CUDA(
         cudaMemcpyAsync(
            CoordinatesDevicePtr[device_id] + size * dim, glm::value_ptr( max_value ), sizeof( T ) * dim,
            cudaMemcpyHostToDevice, Streams[device_id]
         )
      );

      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Root), sizeof( KdtreeNode ) * size ) );

      //cuda::initialize( Root, CoordinatesDevicePtr[device_id], size, ThreadNum * ThreadBlockNum );
   }

   template<typename T, int dim>
   void KdtreeCUDA<T, dim>::create(const std::vector<TVec>& coordinates)
   {
      const auto size = static_cast<int>(coordinates.size());
      for (int i = 0; i < DeviceNum; ++i) {
         const T* ptr = glm::value_ptr( coordinates[0] ) + i * dim * size / DeviceNum;
         initialize( ptr, size / DeviceNum, i );
      }
   }
}