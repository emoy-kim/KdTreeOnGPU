#include "cuda/kdtree.cuh"

namespace cuda
{
   KdtreeCUDA::KdtreeCUDA(const node_type* vertices, int size, int dim) : Dim( dim ), NodeNum( 0 )
   {
      if (DeviceNum == 0) prepareCUDA();
      create( vertices, size );
   }

   void KdtreeCUDA::prepareCUDA()
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

      Buffers.resize( DeviceNum );
      References.resize( DeviceNum );
      Sort.resize( DeviceNum );
      Root.resize( DeviceNum );
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

   __global__
   void cuInitialize(KdtreeNode* root, int size)
   {
      auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         root[i].Index = i;
         root[i].LeftChildIndex = -1;
         root[i].RightChildIndex = -1;
      }
   }

   void KdtreeCUDA::initialize(const node_type* coordinates, int size, int device_id)
   {
      if (CoordinatesDevicePtr[device_id] != nullptr) {
         throw std::runtime_error( "coordinates device ptr already allocated!" );
      }
      if (Root[device_id] != nullptr) throw std::runtime_error( "k-d tree already allocated!" );

      setDevice( device_id );
      CHECK_CUDA(
         cudaMalloc(
            reinterpret_cast<void**>(&CoordinatesDevicePtr[device_id]),
            sizeof( node_type ) * Dim * (size + 1)
         )
      );
      CHECK_CUDA(
         cudaMemcpyAsync(
            CoordinatesDevicePtr[device_id], coordinates, sizeof( node_type ) * Dim * size,
            cudaMemcpyHostToDevice, Streams[device_id]
         )
      );

      node_type max_value[Dim];
      for (int i = 0; i < Dim; ++i) max_value[i] = std::numeric_limits<node_type>::max();
      CHECK_CUDA(
         cudaMemcpyAsync(
            CoordinatesDevicePtr[device_id] + size * Dim, max_value, sizeof( node_type ) * Dim,
            cudaMemcpyHostToDevice, Streams[device_id]
         )
      );

      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Root[device_id]), sizeof( KdtreeNode ) * size ) );

      cuInitialize<<<ThreadBlockNum, ThreadNum, 0, Streams[device_id]>>>( Root[device_id], size );
   }

   __global__
   void cuInitializeReference(int* reference, int size)
   {
      auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         reference[i] = i;
      }
   }

   void KdtreeCUDA::initializeReference(int size, int axis, int device_id)
   {
      setDevice( device_id );
      int** references = References[device_id];
      for (int i = 0; i <= Dim + 1; ++i) {
         if (references[i] == nullptr) {
            CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&references[i]), sizeof( int ) * size ) );
         }
      }
      cuInitializeReference<<<ThreadBlockNum, ThreadNum, 0, Streams[device_id]>>>( references[axis], size );
   }

   void KdtreeCUDA::sort(int* end, int size)
   {
      const int max_sample_num = size / SampleStride + 1;
      for (int i = 0; i < DeviceNum; ++i) {
         setDevice( i );
         Sort[i].MaxSampleNum = max_sample_num;
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].RanksA), sizeof( uint ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].RanksB), sizeof( uint ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].LimitsA), sizeof( uint ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].LimitsB), sizeof( uint ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].Reference), sizeof( int ) * size ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].Buffer), sizeof( int ) * size ) );
      }

      const int size_per_device = size / DeviceNum;
      if (DeviceNum > 1) {
         for (int i = 0; i < DeviceNum; ++i) {
            initializeReference( size_per_device, 0, i );

         }
      }
      else {

      }
      sync();
   }

   void KdtreeCUDA::create(const node_type* coordinates, int size)
   {
      const int size_per_device = size / DeviceNum;
      for (int i = 0; i < DeviceNum; ++i) {
         const node_type* ptr = coordinates + i * Dim * size_per_device;
         initialize( ptr, size_per_device, i );
      }
      cudaDeviceSynchronize();

      int end[Dim];
      sort( end, size );
   }
}