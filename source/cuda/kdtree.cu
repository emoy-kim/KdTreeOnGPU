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

      int gpu_num = 0;
      std::array<int, 2> gpu_id{};
      cudaDeviceProp properties[2];
      for (int i = 0; i < DeviceNum; ++i) {
         CHECK_CUDA( cudaGetDeviceProperties( &properties[i], i ) );
         if (isP2PCapable( properties[i] )) gpu_id[gpu_num++] = i;
      }

      if (gpu_num == 2) {
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

      Sort.resize( DeviceNum );
      Root.resize( DeviceNum );
      Streams.resize( DeviceNum );
      SyncEvents.resize( DeviceNum );
      References.resize( DeviceNum );
      Buffers.resize( DeviceNum );
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

   __global__
   void cuCopyCoordinates(
      node_type* target,
      const node_type* coordinates,
      const int* reference,
      int size,
      int axis,
      int dim
   )
   {
      auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         target[i] = coordinates[reference[i] * dim + axis];
      }
   }

   __device__
   node_type compareSuperKey(
      node_type front_a,
      node_type front_b,
      const node_type* a,
      const node_type* b,
      int axis,
      int dim
   )
   {
      node_type difference = front_a - front_b;
      for (int i = 1; difference == 0 && i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         difference = a[r] - b[r];
      }
      return difference;
   }

   __device__
   uint searchExclusively(
      const node_type* coordinates,
      const node_type* buffer,
      const int* reference,
      node_type value,
      int index,
      uint length,
      uint step,
      int axis,
      int dim
   )
   {
      if (length == 0) return 0;

      uint i = 0;
      while (step > 0) {
         uint j = umin( i + step, length );
         const node_type t = compareSuperKey(
            buffer[j - 1], value, coordinates + reference[j - 1] * dim, coordinates + index * dim, axis, dim
         );
         if (t < 0) i = j;
         step >>= 1;
      }
      return i;
   }

   __global__
   void cuSort(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      int axis,
      int dim
   )
   {
      __shared__ int reference[SharedSizeLimit];
      __shared__ node_type buffer[SharedSizeLimit];

      source_buffer += blockIdx.x * SharedSizeLimit + threadIdx.x;
      source_reference += blockIdx.x * SharedSizeLimit + threadIdx.x;
      target_buffer += blockIdx.x * SharedSizeLimit + threadIdx.x;
      target_reference += blockIdx.x * SharedSizeLimit + threadIdx.x;
      buffer[threadIdx.x] = source_buffer[0];
      reference[threadIdx.x] = source_reference[0];
      buffer[threadIdx.x + SharedSizeLimit / 2] = source_buffer[SharedSizeLimit / 2];
      reference[threadIdx.x + SharedSizeLimit / 2] = source_reference[SharedSizeLimit / 2];

      for (uint step = 1; step < SharedSizeLimit; step <<= 1) {
         const uint i = threadIdx.x & (step - 1);
         node_type* base_buffer = buffer + 2 * (threadIdx.x - i);
         int* base_reference = reference + 2 * (threadIdx.x - i);

         __syncthreads();
         const node_type buffer_x = base_buffer[i];
         const int reference_x = base_reference[i];
         const node_type buffer_y = base_buffer[i + step];
         const int reference_y = base_reference[i + step];
         const uint x = searchExclusively(
            coordinates, base_buffer + step, base_reference + step, buffer_x, reference_x, step, step, axis, dim
         ) + i;
         const uint y = searchExclusively(
            coordinates, base_buffer, base_reference, buffer_y, reference_y, step, step, axis, dim
         ) + i;

         __syncthreads();
         base_buffer[x] = buffer_x;
         base_buffer[y] = buffer_y;
         base_reference[x] = reference_x;
         base_reference[y] = reference_y;
      }

      __syncthreads();
      target_buffer[0] = buffer[threadIdx.x];
      target_reference[0] = reference[threadIdx.x];
      target_buffer[SharedSizeLimit / 2] = buffer[threadIdx.x + SharedSizeLimit / 2];
      target_reference[SharedSizeLimit / 2] = reference[threadIdx.x + SharedSizeLimit / 2];
   }

   void KdtreeCUDA::sortPartially(
      int source_index,
      int target_index,
      int start_offset,
      int size,
      int axis,
      int device_id
   )
   {
      assert( CoordinatesDevicePtr[device_id] != nullptr );
      assert( References[device_id][source_index] != nullptr && References[device_id][target_index] != nullptr );

      setDevice( device_id );
      if (Buffers[device_id][source_index] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&Buffers[device_id][source_index]), sizeof( node_type ) * size )
         );
         cuCopyCoordinates<<<ThreadBlockNum, ThreadNum, 0, Streams[device_id]>>>(
            Buffers[device_id][source_index],
            CoordinatesDevicePtr[device_id] + start_offset * Dim,
            References[device_id][source_index],
            size,
            axis,
            Dim
         );
      }
      if (Buffers[device_id][target_index] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&Buffers[device_id][target_index]), sizeof( node_type ) * size )
         );
      }

      uint stage_num = 0;
      for (uint step = SharedSizeLimit; step < size; step <<= 1) stage_num++;

      int* in_reference = nullptr;
      int* out_reference = nullptr;
      node_type* in_buffer = nullptr;
      node_type* out_buffer = nullptr;
      if (stage_num & 1) {
         in_buffer = Sort[device_id].Buffer;
         in_reference = Sort[device_id].Reference;
         out_buffer = Buffers[device_id][target_index];
         out_reference = References[device_id][target_index] + start_offset;
      }
      else {
         in_buffer = Buffers[device_id][target_index];
         in_reference = References[device_id][target_index] + start_offset;
         out_buffer = Sort[device_id].Buffer;
         out_reference = Sort[device_id].Reference;
      }

      assert( size <= SampleStride * Sort[device_id].MaxSampleNum );
      assert( size % SharedSizeLimit == 0 );

      const uint block_num = size / SharedSizeLimit;
      const uint thread_num = SharedSizeLimit / 2;
      cuSort<<<block_num, thread_num, 0, Streams[device_id]>>>(
         in_reference, in_buffer,
         References[device_id][source_index] + start_offset, Buffers[device_id][source_index],
         CoordinatesDevicePtr[device_id], axis, Dim
      );

      for (uint step = SharedSizeLimit; step < size; step <<= 1) {

      }
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
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Sort[i].Buffer), sizeof( node_type ) * size ) );
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