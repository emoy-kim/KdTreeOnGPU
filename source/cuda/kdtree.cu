#include "cuda/kdtree.cuh"

#ifdef USE_CUDA
namespace cuda
{
   __host__ __device__
   inline int divideUp(int a, int b)
   {
      return (a + b - 1) / b;
   }

   __host__ __device__
   inline int getSampleNum(int x)
   {
      return divideUp( x, SampleStride );
   }

   __device__
   static inline int getNextPowerOfTwo(int x)
   {
      constexpr int bits = sizeof( int ) * 8;
      return 1 << (bits - __clz( x - 1 ));
   }

   KdtreeCUDA::KdtreeCUDA(const node_type* vertices, int size, int dim) :
      Coordinates( vertices ), Dim( dim ), TupleNum( size ), NodeNum( 0 ), RootNode( -1 )
   {
      if (DeviceNum == 0) prepareCUDA();
      create();
   }

   KdtreeCUDA::~KdtreeCUDA()
   {
      for (auto& device : Devices) {
         setDevice( device.ID );
         if (!device.Reference.empty()) {
            for (int axis = 0; axis <= Dim; ++axis) {
               if (device.Reference[axis] != nullptr) cudaFree( device.Reference[axis] );
            }
         }
         if (!device.Buffer.empty()) {
            for (int axis = 0; axis <= Dim; ++axis) {
               if (device.Buffer[axis] != nullptr) cudaFree( device.Buffer[axis] );
            }
         }
         if (device.CoordinatesDevicePtr != nullptr) cudaFree( device.CoordinatesDevicePtr );
         if (device.Root != nullptr) cudaFree( device.Root );
         cudaEventDestroy( device.SyncEvent );
         cudaStreamDestroy( device.Stream );
      }
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

      Devices.resize( DeviceNum );
      for (int i = 0; i < DeviceNum; ++i) {
         Devices[i].ID = gpu_id[i];
         Devices[i].Buffer.resize( Dim + 2, nullptr );
         Devices[i].Reference.resize( Dim + 2, nullptr );

         setDevice( gpu_id[i] );
         CHECK_CUDA( cudaStreamCreate( &Devices[i].Stream ) );
         CHECK_CUDA( cudaEventCreate( &Devices[i].SyncEvent ) );
      }
   }

   __global__
   void cuInitialize(KdtreeNode* root, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         root[i].Index = i;
         root[i].LeftChildIndex = -1;
         root[i].RightChildIndex = -1;
      }
   }

   void KdtreeCUDA::initialize(Device& device, const node_type* coordinates, int size)
   {
      assert( device.Root == nullptr );
      assert( device.CoordinatesDevicePtr == nullptr );

      setDevice( device.ID );
      CHECK_CUDA(
         cudaMalloc(
            reinterpret_cast<void**>(&device.CoordinatesDevicePtr),
            sizeof( node_type ) * Dim * (size + 1)
         )
      );
      CHECK_CUDA(
         cudaMemcpyAsync(
            device.CoordinatesDevicePtr, coordinates, sizeof( node_type ) * Dim * size,
            cudaMemcpyHostToDevice, device.Stream
         )
      );

      node_type max_value[Dim];
      for (int i = 0; i < Dim; ++i) max_value[i] = std::numeric_limits<node_type>::max();
      CHECK_CUDA(
         cudaMemcpyAsync(
            device.CoordinatesDevicePtr + size * Dim, max_value, sizeof( node_type ) * Dim,
            cudaMemcpyHostToDevice, device.Stream
         )
      );

      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Root), sizeof( KdtreeNode ) * size ) );

      cuInitialize<<<ThreadBlockNum, ThreadNum, 0, device.Stream>>>( device.Root, size );
   }

   __global__
   void cuInitializeReference(int* reference, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         reference[i] = i;
      }
   }

   void KdtreeCUDA::initializeReference(Device& device, int size, int axis) const
   {
      setDevice( device.ID );
      std::vector<int*>& references = device.Reference;
      for (int i = 0; i <= Dim + 1; ++i) {
         if (references[i] == nullptr) {
            CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&references[i]), sizeof( int ) * size ) );
         }
      }
      cuInitializeReference<<<ThreadBlockNum, ThreadNum, 0, device.Stream>>>( references[axis], size );
      CHECK_KERNEL;
   }

   __global__
   void cuCopyCoordinates(
      node_type* target,
      const int* reference,
      const node_type* coordinates,
      int size,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         target[i] = coordinates[reference[i] * dim + axis];
      }
   }

   __device__
   node_type compareSuperKey(const node_type* a, const node_type* b, int axis, int dim)
   {
      node_type difference = 0;
      for (int i = 0; i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         difference = a[r] - b[r];
         if (difference != 0) break;
      }
      return difference;
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
   int search(
      int r,
      node_type v,
      const int* reference,
      const node_type* buffer,
      const node_type* coordinates,
      int length,
      int step,
      int axis,
      int dim,
      bool inclusive
   )
   {
      if (length == 0) return 0;

      int i = 0;
      while (step > 0) {
         const int j = min( i + step, length );
         const node_type t = compareSuperKey(
            buffer[j - 1], v, coordinates + reference[j - 1] * dim, coordinates + r * dim, axis, dim
         );
         if (t < 0 || (inclusive && t == 0)) i = j;
         step >>= 1;
      }
      return i;
   }

   __global__
   void cuSortByBlock(
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

      // Given S = SharedSizeLimit, for all threads, [base[i], base[i+step]] is
      // step 1: [0, 1] ... [S-2, S-1]
      // step 2: [0, 2] [1, 3] ... [S-4, S-2], [S-3, S-1]
      // step 4: [0, 4] [1, 5] [2, 6] [3, 7] ... [S-8, S-4] [S-7, S-3] [S-6, S-2] [S-5, S-1]
      //   ...
      // step S/2: [0, S/2] ... [S/2-1, S-1]
      for (int step = 1; step < SharedSizeLimit; step <<= 1) {
         const int i = static_cast<int>(threadIdx.x) & (step - 1);
         int* reference_base = reference + (threadIdx.x - i) * 2;
         node_type* buffer_base = buffer + (threadIdx.x - i) * 2;

         // Merge the sorted array A, base[0] ~ base[step-1], and B, base[step] ~ base[step*2-1]
         __syncthreads();
         const int reference_x = reference_base[i];
         const node_type buffer_x = buffer_base[i];
         const int reference_y = reference_base[i + step];
         const node_type buffer_y = buffer_base[i + step];
         const int x = search(
            reference_x, buffer_x, reference_base + step, buffer_base + step, coordinates, step, step, axis, dim, false
         ) + i;
         const int y = search(
            reference_y, buffer_y, reference_base, buffer_base, coordinates, step, step, axis, dim, true
         ) + i;

         __syncthreads();
         buffer_base[x] = buffer_x;
         buffer_base[y] = buffer_y;
         reference_base[x] = reference_x;
         reference_base[y] = reference_y;
      }

      __syncthreads();
      target_buffer[0] = buffer[threadIdx.x];
      target_reference[0] = reference[threadIdx.x];
      target_buffer[SharedSizeLimit / 2] = buffer[threadIdx.x + SharedSizeLimit / 2];
      target_reference[SharedSizeLimit / 2] = reference[threadIdx.x + SharedSizeLimit / 2];
   }

   __global__
   void cuGenerateSampleRanks(
      int* ranks_a,
      int* ranks_b,
      int* reference,
      node_type* buffer,
      const node_type* coordinates,
      int sorted_size,
      int size,
      int axis,
      int dim,
      int total_thread_num
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= total_thread_num) return;

      const int i = index & (sorted_size / SampleStride - 1);
      const int segment_base = (index - i) * SampleStride * 2;
      buffer += segment_base;
      reference += segment_base;
      ranks_a += segment_base / SampleStride;
      ranks_b += segment_base / SampleStride;

      const int element_a = sorted_size;
      const int element_b = min( sorted_size, size - (segment_base + sorted_size) );
      const int sample_a = getSampleNum( element_a );
      const int sample_b = getSampleNum( element_b );
      if (i < sample_a) {
         ranks_a[i] = i * SampleStride;
         ranks_b[i] = search(
            reference[i * SampleStride], buffer[i * SampleStride],
            reference + sorted_size, buffer + sorted_size, coordinates,
            element_b, getNextPowerOfTwo( element_b ), axis, dim, false
         );
      }
      if (i < sample_b) {
         ranks_b[sorted_size / SampleStride + i] = i * SampleStride;
         ranks_a[sorted_size / SampleStride + i] = search(
            reference[i * SampleStride + sorted_size], buffer[i * SampleStride + sorted_size],
            reference, buffer, coordinates,
            element_a, getNextPowerOfTwo( element_a ), axis, dim, true
         );
      }
   }

   __global__
   void cuMergeRanksAndIndices(int* limits, const int* ranks, int step, int size, int total_thread_num)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= total_thread_num) return;

      const int i = index & (step / SampleStride - 1);
      const int segment_base = (index - i) * 2 * SampleStride;
      ranks += (index - i) * 2;
      limits += (index - i) * 2;

      const int element_a = step;
      const int element_b = min( step, size - step - segment_base );
      const int sample_a = getSampleNum( element_a );
      const int sample_b = getSampleNum( element_b );
      if (i < sample_a) {
         int x = 0;
         if (sample_b != 0) {
            for (int s = getNextPowerOfTwo( sample_b ); s > 0; s >>= 1) {
               const int j = min( x + s, sample_b );
               if (ranks[sample_a + j - 1] < ranks[i]) x = j;
            }
         }
         limits[x + i] = ranks[i];
      }
      if (i < sample_b) {
         int x = 0;
         if (sample_a != 0) {
            for (int s = getNextPowerOfTwo( sample_a ); s > 0; s >>= 1) {
               const int j = min( x + s, sample_a );
               if (ranks[j - 1] <= ranks[sample_a + i]) x = j;
            }
         }
         limits[x + i] = ranks[sample_a + i];
      }
   }

   __device__
   void merge(
      int* reference,
      node_type* buffer,
      const node_type* coordinates,
      int length_a,
      int length_b,
      int axis,
      int dim
   )
   {
      const int* reference_a = reference;
      const int* reference_b = reference + SampleStride;
      const node_type* buffer_a = buffer;
      const node_type* buffer_b = buffer + SampleStride;

      int index_a, index_b, x, y;
      node_type value_a, value_b;
      if (threadIdx.x < length_a) {
         value_a = buffer_a[threadIdx.x];
         index_a = reference_a[threadIdx.x];
         x = static_cast<int>(threadIdx.x) +
            search( index_a, value_a, reference_b, buffer_b, coordinates, length_b, SampleStride, axis, dim, false );
      }
      if (threadIdx.x < length_b) {
         value_b = buffer_b[threadIdx.x];
         index_b = reference_b[threadIdx.x];
         y = static_cast<int>(threadIdx.x) +
            search( index_b, value_b, reference_a, buffer_a, coordinates, length_a, SampleStride, axis, dim, true );
      }

      __syncthreads();
      if (threadIdx.x < length_a) {
         buffer[x] = value_a;
         reference[x] = index_a;
      }
      if (threadIdx.x < length_b) {
         buffer[y] = value_b;
         reference[y] = index_b;
      }
   }

   __global__
   void cuMergeReferences(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      const int* limits_a,
      const int* limits_b,
      int step,
      int size,
      int axis,
      int dim
   )
   {
      const int index = static_cast<int>(blockIdx.x) & (2 * step / SampleStride - 1);
      const int segment_base = (static_cast<int>(blockIdx.x) - index) * SampleStride;
      target_buffer += segment_base;
      target_reference += segment_base;
      source_buffer += segment_base;
      source_reference += segment_base;

      __shared__ int reference[2 * SampleStride];
      __shared__ node_type buffer[2 * SampleStride];
      __shared__ int start_source_a, start_source_b;
      __shared__ int start_target_a, start_target_b;
      __shared__ int length_a, length_b;

      if (threadIdx.x == 0) {
         const int element_a = step;
         const int element_b = min( step, size - step - segment_base );
         const int sample_a = getSampleNum( element_a );
         const int sample_b = getSampleNum( element_b );
         const int sample_num = sample_a + sample_b;
         start_source_a = limits_a[blockIdx.x];
         start_source_b = limits_b[blockIdx.x];
         const int end_source_a = index + 1 < sample_num ? limits_a[blockIdx.x + 1] : element_a;
         const int end_source_b = index + 1 < sample_num ? limits_b[blockIdx.x + 1] : element_b;
         length_a = end_source_a - start_source_a;
         length_b = end_source_b - start_source_b;
         start_target_a = start_source_a + start_source_b;
         start_target_b = start_target_a + length_a;
      }
      __syncthreads();

      if (threadIdx.x < length_a) {
         buffer[threadIdx.x] = source_buffer[start_source_a + threadIdx.x];
         reference[threadIdx.x] = source_reference[start_source_a + threadIdx.x];
      }
      if (threadIdx.x < length_b) {
         buffer[threadIdx.x + SampleStride] = source_buffer[start_source_b + threadIdx.x + step];
         reference[threadIdx.x + SampleStride] = source_reference[start_source_b + threadIdx.x + step];
      }

      __syncthreads();
      merge( reference, buffer, coordinates, length_a, length_b, axis, dim );

      __syncthreads();
      if (threadIdx.x < length_a) {
         target_buffer[start_target_a + threadIdx.x] = buffer[threadIdx.x];
         target_reference[start_target_a + threadIdx.x] = reference[threadIdx.x];
      }
      if (threadIdx.x < length_b) {
         target_buffer[start_target_b + threadIdx.x] = buffer[length_a + threadIdx.x];
         target_reference[start_target_b + threadIdx.x] = reference[length_a + threadIdx.x];
      }
   }

   void KdtreeCUDA::sortByAxis(Device& device, int size, int axis) const
   {
      assert( device.CoordinatesDevicePtr != nullptr );
      assert( device.Reference[axis] != nullptr && device.Reference[Dim] != nullptr );

      setDevice( device.ID );
      if (device.Buffer[axis] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&device.Buffer[axis]), sizeof( node_type ) * size )
         );
         cuCopyCoordinates<<<ThreadBlockNum, ThreadNum, 0, device.Stream>>>(
            device.Buffer[axis], device.Reference[axis],
            device.CoordinatesDevicePtr, size, axis, Dim
         );
         CHECK_KERNEL;
      }
      if (device.Buffer[Dim] == nullptr) {
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Buffer[Dim]), sizeof( node_type ) * size ) );
      }

      int stage_num = 0;
      int* in_reference = nullptr;
      int* out_reference = nullptr;
      node_type* in_buffer = nullptr;
      node_type* out_buffer = nullptr;
      for (int step = SharedSizeLimit; step < size; step <<= 1) stage_num++;
      if (stage_num & 1) {
         in_buffer = device.Sort.Buffer;
         in_reference = device.Sort.Reference;
         out_buffer = device.Buffer[Dim];
         out_reference = device.Reference[Dim];
      }
      else {
         in_buffer = device.Buffer[Dim];
         in_reference = device.Reference[Dim];
         out_buffer = device.Sort.Buffer;
         out_reference = device.Sort.Reference;
      }

      assert( size <= SampleStride * device.Sort.MaxSampleNum );
      assert( size % SharedSizeLimit == 0 );

      cuSortByBlock<<<size / SharedSizeLimit, SharedSizeLimit / 2, 0, device.Stream>>>(
         in_reference, in_buffer,
         device.Reference[axis], device.Buffer[axis],
         device.CoordinatesDevicePtr, axis, Dim
      );
      CHECK_KERNEL;

      /*std::vector<node_type> data(size);
      CHECK_CUDA(
         cudaMemcpyAsync(
            data.data(), in_buffer, sizeof( node_type ) * size,
            cudaMemcpyDeviceToHost, device.Stream
         )
      );
      for (int i = 0; i < block_num; ++i) {
         const node_type* ptr = data.data() + i * SharedSizeLimit;
         for (int j = 0; j < SharedSizeLimit - 1; ++j) assert( ptr[j] <= ptr[j + 1] );
      }*/

      for (int sorted_size = SharedSizeLimit; sorted_size < size; sorted_size <<= 1) {
         constexpr int thread_num = SampleStride * 2;
         const int remained_threads = size % (sorted_size * 2);
         const int total_thread_num = remained_threads > sorted_size ?
            (size - remained_threads + sorted_size * 2) / thread_num : (size - remained_threads) / thread_num;
         const int block_num = divideUp( total_thread_num, thread_num );
         cuGenerateSampleRanks<<<block_num, thread_num, 0, device.Stream>>>(
            device.Sort.RanksA, device.Sort.RanksB,
            in_reference, in_buffer, device.CoordinatesDevicePtr,
            sorted_size, size, axis, Dim, total_thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<block_num, thread_num, 0, device.Stream>>>(
            device.Sort.LimitsA, device.Sort.RanksA, sorted_size, size, total_thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<block_num, thread_num, 0, device.Stream>>>(
            device.Sort.LimitsB, device.Sort.RanksB, sorted_size, size, total_thread_num
         );
         CHECK_KERNEL;

         const int merge_pairs = remained_threads > sorted_size ?
            getSampleNum( size ) : (size - remained_threads) / SampleStride;
         cuMergeReferences<<<merge_pairs, SampleStride, 0, device.Stream>>>(
            out_reference, out_buffer,
            in_reference, in_buffer, device.CoordinatesDevicePtr,
            device.Sort.LimitsA, device.Sort.LimitsB,
            sorted_size, size, axis, Dim
         );
         CHECK_KERNEL;

         if (remained_threads <= sorted_size) {
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_reference + size - remained_threads, in_reference + size - remained_threads,
                  sizeof( int ) * remained_threads, cudaMemcpyDeviceToDevice, device.Stream
               )
            );
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_buffer + size - remained_threads, in_buffer + size - remained_threads,
                  sizeof( node_type ) * remained_threads, cudaMemcpyDeviceToDevice, device.Stream
               )
            );
         }

         std::swap( in_reference, out_reference );
         std::swap( in_buffer, out_buffer );
      }
   }

   // reference: Comparison Based Sorting for Systems with Multiple GPUs, 2013
   __device__ int selected_pivot;

   __global__
   void cuSelectPivot(
      const int* reference_a,
      const node_type* buffer_a,
      const node_type* coordinates_a,
      const int* reference_b,
      const node_type* buffer_b,
      const node_type* coordinates_b,
      int length,
      int axis,
      int dim
   )
   {
      if (length == 0) {
         selected_pivot = 0;
         return;
      }

      int pivot = length / 2;
      for (int step = pivot / 2; step > 0; step >>= 1) {
         const node_type t = compareSuperKey(
            buffer_a[length - pivot - 1], buffer_b[pivot],
            coordinates_a + reference_a[length - pivot - 1] * dim, coordinates_b + reference_b[pivot] * dim,
            axis, dim
         );
         if (t < 0) {
            const node_type x = compareSuperKey(
               buffer_a[length - pivot - 2], buffer_b[pivot + 1],
               coordinates_a + reference_a[length - pivot - 2] * dim, coordinates_b + reference_b[pivot + 1] * dim,
               axis, dim
            );
            const node_type y = compareSuperKey(
               buffer_a[length - pivot], buffer_b[pivot - 1],
               coordinates_a + reference_a[length - pivot] * dim, coordinates_b + reference_b[pivot - 1] * dim,
               axis, dim
            );
            if (x < 0 && y > 0) {
               selected_pivot = pivot;
               return;
            }
            else pivot -= step;
         }
         else pivot += step;
      }

      if (pivot == 1 &&
          compareSuperKey(
            buffer_a[length - 1], buffer_b[0],
            coordinates_a + reference_a[length - 1] * dim, coordinates_b + reference_b[0] * dim,
            axis, dim
          ) < 0) {
         pivot = 0;
      }
      else if (pivot == length - 1 &&
               compareSuperKey(
                  buffer_a[0], buffer_b[length - 1],
                  coordinates_a + reference_a[0] * dim, coordinates_b + reference_b[length - 1] * dim,
                  axis, dim
               ) > 0) {
         pivot = length;
      }
      selected_pivot = pivot;
   }

   __global__
   void cuSwapPivot(
      node_type* coordinates_a,
      node_type* buffer_a,
      node_type* coordinates_b,
      node_type* buffer_b,
      const int* reference_a,
      const int* reference_b,
      int length,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      int pivot = selected_pivot;
      reference_a += length - pivot;
      buffer_a += length - pivot;
      for (int i = index; i < pivot * dim; i += step) {
         const int p = i / dim;
         const int swap = i - p * dim;
         const int index_a = reference_a[p] * dim + swap;
         const int index_b = reference_b[p] * dim + swap;
         const node_type a = coordinates_a[index_a];
         const node_type b = coordinates_b[index_b];
         coordinates_a[index_a] = b;
         coordinates_b[index_b] = a;
         if (swap == axis) {
            buffer_a[p] = b;
            buffer_b[p] = a;
         }
      }
   }

   int KdtreeCUDA::swapBalanced(int source_index, int start_offset, int size, int axis)
   {
      auto& d0 = Devices[0];
      auto& d1 = Devices[1];

      setDevice( d0.ID );
      cuSelectPivot<<<1, 1, 0, Devices[0].Stream>>>(
         d0.Reference[source_index] + start_offset, d0.Buffer[source_index], d0.CoordinatesDevicePtr,
         d1.Reference[source_index] + start_offset, d1.Buffer[source_index], d1.CoordinatesDevicePtr,
         size, axis, Dim
      );
      CHECK_KERNEL;

      int pivot;
      CHECK_CUDA(
         cudaMemcpyFromSymbolAsync( &pivot, selected_pivot, sizeof( pivot ), 0, cudaMemcpyDeviceToHost, d0.Stream )
      );

      cuSwapPivot<<<ThreadBlockNum * ThreadNum, 1024, 0, d0.Stream>>>(
         d0.CoordinatesDevicePtr, d0.Buffer[source_index],
         d1.CoordinatesDevicePtr, d1.Buffer[source_index],
         d0.Reference[source_index] + start_offset, d1.Reference[source_index] + start_offset,
         size, axis, Dim
      );
      CHECK_KERNEL;
      return pivot;
   }

   __device__
   int mergePath(
      const int* reference_a,
      const node_type* buffer_a,
      int count_a,
      const int* reference_b,
      const node_type* buffer_b,
      int count_b,
      const node_type* coordinates,
      int diagonal,
      int size,
      int axis,
      int dim
   )
   {
      int begin = max( 0, diagonal - count_b );
      int end = min( diagonal, count_a );
      while (begin < end) {
         const int mid = begin + (end - begin) / 2;
         const int index_a = reference_a[mid];
         const int index_b = reference_b[diagonal - 1 - mid];
         const node_type a = buffer_a[mid];
         const node_type b = buffer_b[diagonal - 1 - mid];
         const node_type t =
            compareSuperKey( a, b, coordinates + index_a * dim, coordinates + index_b * dim, axis, dim );
         if (t < 0) begin = mid + 1;
         else end = mid;
      }
      return begin;
   }

   __global__
   void cuMergePaths(
      int* merge_path,
      const int* reference_a,
      const node_type* buffer_a,
      int count_a,
      const int* reference_b,
      const node_type* buffer_b,
      int count_b,
      const node_type* coordinates,
      int size,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      const int partitions = divideUp( count_a + count_b, MergePathBlockSize );
      if (index == 0) {
         merge_path[0] = 0;
         merge_path[partitions] = count_a;
      }
      for (int i = index + 1; i < partitions; i += step) {
         merge_path[i] = mergePath(
            reference_a, buffer_a, count_a,
            reference_b, buffer_b, count_b,
            coordinates,
            i * MergePathBlockSize, size, axis, dim
         );
      }
   }

   __global__
   void cuMergePartitions(
      int* target_reference,
      node_type* target_buffer,
      const int* reference_a,
      const node_type* buffer_a,
      int count_a,
      const int* reference_b,
      const node_type* buffer_b,
      int count_b,
      const node_type* coordinates,
      const int* merge_path,
      int size,
      int axis,
      int dim
   )
   {
      __shared__ int in_reference[MergePathBlockSize];
      __shared__ int out_reference[MergePathBlockSize];
      __shared__ node_type in_buffer[MergePathBlockSize];
      __shared__ node_type out_buffer[MergePathBlockSize];

      int r;
      node_type v;
      const auto x = static_cast<int>(threadIdx.x);
      const int partitions = divideUp( size, MergePathBlockSize );
      for (int i = static_cast<int>(blockIdx.x); i < partitions; i += static_cast<int>(gridDim.x)) {
         const int grid = i * MergePathBlockSize;
         const int a0 = merge_path[i];
         const int a1 = merge_path[i + 1];
         const int b0 = grid - a0;
         const int b1 = min( count_a + count_b, grid + MergePathBlockSize ) - a1;
         const int j = x + grid;
         if (a0 == a1) {
            const int k = b0 + x;
            v = buffer_b[k];
            r = reference_b[k];
            target_buffer[j] = v;
            target_reference[j] = r;
         }
         else if (b0 == b1) {
            const int k = a0 + x;
            v = buffer_a[k];
            r = reference_a[k];
            target_buffer[j] = v;
            target_reference[j] = r;
         }
         else {
            const bool inclusive = x < a1 - a0;
            if (inclusive) {
               const int k = a0 + x;
               v = buffer_a[k];
               r = reference_a[k];
            }
            else {
               const int k = b0 + x - (a1 - a0);
               v = buffer_b[k];
               r = reference_b[k];
            }
            in_buffer[x] = v;
            in_reference[x] = r;
            __syncthreads();

            const int n = inclusive ? b1 - b0 : a1 - a0;
            const int offset = inclusive ? a1 - a0 : 0;
            const int t = search(
               r, v, in_reference + offset, in_buffer + offset,
               coordinates, n, getNextPowerOfTwo( n ), axis, dim, inclusive
            );
            const int k = inclusive ? t + x : t + x - (a1 - a0);
            out_buffer[k] = v;
            out_reference[k] = r;
            __syncthreads();

            target_buffer[j] = out_buffer[x];
            target_reference[j] = out_reference[x];
         }
      }
   }

   void KdtreeCUDA::mergeSwap(
      Device& device,
      int source_index,
      int target_index,
      int merge_point,
      int size
   ) const
   {
      setDevice( device.ID );

      const int count_a = merge_point;
      const int count_b = size - merge_point;
      const int partitions = divideUp( size, MergePathBlockSize );
      const int thread_num = divideUp( ThreadBlockNum * ThreadNum, MergePathBlockSize );
      CHECK_CUDA(
         cudaMalloc( reinterpret_cast<void**>(&device.Sort.MergePath), sizeof( int ) * (partitions + 1) )
      );
      cuMergePaths<<<thread_num, MergePathBlockSize, 0, device.Stream>>>(
         device.Sort.MergePath,
         device.Reference[source_index], device.Buffer[source_index], count_a,
         device.Reference[source_index] + count_a, device.Buffer[source_index] + count_a, count_b,
         device.CoordinatesDevicePtr, size, 0, Dim
      );
      CHECK_KERNEL;

      cuMergePartitions<<<thread_num, MergePathBlockSize, 0, device.Stream>>>(
         device.Reference[target_index], device.Buffer[target_index],
         device.Reference[source_index], device.Buffer[source_index], count_a,
         device.Reference[source_index] + count_a, device.Buffer[source_index] + count_a, count_b,
         device.CoordinatesDevicePtr, device.Sort.MergePath, size, 0, Dim
      );
      CHECK_KERNEL;
   }

   __device__ int num_after_removal;
   __device__ int removal_error;
   __device__ int removal_error_address;

   __global__
   void cuRemoveDuplicates(
      int* segment_lengths,
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      const int* other_reference,
      const node_type* other_coordinates,
      int segment_size,
      int size,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      const int warps_per_block = SharedSizeLimit / (2 * warpSize);
      const int warp_index = (index - thread_index) / warpSize;
      const int start_segment = warp_index * segment_size;
      if (start_segment + segment_size > size) segment_size = size - start_segment;

      __shared__ int reference[SharedSizeLimit];
      __shared__ node_type buffer[SharedSizeLimit];

      int* out_reference = target_reference + start_segment;
      node_type* out_buffer = target_buffer + start_segment;
      const int* in_reference = source_reference + start_segment;
      const node_type* in_buffer = source_buffer + start_segment;
      const int shared_base = 2 * warpSize * (warp_index % warps_per_block);
      const int shared_address_mask = 2 * warpSize - 1;
      const int mask = (1 << thread_index) - 1;

      node_type t, v;
      int i, r, count = 0;
      int shuffle_mask = 0;
      for (i = 0; i < segment_size && shuffle_mask == 0; i += warpSize) {
         if (thread_index < segment_size) {
            buffer[shared_base + thread_index] = v = in_buffer[thread_index];
            reference[shared_base + thread_index] = r = in_reference[thread_index];
            if (thread_index != 0) {
               t = compareSuperKey(
                  v, buffer[shared_base + thread_index - 1],
                  coordinates + r * dim, coordinates + reference[shared_base + thread_index - 1] * dim,
                  axis, dim
               );
            }
            else if (warp_index != 0) {
               t = compareSuperKey(
                  v, *(in_buffer - 1),
                  coordinates + r * dim, coordinates + *(in_reference - 1) * dim,
                  axis, dim
               );
            }
            else if (other_coordinates != nullptr) {
               t = compareSuperKey(
                  v, *(other_coordinates + (*other_reference) * dim),
                  coordinates + r * dim, other_coordinates + (*other_reference) * dim,
                  axis, dim
               );
            }
            else t = 1;
         }
         else t = 0;

         if (t < 0) {
            removal_error = -1;
            atomicMin( &removal_error_address, start_segment + thread_index );
         }
         in_buffer += warpSize;
         in_reference += warpSize;

         shuffle_mask = static_cast<int>(__ballot( t > 0 ));
         if (t > 0) {
            const int j = __popc( shuffle_mask & mask );
            buffer[shared_base + ((count + j) & shared_address_mask)] = v;
            reference[shared_base + ((count + j) & shared_address_mask)] = r;
         }
      }

      int old_count = count;
      count += __popc( shuffle_mask );
      if (((old_count ^ count) & warpSize) != 0) {
         out_buffer[(old_count & ~(warpSize - 1)) + thread_index] =
            buffer[shared_base + (old_count & warpSize) + thread_index];
         out_reference[(old_count & ~(warpSize - 1)) + thread_index] =
            reference[shared_base + (old_count & warpSize) + thread_index];
      }

      for (; i < segment_size; i += warpSize) {
         if (i + thread_index < segment_size) {
            buffer[shared_base + ((count + thread_index) & shared_address_mask)] = v = in_buffer[thread_index];
            reference[shared_base + ((count + thread_index) & shared_address_mask)] = r = in_reference[thread_index];
            t = compareSuperKey(
               v, buffer[shared_base + ((count + thread_index - 1) & shared_address_mask)],
               coordinates + r * dim,
               coordinates + reference[shared_base + ((count + thread_index - 1) & shared_address_mask)] * dim,
               axis, dim
            );
         }
         else t = 0;

         if (t < 0) {
            removal_error = -1;
            atomicMin( &removal_error_address, start_segment + thread_index );
         }
         in_buffer += warpSize;
         in_reference += warpSize;

         shuffle_mask = static_cast<int>(__ballot( t > 0 ));
         if (t > 0) {
            const int j = __popc( shuffle_mask & mask );
            buffer[shared_base + ((count + j) & shared_address_mask)] = v;
            reference[shared_base + ((count + j) & shared_address_mask)] = r;
         }

         old_count = count;
         count += __popc( shuffle_mask );
         if (((old_count ^ count) & warpSize) != 0) {
            out_buffer[(old_count & ~(warpSize - 1)) + thread_index] =
               buffer[shared_base + (old_count & warpSize) + thread_index];
            out_reference[(old_count & ~(warpSize - 1)) + thread_index] =
               reference[shared_base + (old_count & warpSize) + thread_index];
         }
      }

      if ((count & (warpSize - 1)) > thread_index) {
         out_buffer[(count & ~(warpSize - 1)) + thread_index] =
            buffer[shared_base + (count & warpSize) + thread_index];
         out_reference[(count & ~(warpSize - 1)) + thread_index] =
            reference[shared_base + (count & warpSize) + thread_index];
      }

      if (thread_index == 0 && segment_lengths != nullptr) segment_lengths[warp_index] = count;
   }

   __device__
   void copyWarp(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      int segment_size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      for (int i = thread_index; i < segment_size; i += warpSize) {
         target_buffer[i] = source_buffer[i];
         target_reference[i] = source_reference[i];
      }
   }

   __global__
   void cuRemoveGaps(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const int* segment_lengths,
      int segment_size,
      int size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      const int warp_index = (index - thread_index) / warpSize;
      const int in_start = warp_index * segment_size;
      int out_start = 0, length = 0;
      if (thread_index == 0) {
         for (int i = 0; i < warp_index; ++i) out_start += segment_lengths[i];
         length = segment_lengths[warp_index];
      }

      out_start = __shfl( out_start, 0 );
      length = __shfl( length, 0 );
      copyWarp(
         target_reference + out_start, target_buffer + out_start,
         source_reference + in_start, source_buffer + in_start, segment_size
      );

      if (thread_index == 0 && in_start + segment_size >= size) {
         num_after_removal = out_start + segment_lengths[warp_index];
      }
   }

   int KdtreeCUDA::removeDuplicates(
      Device& device,
      int source_index,
      int target_index,
      int size,
      int axis,
      Device* other_device,
      int other_size
   ) const
   {
      assert( device.Buffer[source_index] != nullptr && device.Buffer[target_index] != nullptr );
      assert( device.Reference[source_index] != nullptr && device.Reference[target_index] != nullptr );

      int error = 0;
      const int error_address = 0x7FFFFFFF;
      CHECK_CUDA(
         cudaMemcpyToSymbolAsync(
            removal_error, &error, sizeof( removal_error ), 0,
            cudaMemcpyHostToDevice, device.Stream
         )
      );
      CHECK_CUDA(
         cudaMemcpyToSymbolAsync(
            removal_error_address, &error_address, sizeof( removal_error_address ), 0,
            cudaMemcpyHostToDevice, device.Stream
         )
      );

      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSizeLimit, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSizeLimit / 2 );
      constexpr int segment = total_thread_num / WarpSize;
      const int segment_size = (size - 1 + segment) / segment;
      const int* other_reference = other_device == nullptr ?
         nullptr : other_device->Reference[source_index] + other_size - 1;
      const node_type* other_coordinates = other_device == nullptr ? nullptr : other_device->CoordinatesDevicePtr;

      int* segment_lengths;
      setDevice( device.ID );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&segment_lengths), sizeof( int ) * segment ) );
      cuRemoveDuplicates<<<block_num, thread_num_per_block, 0, device.Stream>>>(
         segment_lengths, device.Sort.Reference, device.Sort.Buffer,
         device.Reference[source_index], device.Buffer[source_index],
         device.CoordinatesDevicePtr, other_reference, other_coordinates,
         segment_size, size, axis, Dim
      );
      CHECK_KERNEL;

      cuRemoveGaps<<<block_num, thread_num_per_block, 0, device.Stream>>>(
         device.Reference[target_index], device.Buffer[target_index],
         device.Sort.Reference, device.Sort.Buffer,
         segment_lengths, segment_size, size
      );
      CHECK_KERNEL;

      CHECK_CUDA( cudaFree( segment_lengths ) );

      CHECK_CUDA(
         cudaMemcpyFromSymbolAsync( &error, removal_error, sizeof( error ), 0, cudaMemcpyDeviceToHost, device.Stream )
      );
      if (error != 0) {
         std::ostringstream buffer;
         buffer << "error in removeDuplicates(): " << error << "\n";
         throw std::runtime_error( buffer.str() );
      }

      int num;
      CHECK_CUDA(
         cudaMemcpyFromSymbolAsync( &num, num_after_removal, sizeof( num ), 0, cudaMemcpyDeviceToHost, device.Stream )
      );
      return num;
   }

   __global__
   void cuFillMemory(int* ptr, int value, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) ptr[i] = value;
   }

   void KdtreeCUDA::fillUp(Device& device, int size) const
   {
      if (device.TupleNum == size) return;

      assert( device.Reference[Dim] + device.TupleNum == nullptr );

      setDevice( device.ID );
      cuFillMemory<<<ThreadBlockNum, ThreadNum, 0, device.Stream>>>(
         device.Reference[Dim] + device.TupleNum, size, size - device.TupleNum
      );
      CHECK_KERNEL;
   }

   __global__
   void cuCopyReferenceAndBuffer(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      int segment_size,
      int size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      const int warp_index = (index - thread_index) / warpSize;
      const int start_segment = warp_index * segment_size;
      if (start_segment + segment_size > size) segment_size = size - start_segment;

      copyWarp(
         target_reference + start_segment, target_buffer + start_segment,
         source_reference + start_segment, source_buffer + start_segment, segment_size
      );
   }

   void KdtreeCUDA::copyReferenceAndBuffer(Device& device, int source_index, int target_index, int size)
   {
      assert( device.Reference[source_index] != nullptr && device.Buffer[source_index] != nullptr );

      if (device.Reference[target_index] == nullptr) {
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Reference[target_index]), sizeof( int ) * size ) );
      }
      if (device.Buffer[target_index] == nullptr) {
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Buffer[target_index]), sizeof( node_type ) * size ) );
      }

      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSizeLimit, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSizeLimit / 2 );
      constexpr int segment = total_thread_num / 32;
      const int segment_size = (size - 1 + segment) / segment;

      setDevice( device.ID );
      cuCopyReferenceAndBuffer<<<block_num, thread_num_per_block, 0, device.Stream>>>(
         device.Reference[target_index], device.Buffer[target_index],
         device.Reference[source_index], device.Buffer[source_index],
         segment_size, size
      );
      CHECK_KERNEL;
   }

   __global__
   void cuCopyReference(int* target_reference, const int* source_reference, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) target_reference[i] = source_reference[i];
   }

   void KdtreeCUDA::copyReference(Device& device, int source_index, int target_index, int size)
   {
      assert( device.Reference[source_index] != nullptr );

      if (device.Reference[target_index] == nullptr) {
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Reference[target_index]), sizeof( int ) * size ) );
      }

      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSizeLimit, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSizeLimit / 2 );

      setDevice( device.ID );
      cuCopyReference<<<block_num, thread_num_per_block, 0, device.Stream>>>(
         device.Reference[target_index], device.Reference[source_index], size
      );
      CHECK_KERNEL;
   }

   void KdtreeCUDA::sort(std::vector<int>& end)
   {
      const int max_sample_num = TupleNum / SampleStride + 1;
      for (auto& device : Devices) {
         setDevice( device.ID );
         device.Sort.MaxSampleNum = max_sample_num;
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.RanksA), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.RanksB), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.LimitsA), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.LimitsB), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.Reference), sizeof( int ) * TupleNum ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.Buffer), sizeof( node_type ) * TupleNum ) );
      }

      const int size_per_device = TupleNum / DeviceNum;
      if (DeviceNum > 1) {
         auto& d0 = Devices[0];
         auto& d1 = Devices[1];
         initializeReference( d0, size_per_device, 0 );
         sortByAxis( d0, size_per_device, 0 );
         initializeReference( d1, size_per_device, 0 );
         sortByAxis( d1, size_per_device, 0 );
         sync();

         const int pivot = swapBalanced( Dim, 0, size_per_device, 0 );
         std::cout << " >> Pivot = " << pivot << "\n";
         sync();

         mergeSwap( d0, Dim, 0, size_per_device - pivot, size_per_device );
         mergeSwap( d1, Dim, 0, pivot, size_per_device );
         sync();

         std::vector<std::vector<int>> ends(DeviceNum, std::vector<int>(Dim));
         d0.TupleNum = ends[0][0] = removeDuplicates( Devices[0], 0, Dim, size_per_device, 0 );
         fillUp( d0, size_per_device );
         copyReferenceAndBuffer( d0, Dim, 0, size_per_device );
         d1.TupleNum = ends[1][0] = removeDuplicates( d1, 0, Dim, size_per_device, 0, &Devices[0], size_per_device );
         fillUp( d1, size_per_device );
         copyReferenceAndBuffer( d1, Dim, 0, size_per_device );
         sync();

         setDevice( d0.ID );
         CHECK_CUDA(
            cudaMemcpyAsync(
               &RootNode, d0.Reference[0] + d0.TupleNum - 1, sizeof( int ), cudaMemcpyDeviceToHost, d0.Stream
            )
         );
         CHECK_CUDA(
            cudaMemcpyAsync(
               d0.Reference[0] + d0.TupleNum - 1, &size_per_device, sizeof( int ), cudaMemcpyHostToDevice, d0.Stream
            )
         );
         d0.TupleNum--;

         for (int axis = 1; axis < Dim; ++axis) {
            copyReference( Devices[0], 0, axis, size_per_device );
            sortByAxis( Devices[0], size_per_device, axis );
            ends[0][axis] = removeDuplicates( Devices[0], Dim, axis, size_per_device, axis );
         }
         for (int axis = 1; axis < Dim; ++axis) {
            copyReference( Devices[1], 0, axis, size_per_device );
            sortByAxis( Devices[1], size_per_device, axis );
            ends[1][axis] = removeDuplicates( Devices[1], Dim, axis, size_per_device, axis );
         }

         for (int axis = 0; axis < Dim; ++axis) {
            end[axis] = ends[0][axis] < 0 || ends[1][axis] < 0 ? -1 : ends[0][axis] + ends[1][axis];
         }
      }
      else {
         setDevice( Devices[0].ID );
         for (int axis = 0; axis < Dim; ++axis) {
            initializeReference( Devices[0], size_per_device, axis );
            sortByAxis( Devices[0], size_per_device, axis );
            end[axis] = removeDuplicates( Devices[0], Dim, axis, size_per_device, axis );
         }
         Devices[0].TupleNum = end[0];
      }
      sync();

      for (auto& device : Devices) {
         setDevice( device.ID );
         CHECK_CUDA( cudaStreamSynchronize( device.Stream ) );
         CHECK_CUDA( cudaFree( device.Sort.RanksA ) );
         CHECK_CUDA( cudaFree( device.Sort.RanksB ) );
         CHECK_CUDA( cudaFree( device.Sort.LimitsA ) );
         CHECK_CUDA( cudaFree( device.Sort.LimitsB ) );
         CHECK_CUDA( cudaFree( device.Sort.Reference ) );
         CHECK_CUDA( cudaFree( device.Sort.Buffer ) );
         if (device.Sort.MergePath != nullptr) CHECK_CUDA( cudaFree( device.Sort.MergePath ) );
         for (int axis = 0; axis <= Dim; ++axis) CHECK_CUDA( cudaFree( device.Buffer[axis] ) );
      }
   }

   __device__
   void partition(
      int* target_left_reference,
      int* target_right_reference,
      int* segment_left_lengths,
      int* segment_right_lengths,
      const int* source_reference,
      const node_type* __restrict__ coordinates,
      int mid_reference,
      int segment_size,
      int partition_size,
      int axis,
      int dim,
      int warp_group_size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      const int warps_per_block = SharedSizeLimit / (2 * warpSize);
      const int warp_index = ((index - thread_index) / warpSize) % warp_group_size;
      const int start_segment = warp_index * segment_size;
      if (start_segment + segment_size > partition_size) segment_size = partition_size - start_segment;

      __shared__ int left_reference[SharedSizeLimit];
      __shared__ int right_reference[SharedSizeLimit];

      const int* in_reference = source_reference + start_segment;
      int* out_left_reference = target_left_reference + start_segment;
      int* out_right_reference = target_right_reference + start_segment;
      const int shared_base = 2 * warpSize * (((index - thread_index) / warpSize) % warps_per_block);
      const int shared_address_mask = 2 * warpSize - 1;
      const int mask = (1 << thread_index) - 1;

      node_type t;
      int r, old_left_count, old_right_count, left_count = 0, right_count = 0;
      for (int i = 0; i < segment_size; i += warpSize) {
         if (i + thread_index < segment_size) {
            r = in_reference[thread_index];
            t = compareSuperKey(
               coordinates[r * dim + axis], coordinates[mid_reference * dim + axis],
               coordinates + r * dim, coordinates + mid_reference * dim, axis, dim
            );
         }
         else t = 0;
         in_reference += warpSize;

         int shuffle_mask = static_cast<int>(__ballot( t < 0 ));
         if (t < 0) {
            const int j = __popc( shuffle_mask & mask );
            left_reference[shared_base + ((left_count + j) & shared_address_mask)] = r;
         }

         old_left_count = left_count;
         left_count += __popc( shuffle_mask );
         if (((old_left_count ^ left_count) & warpSize) != 0) {
            out_left_reference[(old_left_count & ~(warpSize - 1)) + thread_index] =
               left_reference[shared_base + (old_left_count & warpSize) + thread_index];
         }

         shuffle_mask = static_cast<int>(__ballot( t > 0 ));
         if (t > 0) {
            const int j = __popc( shuffle_mask & mask );
            right_reference[shared_base + ((right_count + j) & shared_address_mask)] = r;
         }

         old_right_count = right_count;
         right_count += __popc( shuffle_mask );
         if (((old_right_count ^ right_count) & warpSize) != 0) {
            out_right_reference[(old_right_count & ~(warpSize - 1)) + thread_index] =
               right_reference[shared_base + (old_right_count & warpSize) + thread_index];
         }
      }

      if ((left_count & (warpSize - 1)) > thread_index) {
         out_left_reference[(left_count & ~(warpSize - 1)) + thread_index] =
            left_reference[shared_base + (left_count & warpSize) + thread_index];
      }
      if ((right_count & (warpSize - 1)) > thread_index) {
         out_right_reference[(right_count & ~(warpSize - 1)) + thread_index] =
            right_reference[shared_base + (right_count & warpSize) + thread_index];
      }

      if (thread_index == 0 && segment_left_lengths != nullptr) segment_left_lengths[warp_index] = left_count;
      if (thread_index == 0 && segment_right_lengths != nullptr) segment_right_lengths[warp_index] = right_count;
   }

   __global__
   void cuPartition(
      KdtreeNode* root,
      int* segment_left_lengths,
      int* segment_right_lengths,
      int* target_left_reference,
      int* target_right_reference,
      int* mid_references,
      const int* last_mid_references,
      const int* source_reference,
      const int* primary_reference,
      const node_type* __restrict__ coordinates,
      int start,
      int end,
      int axis,
      int dim,
      int depth
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto all_warps = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_group_size = all_warps >> depth;
      const int thread_index = index & (warpSize - 1);
      const int warp_index = (index - thread_index) / warpSize;

      int mid;
      for (int i = 0; i < depth; ++i) {
         mid = start + (end - start) / 2;
         if (warp_index & (all_warps >> (i + 1))) start = mid + 1;
         else end = mid - 1;
      }
      mid = start + (end - start) / 2;

      const int partition_size = end - start + 1;
      const int segment_size = (partition_size + warp_group_size - 1) / warp_group_size;
      const int mid_reference = primary_reference[mid];
      partition(
         target_left_reference + start, target_right_reference + start,
         segment_left_lengths + (warp_index & ~(warp_group_size - 1)),
         segment_right_lengths + (warp_index & ~(warp_group_size - 1)),
         source_reference + start, coordinates,
         mid_reference, segment_size, partition_size, axis, dim, warp_group_size
      );

      if (thread_index == 0) {
         const int m = warp_index / warp_group_size;
         mid_references[m] = mid_reference;
         if (last_mid_references != nullptr) {
            if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
            else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
         }
      }
   }

   __device__
   void copyReferenceWarp(int* target_reference, const int* source_reference, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      if (size < warpSize * 200) {
         for (int i = thread_index; i < size; i += warpSize) target_reference[i] = source_reference[i];
         return;
      }

      __shared__ int tag[SharedSizeLimit];
      __shared__ int reference[SharedSizeLimit];

      const int warp_index = (index - thread_index) / warpSize;
      const int warps_per_block = SharedSizeLimit / (2 * warpSize);
      const int shared_base = 2 * warpSize * (warp_index % warps_per_block);
      const int shared_address_mask = 2 * warpSize - 1;
      tag[shared_base + thread_index] = 0;
      tag[shared_base + thread_index + warpSize] = 0;

      int* ptr = reinterpret_cast<int*>(reinterpret_cast<ulong>(target_reference) & ~(sizeof( int ) * warpSize - 1));
      int count = static_cast<int>(reinterpret_cast<ulong>(target_reference) - reinterpret_cast<ulong>(ptr));
      target_reference = ptr;

      ptr = reinterpret_cast<int*>(reinterpret_cast<ulong>(source_reference) & ~(sizeof( int ) * warpSize - 1));
      int in_count = warpSize +
         static_cast<int>(reinterpret_cast<ulong>(ptr) - reinterpret_cast<ulong>(source_reference));

      int r;
      if (thread_index < in_count) {
         r = source_reference[thread_index];
         tag[shared_base + ((count + thread_index) & shared_address_mask)] = 1;
         reference[shared_base + ((count + thread_index) & shared_address_mask)] = r;
      }

      source_reference = ptr + warpSize;
      int old_count = count;
      count += in_count;
      if (((old_count ^ count) & warpSize) != 0) {
         if (tag[shared_base + (old_count & warpSize) + thread_index] == 1) {
            target_reference[(old_count & ~(warpSize - 1)) + thread_index] =
               reference[shared_base + (old_count & warpSize) + thread_index];
            tag[shared_base + (old_count & warpSize) + thread_index] = 0;
         }
      }
      else {
         r = source_reference[thread_index];
         tag[shared_base + ((count + thread_index) & shared_address_mask)] = 1;
         reference[shared_base + ((count + thread_index) & shared_address_mask)] = r;
         old_count = count;
         count += warpSize;
         if (((old_count ^ count) & warpSize) != 0) {
            if (tag[shared_base + (old_count & warpSize) + thread_index] == 1) {
               target_reference[(old_count & ~(warpSize - 1)) + thread_index] =
                  reference[shared_base + (old_count & warpSize) + thread_index];
               tag[shared_base + (old_count & warpSize) + thread_index] = 0;
            }
         }
         in_count += warpSize;
         source_reference += warpSize;
      }

      while (in_count < size) {
         if (in_count + thread_index < size) {
            r = source_reference[thread_index];
            tag[shared_base + ((count + thread_index) & shared_address_mask)] = 1;
            reference[shared_base + ((count + thread_index) & shared_address_mask)] = r;
         }
         old_count = count;
         count += in_count + warpSize <= size ? warpSize : size - in_count;
         if (((old_count ^ count) & warpSize) != 0) {
            if (tag[shared_base + (old_count & warpSize) + thread_index] == 1) {
               target_reference[(old_count & ~(warpSize - 1)) + thread_index] =
                  reference[shared_base + (old_count & warpSize) + thread_index];
               tag[shared_base + (old_count & warpSize) + thread_index] = 0;
            }
         }
         in_count += warpSize;
         source_reference += warpSize;
      }

      if (tag[shared_base + (count & warpSize) + thread_index] == 1) {
         target_reference[(count & ~(warpSize - 1)) + thread_index] =
            reference[shared_base + (count & warpSize) + thread_index];
         tag[shared_base + (count & warpSize) + thread_index] = 0;
      }
   }

   __global__
   void cuRemovePartitionGaps(
      int* target_reference,
      const int* source_left_reference,
      const int* source_right_reference,
      const int* segment_left_lengths,
      const int* segment_right_lengths,
      int start,
      int end,
      int depth
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto all_warps = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_group_size = all_warps >> depth;
      const int thread_index = index & (warpSize - 1);
      const int warp_index = (index - thread_index) / warpSize;

      int mid;
      for (int i = 0; i < depth; ++i) {
         mid = start + (end - start) / 2;
         if (warp_index & (all_warps >> (i + 1))) start = mid + 1;
         else end = mid - 1;
      }
      mid = start + (end - start) / 2;

      const int partition_size = end - start + 1;
      int segment_size = (partition_size + warp_group_size - 1) / warp_group_size;
      const int start_segment = start + segment_size * (warp_index - (warp_index & ~(warp_group_size - 1)));

      int out_start = start;
      if (thread_index == 0) {
         for (int i = warp_index & ~(warp_group_size - 1); i < warp_index; ++i) out_start += segment_left_lengths[i];
         segment_size = segment_left_lengths[warp_index];
      }

      out_start = __shfl( out_start, 0 );
      segment_size = __shfl( segment_size, 0 );
      copyReferenceWarp( target_reference + out_start, source_left_reference + start_segment, segment_size );

      out_start = mid + 1;
      if (thread_index == 0) {
         for (int i = warp_index & ~(warp_group_size - 1); i < warp_index; ++i) out_start += segment_right_lengths[i];
         segment_size = segment_right_lengths[warp_index];
      }

      out_start = __shfl( out_start, 0 );
      segment_size = __shfl( segment_size, 0 );
      copyReferenceWarp( target_reference + out_start, source_right_reference + start_segment, segment_size );
   }

   __global__
   void cuPartition(
      KdtreeNode* root,
      int* target_reference,
      int* mid_references,
      const int* last_mid_references,
      const int* source_reference,
      const int* primary_reference,
      const node_type* __restrict__ coordinates,
      int start,
      int end,
      int axis,
      int dim,
      int depth,
      int log_warp_num
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto all_warps = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int thread_index = index & (warpSize - 1);
      const int warp_index = (index - thread_index) / warpSize;
      const int loop_levels = depth - log_warp_num;
      for (int loop = 0; loop < (1 << loop_levels); ++loop) {
         int mid;
         for (int i = 1; i <= loop_levels; ++i) {
            mid = start + (end - start) / 2;
            if (loop & (1 << (loop_levels - i))) start = mid + 1;
            else end = mid - 1;
         }
         for (int i = 0; i < log_warp_num; ++i) {
            mid = start + (end - start) / 2;
            if (warp_index & (all_warps >> (i + 1))) start = mid + 1;
            else end = mid - 1;
         }
         mid = start + (end - start) / 2;

         const int partition_size = end - start + 1;
         const int mid_reference = primary_reference[mid];
         partition(
            target_reference + start, target_reference + mid + 1, nullptr, nullptr,
            source_reference + start, coordinates,
            mid_reference, partition_size, partition_size, axis, dim, 1
         );

         if (thread_index == 0) {
            const int m = warp_index + all_warps * loop;
            mid_references[m] = mid_reference;
            if (last_mid_references != nullptr) {
               if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
               else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
            }
         }
      }
   }

   __device__
   void subpartition(
      int* target_left_reference,
      int* target_right_reference,
      const int* source_reference,
      const node_type* __restrict__ coordinates,
      int mid_reference,
      int segment_size,
      int axis,
      int dim,
      int sub_warp_size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int thread_index = index & (warpSize - 1);
      const int sub_warp_index = thread_index / sub_warp_size;
      const int sub_thread_index = thread_index - sub_warp_index * sub_warp_size;
      const int sub_warp_mask = ((1 << sub_warp_size) - 1) << sub_warp_index * sub_warp_size;
      const int mask = (1 << thread_index) - 1;

      int r;
      node_type t;
      if (sub_thread_index < segment_size) {
         r = source_reference[sub_thread_index];
         t = compareSuperKey(
            coordinates[r * dim + axis], coordinates[mid_reference * dim + axis],
            coordinates + r * dim, coordinates + mid_reference * dim, axis, dim
         );
      }
      else t = 0;

      int shuffle_mask = static_cast<int>(__ballot( t < 0 )) & sub_warp_mask;
      if (t < 0) {
         const int j = __popc( shuffle_mask & mask );
         target_left_reference[j] = r;
      }

      shuffle_mask = static_cast<int>(__ballot( t > 0 )) & sub_warp_mask;
      if (t > 0) {
         const int j = __popc( shuffle_mask & mask );
         target_right_reference[j] = r;
      }
   }

   __global__
   void cuPartition(
      KdtreeNode* root,
      int* target_reference,
      int* mid_references,
      const int* last_mid_references,
      const int* source_reference,
      const int* primary_reference,
      const node_type* __restrict__ coordinates,
      int start,
      int end,
      int axis,
      int dim,
      int depth,
      int sub_size,
      int log_sub_warp_num
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto all_sub_warps = static_cast<int>(blockDim.x * gridDim.x / sub_size);
      const int sub_thread_index = index & (sub_size - 1);
      const int sub_warp_index = (index - sub_thread_index) / sub_size;
      const int loop_levels = depth - log_sub_warp_num;
      for (int loop = 0; loop < (1 << loop_levels); ++loop) {
         int mid;
         for (int i = 1; i <= loop_levels; ++i) {
            mid = start + (end - start) / 2;
            if (loop & (1 << (loop_levels - i))) start = mid + 1;
            else end = mid - 1;
         }
         for (int i = 0; i < log_sub_warp_num; ++i) {
            mid = start + (end - start) / 2;
            if (sub_warp_index & (all_sub_warps >> (i + 1))) start = mid + 1;
            else end = mid - 1;
         }
         mid = start + (end - start) / 2;

         const int partition_size = end - start + 1;
         const int mid_reference = primary_reference[mid];
         subpartition(
            target_reference + start, target_reference + mid + 1,
            source_reference + start, coordinates,
            mid_reference, partition_size, axis, dim, sub_size
         );

         if (sub_thread_index == 0) {
            const int m = sub_warp_index + all_sub_warps * loop;
            mid_references[m] = mid_reference;
            if (last_mid_references != nullptr) {
               if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
               else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
            }
         }
      }
   }

   void KdtreeCUDA::partitionDimension(Device& device, int axis, int depth) const
   {
      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int warp_num = total_thread_num / WarpSize;
      const auto log_warp_num = static_cast<int>(std::log2( static_cast<double>(warp_num) ));
      const auto log_size = static_cast<int>(std::ceil( std::log2( static_cast<double>(device.TupleNum) ) ));
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSizeLimit, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSizeLimit / 2 );

      setDevice( device.ID );
      int* mid_references = device.MidReferences[depth % 2];
      int* last_mid_references = depth == 0 ? nullptr : device.MidReferences[(depth - 1) % 2];
      if (depth < log_warp_num) {
         for (int i = 1; i < Dim; ++i) {
            int r = i + axis;
            r = r < Dim ? r : r - Dim;
            cuPartition<<<block_num, thread_num_per_block, 0, device.Stream>>>(
               device.Root, device.LeftSegmentLengths, device.RightSegmentLengths,
               device.Reference[Dim], device.Reference[Dim + 1], mid_references,
               last_mid_references, device.Reference[r], device.Reference[axis],
               device.CoordinatesDevicePtr, 0, device.TupleNum - 1, axis, Dim, depth
            );
            CHECK_KERNEL;

            cuRemovePartitionGaps<<<block_num, thread_num_per_block, 0, device.Stream>>>(
               device.Reference[r],
               device.Reference[Dim], device.Reference[Dim + 1],
               device.LeftSegmentLengths, device.RightSegmentLengths,
               0, device.TupleNum - 1, depth
            );
            CHECK_KERNEL;
         }
      }
      else {
         const int log_sub_size = log_size - depth;
         if (log_sub_size > 5) {
            for (int i = 1; i < Dim; ++i) {
               int r = i + axis;
               r = r < Dim ? r : r - Dim;
               cuPartition<<<block_num, thread_num_per_block, 0, device.Stream>>>(
                  device.Root, device.Reference[Dim], mid_references,
                  last_mid_references, device.Reference[r], device.Reference[axis],
                  device.CoordinatesDevicePtr, 0, device.TupleNum - 1, axis, Dim, depth, log_warp_num
               );
               CHECK_KERNEL;

               cuCopyReference<<<block_num, thread_num_per_block, 0, device.Stream>>>(
                  device.Reference[r], device.Reference[Dim], device.TupleNum
               );
               CHECK_KERNEL;
            }
         }
         else {
            const int log_sub_warp_num = log_warp_num - (log_sub_size - 5);
            for (int i = 1; i < Dim; ++i) {
               int r = i + axis;
               r = r < Dim ? r : r - Dim;
               cuPartition<<<block_num, thread_num_per_block, 0, device.Stream>>>(
                  device.Root, device.Reference[Dim], mid_references,
                  last_mid_references, device.Reference[r], device.Reference[axis],
                  device.CoordinatesDevicePtr,
                  0, device.TupleNum - 1, axis, Dim, depth, 1 << log_sub_size, log_sub_warp_num
               );
               CHECK_KERNEL;

               cuCopyReference<<<block_num, thread_num_per_block, 0, device.Stream>>>(
                  device.Reference[r], device.Reference[Dim], device.TupleNum
               );
               CHECK_KERNEL;
            }
         }
      }

      if (depth == 0) {
         CHECK_CUDA(
            cudaMemcpyAsync(
               &device.RootNode, device.MidReferences[0], sizeof( int ), cudaMemcpyDeviceToHost, device.Stream
            )
         );
      }

      assert( device.RootNode != -1 );
   }

   __global__
   void cuPartitionFinal(
      KdtreeNode* root,
      int* mid_references,
      const int* last_mid_references,
      const int* primary_reference,
      int start,
      int end,
      int depth
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto all_warps = static_cast<int>(blockDim.x * gridDim.x);

      int mid;
      for (int i = 0; i < depth; ++i) {
         mid = start + (end - start) / 2;
         if (index & (all_warps >> (i + 1))) start = mid + 1;
         else end = mid - 1;
      }

      int mid_reference = -1;
      if (end == start) mid_reference = primary_reference[end];
      else if (end == start + 1) {
         mid_reference = primary_reference[start];
         root[mid_reference].RightChildIndex = primary_reference[end];
      }
      else if (end == start + 2) {
         mid_reference = primary_reference[start + 1];
         root[mid_reference].LeftChildIndex = primary_reference[start];
         root[mid_reference].RightChildIndex = primary_reference[end];
      }

      if (mid_reference != -1) {
         mid_references[index] = mid_reference;
         if (index & 1) root[last_mid_references[index >> 1]].RightChildIndex = mid_reference;
         else root[last_mid_references[index >> 1]].LeftChildIndex = mid_reference;
      }
   }

   void KdtreeCUDA::partitionDimensionFinal(Device& device, int axis, int depth) const
   {
      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int warp_num = total_thread_num;
      const auto log_warp_num = static_cast<int>(std::log2( static_cast<double>(warp_num) ));
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSizeLimit, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSizeLimit / 2 );
      const int loop_levels = log_warp_num < depth ? depth - log_warp_num : 0;

      setDevice( device.ID );
      int* mid_references = device.MidReferences[depth % 2];
      int* last_mid_references = depth == 0 ? nullptr : device.MidReferences[(depth - 1) % 2];
      for (int loop = 0; loop < (1 << loop_levels); ++loop) {
         int start = 0, end = device.TupleNum - 1, mid;
         for (int i = 1; i <= loop_levels; ++i) {
            mid = start + (end - start) / 2;
            if (loop & (1 << (loop_levels - i))) start = mid + 1;
            else end = mid - 1;

            int r = 1 + axis;
            r = r < Dim ? r : r - Dim;
            cuPartitionFinal<<<block_num, thread_num_per_block, 0, device.Stream>>>(
               device.Root,
               mid_references + loop * total_thread_num,
               last_mid_references + loop * total_thread_num / 2,
               device.Reference[axis],
               start, end, depth - loop_levels
            );
            CHECK_KERNEL;
         }
      }
   }

   void KdtreeCUDA::build()
   {
      constexpr int warp_num = ThreadBlockNum * ThreadNum / WarpSize;
      for (auto& device : Devices) {
         setDevice( device.ID );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.LeftSegmentLengths), sizeof( int ) * warp_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.RightSegmentLengths), sizeof( int ) * warp_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.MidReferences[0]), sizeof( int ) * device.TupleNum ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.MidReferences[1]), sizeof( int ) * device.TupleNum ) );
      }

      const int start_axis = DeviceNum == 1 ? 0 : 1;
      for (auto& device : Devices) {
         assert( !device.Reference.empty() );
         for (int axis = 0; axis < Dim; ++axis) assert( device.Reference[axis] != nullptr );

         setDevice( device.ID );
         if (device.Reference[Dim] == nullptr) {
            CHECK_CUDA(
               cudaMalloc( reinterpret_cast<void**>(&device.Reference[Dim]), sizeof( int ) * device.TupleNum )
            );
         }

         const auto depth = static_cast<int>(std::floor( std::log2( static_cast<double>(device.TupleNum) ) ));
         for (int i = 0; i < depth - 1; ++i) {
            const int axis = (i + start_axis) % Dim;
            partitionDimension( device, axis, i );
         }
         partitionDimensionFinal( device, (depth - 1 + start_axis) % Dim, depth - 1 );
      }

      if (DeviceNum == 1) RootNode = Devices[0].RootNode;

      for (auto& device : Devices) {
         setDevice( device.ID );
         CHECK_CUDA( cudaStreamSynchronize( device.Stream ) );
         CHECK_CUDA( cudaFree( device.LeftSegmentLengths ) );
         CHECK_CUDA( cudaFree( device.RightSegmentLengths ) );
         CHECK_CUDA( cudaFree( device.MidReferences[0] ) );
         CHECK_CUDA( cudaFree( device.MidReferences[1] ) );
      }
   }

   __device__ int verify_error;

   __global__
   void cuVerify(
      int* node_sums,
      int* next_child,
      const int* child,
      const KdtreeNode* root,
      const node_type* coordinates,
      int size,
      int axis,
      int dim
   )
   {
      __shared__ int sums[SharedSizeLimit];

      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      const auto id = static_cast<int>(threadIdx.x);

      int node, count = 0;
      for (int i = index; i < size; i += step) {
         node = child[i];
         if (node >= 0) {
            count++;
            const int right = root[node].RightChildIndex;
            next_child[i * 2 + 1] = right;
            if (right != -1) {
               if (compareSuperKey(
                     coordinates + root[right].Index * dim, coordinates + root[node].Index * dim, axis, dim
                   ) <= 0) {
                  verify_error = 1;
               }
            }

            const int left = root[node].LeftChildIndex;
            next_child[i * 2] = left;
            if (left != -1) {
               if (compareSuperKey(
                     coordinates + root[left].Index * dim, coordinates + root[node].Index * dim, axis, dim
                   ) >= 0) {
                  verify_error = 1;
               }
            }
         }
         else next_child[i * 2] = next_child[i * 2 + 1] = -1;
      }
      sums[id] = count;
      __syncthreads();

      for (int i = static_cast<int>(blockDim.x / 2); i > warpSize; i >>= 1) {
         if (id < i) {
            count += sums[id + i];
            sums[id] = count;
         }
         __syncthreads();
      }

      if (id < warpSize) {
         if (blockDim.x >= 2 * warpSize) count += sums[id + warpSize];
         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            count += __shfl_down( count, offset );
         }
      }

      if (id == 0) node_sums[blockIdx.x] += count;
   }

   __global__
   void cuSumNodeNum(int* node_sums)
   {
      __shared__ int sums[SharedSizeLimit];

      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      const auto id = static_cast<int>(threadIdx.x);

      int sum = 0;
      for (int i = id; i < ThreadBlockNum; i += step) sum += node_sums[i];
      sums[id] = sum;
      __syncthreads();

      for (int i = static_cast<int>(blockDim.x / 2); i > warpSize; i >>= 1) {
         if (id < i) {
            sum += sums[id + i];
            sums[id] = sum;
         }
         __syncthreads();
      }

      if (id < warpSize) {
         if (blockDim.x >= 2 * warpSize) sum += sums[id + warpSize];
         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down( sum, offset );
         }
      }

      if (id == 0) node_sums[blockIdx.x] += sum;
   }

   int KdtreeCUDA::verify(Device& device, int start_axis) const
   {
      const auto log_size = static_cast<int>(std::floor( std::log2( static_cast<double>(device.TupleNum) ) ));

      setDevice( device.ID );
      CHECK_CUDA(
         cudaMemcpyAsync(
            device.MidReferences[0], &device.RootNode, sizeof( int ), cudaMemcpyHostToDevice, device.Stream
         )
      );

      int error = 0;
      CHECK_CUDA(
         cudaMemcpyToSymbolAsync( verify_error, &error, sizeof( error ), 0, cudaMemcpyHostToDevice, device.Stream )
      );

      int* child;
      int* next_child;
      for (int i = 0; i <= log_size; ++i) {
         const int needed_threads = 1 << i;
         const int block_num = std::clamp( needed_threads / ThreadNum, 1, ThreadBlockNum );
         const int axis = (i + start_axis) % Dim;
         child = device.MidReferences[i % 2];
         next_child = device.MidReferences[(i + 1) % 2];
         cuVerify<<<block_num, ThreadNum, 0, device.Stream>>>(
            device.NodeSums, next_child,
            child, device.Root, device.CoordinatesDevicePtr, needed_threads, axis, Dim
         );
         CHECK_KERNEL;

         CHECK_CUDA(
            cudaMemcpyFromSymbolAsync( &error, verify_error, sizeof( error ), 0, cudaMemcpyDeviceToHost, device.Stream )
         );
         CHECK_CUDA( cudaStreamSynchronize( device.Stream ) );
      }

      cuSumNodeNum<<<1, ThreadNum, 0, device.Stream>>>( device.NodeSums );
      CHECK_KERNEL;

      int node_num;
      CHECK_CUDA( cudaMemcpyAsync( &node_num, device.NodeSums, sizeof( int ), cudaMemcpyDeviceToHost, device.Stream ) );
      return node_num;
   }

   int KdtreeCUDA::verify()
   {
      for (auto& device : Devices) {
         setDevice( device.ID );
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&device.MidReferences[0]), sizeof( int ) * 2 * device.TupleNum )
         );
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&device.MidReferences[1]), sizeof( int ) * 2 * device.TupleNum )
         );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.NodeSums), sizeof( int ) * ThreadBlockNum ) );
         CHECK_CUDA( cudaMemset( device.NodeSums, 0, sizeof( int ) * ThreadBlockNum ) );
      }

      int node_num = 0;
      if (DeviceNum > 1) {
         node_num += verify( Devices[0], 1 );
         node_num += verify( Devices[1], 1 );
         node_num++;
      }
      else node_num += verify( Devices[0], 0 );

      for (auto& device : Devices) {
         setDevice( device.ID );
         CHECK_CUDA( cudaStreamSynchronize( device.Stream ) );
         CHECK_CUDA( cudaFree( device.MidReferences[0] ) );
         CHECK_CUDA( cudaFree( device.MidReferences[1] ) );
         CHECK_CUDA( cudaFree( device.NodeSums ) );
      }
      return node_num;
   }

   void KdtreeCUDA::create()
   {
      const int size_per_device = TupleNum / DeviceNum;
      for (int i = 0; i < DeviceNum; ++i) {
         const node_type* ptr = Coordinates + i * Dim * size_per_device;
         initialize( Devices[i], ptr, size_per_device );
      }
      CHECK_CUDA( cudaDeviceSynchronize() );

      auto start_time = std::chrono::system_clock::now();
      std::vector<int> end(Dim);
      sort( end );
      auto end_time = std::chrono::system_clock::now();
      const auto sort_time =
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

      for (int i = 0; i < Dim - 1; ++i) {
         assert( end[i] >= 0 );
         for (int j = i + 1; j < Dim; ++j) assert( end[i] == end[j] );
      }

      start_time = std::chrono::system_clock::now();
      build();
      end_time = std::chrono::system_clock::now();
      const auto build_time =
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

      start_time = std::chrono::system_clock::now();
      NodeNum = verify();
      end_time = std::chrono::system_clock::now();
      const auto verify_time =
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

      std::cout << " >> " << TupleNum - end[0] << " duplicates removed\n";
      std::cout << " >> Number of nodes = " << NodeNum << "\n" << std::fixed << std::setprecision( 2 )
         << " >> Total Time = "  << sort_time + build_time + verify_time << " sec."
         << "\n\t* Sort Time = " << sort_time << " sec."
         << "\n\t* Build Time = " << build_time << " sec."
         << "\n\t* Verify Time = " << verify_time << " sec.\n\n";
   }

   void KdtreeCUDA::print(const std::vector<KdtreeNode>& kd_nodes, int index, int depth) const
   {
      if (kd_nodes[index].RightChildIndex >= 0) print( kd_nodes, kd_nodes[index].RightChildIndex, depth + 1 );

      for (int i = 0; i < depth; ++i) std::cout << "       ";

      const node_type* tuple = Coordinates + kd_nodes[index].Index;
      std::cout << "(" << tuple[0] << ",";
      for (int i = 1; i < Dim - 1; ++i) std::cout << tuple[i] << ",";
      std::cout << tuple[Dim - 1] << ")\n";

      if (kd_nodes[index].LeftChildIndex >= 0) print( kd_nodes, kd_nodes[index].LeftChildIndex, depth + 1 );
   }

   void KdtreeCUDA::print() const
   {
      if (RootNode < 0 || Coordinates == nullptr) return;

      std::vector<KdtreeNode> kd_nodes(TupleNum);
      const int size_per_device = TupleNum / DeviceNum;
      for (int i = 0; i < DeviceNum; ++i) {
         setDevice( Devices[i].ID );
         CHECK_CUDA(
            cudaMemcpyAsync(
               kd_nodes.data() + i * size_per_device, Devices[i].Root, sizeof( KdtreeNode ) * size_per_device,
               cudaMemcpyDeviceToHost, Devices[i].Stream
            )
         );
      }

      if (DeviceNum > 1) {
         kd_nodes[RootNode].LeftChildIndex = Devices[0].RootNode;
         kd_nodes[RootNode].RightChildIndex = Devices[1].RootNode;
         kd_nodes[RootNode].Index = RootNode;
         for (int i = size_per_device; i < TupleNum; ++i) {
            if (kd_nodes[i].LeftChildIndex >= 0) kd_nodes[i].LeftChildIndex += size_per_device;
            if (kd_nodes[i].RightChildIndex >= 0) kd_nodes[i].RightChildIndex += size_per_device;
            if (kd_nodes[i].Index >= 0) kd_nodes[i].Index += size_per_device;
         }
      }

      print( kd_nodes, RootNode, 0 );
   }
}
#endif