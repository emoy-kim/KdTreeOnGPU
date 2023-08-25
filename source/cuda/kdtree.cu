#include "cuda/kdtree.cuh"

namespace cuda
{
   __host__ __device__
   inline int divideUp(int a, int b)
   {
      return (a + b - 1) / b;
   }

   __host__ __device__
   int getSampleNum(int x)
   {
      return divideUp( x, SampleStride );
   }

   __device__
   static inline int getNextPowerOfTwo(int x)
   {
      /*
         --x;
         x |= x >> 1;
         x |= x >> 2;
         x |= x >> 4;
         x |= x >> 8;
         x |= x >> 16;
         return ++x;
       */
      constexpr int bits = sizeof( int ) * 8;
      return 1 << (bits - __clz( x - 1 ));
   }

   KdtreeCUDA::KdtreeCUDA(const node_type* vertices, int size, int dim) : Dim( dim ), NodeNum( 0 )
   {
      if (DeviceNum == 0) prepareCUDA();
      create( vertices, size );
   }

   KdtreeCUDA::~KdtreeCUDA()
   {
      for (auto& device : Devices) {
         setDevice( device.ID );
         if (!device.Reference.empty()) {
            for (int i = 0; i <= Dim; ++i) {
               if (device.Reference[i] != nullptr) cudaFree( device.Reference[i] );
            }
         }
         if (!device.Buffer.empty()) {
            for (int i = 0; i <= Dim; ++i) {
               if (device.Buffer[i] != nullptr) cudaFree( device.Buffer[i] );
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
      if (device.CoordinatesDevicePtr != nullptr) {
         throw std::runtime_error( "coordinates device ptr already allocated!" );
      }
      if (device.Root != nullptr) throw std::runtime_error( "k-d tree already allocated!" );

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

   void KdtreeCUDA::initializeReference(Device& device, int size, int axis)
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
      const node_type* coordinates,
      const int* reference,
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
      int index,
      node_type value,
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
            buffer[j - 1], value, coordinates + reference[j - 1] * dim, coordinates + index * dim, axis, dim
         );
         if (t < 0 || (inclusive && t == 0)) i = j;
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

      for (int step = 1; step < SharedSizeLimit; step <<= 1) {
         const int i = static_cast<int>(threadIdx.x) & (step - 1);
         node_type* base_buffer = buffer + 2 * (threadIdx.x - i);
         int* base_reference = reference + 2 * (threadIdx.x - i);

         __syncthreads();
         const node_type buffer_x = base_buffer[i];
         const int reference_x = base_reference[i];
         const node_type buffer_y = base_buffer[i + step];
         const int reference_y = base_reference[i + step];
         const int x = search(
            reference_x, buffer_x, base_reference + step, base_buffer + step, coordinates, step, step, axis, dim, false
         ) + i;
         const int y = search(
            reference_y, buffer_y, base_reference, base_buffer, coordinates, step, step, axis, dim, false
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

   __global__
   void cuGenerateSampleRanks(
      int* ranks_a,
      int* ranks_b,
      int* reference,
      node_type* buffer,
      const node_type* coordinates,
      int step,
      int size,
      int axis,
      int dim,
      int thread_num
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= thread_num) return;

      const int i = index & (step / SampleStride - 1);
      const int segment_base = (index - i) * 2 * SampleStride;
      buffer += segment_base;
      reference += segment_base;
      ranks_a += segment_base / SampleStride;
      ranks_b += segment_base / SampleStride;

      const int element_a = step;
      const int element_b = min( step, size - step - segment_base );
      const int sample_a = getSampleNum( element_a );
      const int sample_b = getSampleNum( element_b );
      if (i < sample_a) {
         ranks_a[i] = i * SampleStride;
         ranks_b[i] = search(
            reference[i * SampleStride], buffer[i * SampleStride],
            reference + step, buffer + step, coordinates,
            element_b, getNextPowerOfTwo( element_b ), axis, dim, false
         );
      }
      if (i < sample_b) {
         ranks_b[step / SampleStride + i] = i * SampleStride;
         ranks_a[step / SampleStride + i] = search(
            reference[i * SampleStride + step], buffer[i * SampleStride + step],
            reference, buffer, coordinates,
            element_a, getNextPowerOfTwo( element_a ), axis, dim, true
         );
      }
   }

   __global__
   void cuMergeRanksAndIndices(int* limits, const int* ranks, int step, int size, int thread_num)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= thread_num) return;

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
      __shared__ int reference[2 * SampleStride];
      __shared__ node_type buffer[2 * SampleStride];

      const int index = static_cast<int>(blockIdx.x) & (2 * step / SampleStride - 1);
      const int segment_base = (static_cast<int>(blockIdx.x) - index) * SampleStride;
      target_buffer += segment_base;
      target_reference += segment_base;
      source_buffer += segment_base;
      source_reference += segment_base;

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

   void KdtreeCUDA::sortPartially(
      Device& device,
      int source_index,
      int target_index,
      int start_offset,
      int size,
      int axis
   )
   {
      assert( device.CoordinatesDevicePtr != nullptr );
      assert( device.Reference[source_index] != nullptr && device.Reference[target_index] != nullptr );

      setDevice( device.ID );
      if (device.Buffer[source_index] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&device.Buffer[source_index]), sizeof( node_type ) * size )
         );
         cuCopyCoordinates<<<ThreadBlockNum, ThreadNum, 0, device.Stream>>>(
            device.Buffer[source_index],
            device.CoordinatesDevicePtr + start_offset * Dim,
            device.Reference[source_index],
            size, axis, Dim
         );
         CHECK_KERNEL;
      }
      if (device.Buffer[target_index] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&device.Buffer[target_index]), sizeof( node_type ) * size )
         );
      }

      int stage_num = 0;
      for (int step = SharedSizeLimit; step < size; step <<= 1) stage_num++;

      int* in_reference = nullptr;
      int* out_reference = nullptr;
      node_type* in_buffer = nullptr;
      node_type* out_buffer = nullptr;
      if (stage_num & 1) {
         in_buffer = device.Sort.Buffer;
         in_reference = device.Sort.Reference;
         out_buffer = device.Buffer[target_index];
         out_reference = device.Reference[target_index] + start_offset;
      }
      else {
         in_buffer = device.Buffer[target_index];
         in_reference = device.Reference[target_index] + start_offset;
         out_buffer = device.Sort.Buffer;
         out_reference = device.Sort.Reference;
      }

      assert( size <= SampleStride * device.Sort.MaxSampleNum );
      assert( size % SharedSizeLimit == 0 );

      const int block_num = size / SharedSizeLimit;
      int thread_num = SharedSizeLimit / 2;
      cuSort<<<block_num, thread_num, 0, device.Stream>>>(
         in_reference, in_buffer,
         device.Reference[source_index] + start_offset, device.Buffer[source_index],
         device.CoordinatesDevicePtr, axis, Dim
      );
      CHECK_KERNEL;

      for (int step = SharedSizeLimit; step < size; step <<= 1) {
         const int last = size % (2 * step);
         thread_num = last > step ? (size + 2 * step - last) / (2 * SampleStride) : (size - last) / (2 * SampleStride);
         cuGenerateSampleRanks<<<divideUp( thread_num, 256 ), 256, 0, device.Stream>>>(
            device.Sort.RanksA, device.Sort.RanksB,
            in_reference, in_buffer, device.CoordinatesDevicePtr,
            step, size, axis, Dim, thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<divideUp( thread_num, 256 ), 256, 0, device.Stream>>>(
            device.Sort.LimitsA, device.Sort.RanksA, step, size, thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<divideUp( thread_num, 256 ), 256, 0, device.Stream>>>(
            device.Sort.LimitsB, device.Sort.RanksB, step, size, thread_num
         );
         CHECK_KERNEL;

         const int merge_pairs = last > step ? getSampleNum( size ) : (size - last) / SampleStride;
         cuMergeReferences<<<merge_pairs, SampleStride, 0, device.Stream>>>(
            out_reference, out_buffer,
            in_reference, in_buffer, device.CoordinatesDevicePtr,
            device.Sort.LimitsA, device.Sort.LimitsB,
            step, size, axis, Dim
         );
         CHECK_KERNEL;

         if (last <= step) {
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_reference + size - last, in_reference + size - last, sizeof( int ) * last,
                  cudaMemcpyDeviceToDevice, device.Stream
               )
            );
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_buffer + size - last, in_buffer + size - last, sizeof( node_type ) * last,
                  cudaMemcpyDeviceToDevice, device.Stream
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

      int index;
      node_type value;
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
            value = buffer_b[k];
            index = reference_b[k];
            target_buffer[j] = value;
            target_reference[j] = index;
         }
         else if (b0 == b1) {
            const int k = a0 + x;
            value = buffer_a[k];
            index = reference_a[k];
            target_buffer[j] = value;
            target_reference[j] = index;
         }
         else {
            const bool inclusive = x < a1 - a0;
            if (inclusive) {
               const int k = a0 + x;
               value = buffer_a[k];
               index = reference_a[k];
            }
            else {
               const int k = b0 + x - (a1 - a0);
               value = buffer_b[k];
               index = reference_b[k];
            }
            in_buffer[x] = value;
            in_reference[x] = index;
            __syncthreads();

            const int n = inclusive ? b1 - b0 : a1 - a0;
            const int offset = inclusive ? a1 - a0 : 0;
            const int t = search(
               index, value, in_reference + offset, in_buffer + offset,
               coordinates, n, getNextPowerOfTwo( n ), axis, dim, inclusive
            );
            const int k = inclusive ? t + x : t + x - (a1 - a0);
            out_buffer[k] = value;
            out_reference[k] = index;
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
   )
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

   void KdtreeCUDA::sort(int* end, int size)
   {
      const int max_sample_num = size / SampleStride + 1;
      for (auto& device : Devices) {
         setDevice( device.ID );
         device.Sort.MaxSampleNum = max_sample_num;
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.RanksA), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.RanksB), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.LimitsA), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.LimitsB), sizeof( int ) * max_sample_num ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.Reference), sizeof( int ) * size ) );
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device.Sort.Buffer), sizeof( node_type ) * size ) );
      }

      const int size_per_device = size / DeviceNum;
      if (DeviceNum > 1) {
         for (auto& device : Devices) {
            initializeReference( device, size_per_device, 0 );
            sortPartially( device, 0, Dim, 0, size_per_device, 0 );
         }
         sync();

         const int pivot = swapBalanced( Dim, 0, size_per_device, 0 );
         std::cout << " >> Pivot = " << pivot << "\n";
         sync();

         mergeSwap( Devices[0], Dim, 0, size_per_device - pivot, size_per_device );
         mergeSwap( Devices[1], Dim, 0, pivot, size_per_device );
         sync();

         for (int i = 0; i < DeviceNum; ++i) {

         }
      }
      else {
         setDevice( Devices[0].ID );
         for (int axis = 0; axis < Dim; ++axis) {
            initializeReference( Devices[0], size_per_device, axis );
            sortPartially( Devices[0], axis, Dim, 0, size_per_device, axis );
         }
      }
      sync();
   }

   void KdtreeCUDA::create(const node_type* coordinates, int size)
   {
      const int size_per_device = size / DeviceNum;
      for (int i = 0; i < DeviceNum; ++i) {
         const node_type* ptr = coordinates + i * Dim * size_per_device;
         initialize( Devices[i], ptr, size_per_device );
      }
      cudaDeviceSynchronize();

      int end[Dim];
      sort( end, size );
   }
}