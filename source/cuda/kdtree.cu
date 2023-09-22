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
      Coordinates( vertices ), Dim( dim ), TupleNum( size ), NodeNum( 0 )
   {
      prepareCUDA();
      create();
   }

   KdtreeCUDA::~KdtreeCUDA()
   {
      if (!Device.Reference.empty()) {
         for (int axis = 0; axis <= Dim; ++axis) {
            if (Device.Reference[axis] != nullptr) cudaFree( Device.Reference[axis] );
         }
      }
      if (!Device.Buffer.empty()) {
         for (int axis = 0; axis <= Dim; ++axis) {
            if (Device.Buffer[axis] != nullptr) cudaFree( Device.Buffer[axis] );
         }
      }
      if (Device.CoordinatesDevicePtr != nullptr) cudaFree( Device.CoordinatesDevicePtr );
      if (Device.Root != nullptr) cudaFree( Device.Root );
      cudaStreamDestroy( Device.Stream );
   }

   void KdtreeCUDA::prepareCUDA()
   {
      int device_num = 0;
      CHECK_CUDA( cudaGetDeviceCount( &device_num ) );
      if( device_num <= 0 ) throw std::runtime_error( "cuda device not found\n" );

      Device.ID = 0;
      Device.Buffer.resize( Dim + 2, nullptr );
      Device.Reference.resize( Dim + 2, nullptr );

      CHECK_CUDA( cudaSetDevice( Device.ID ) );
      CHECK_CUDA( cudaStreamCreate( &Device.Stream ) );
   }

   __global__
   void cuInitialize(KdtreeNode* root, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         root[i].Index = i;
         root[i].ParentIndex = -1;
         root[i].LeftChildIndex = -1;
         root[i].RightChildIndex = -1;
      }
   }

   void KdtreeCUDA::initialize(const node_type* coordinates, int size)
   {
      assert( Device.Root == nullptr );
      assert( Device.CoordinatesDevicePtr == nullptr );

      CHECK_CUDA(
         cudaMalloc(
            reinterpret_cast<void**>(&Device.CoordinatesDevicePtr),
            sizeof( node_type ) * Dim * (size + 1)
         )
      );
      CHECK_CUDA(
         cudaMemcpyAsync(
            Device.CoordinatesDevicePtr, coordinates, sizeof( node_type ) * Dim * size,
            cudaMemcpyHostToDevice, Device.Stream
         )
      );

      node_type max_value[Dim];
      for (int i = 0; i < Dim; ++i) max_value[i] = std::numeric_limits<node_type>::max();
      CHECK_CUDA(
         cudaMemcpyAsync(
            Device.CoordinatesDevicePtr + size * Dim, max_value, sizeof( node_type ) * Dim,
            cudaMemcpyHostToDevice, Device.Stream
         )
      );

      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Root), sizeof( KdtreeNode ) * size ) );

      cuInitialize<<<ThreadBlockNum, ThreadNum, 0, Device.Stream>>>( Device.Root, size );
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

   void KdtreeCUDA::initializeReference(int axis)
   {
      std::vector<int*>& references = Device.Reference;
      for (int i = 0; i <= Dim + 1; ++i) {
         if (references[i] == nullptr) {
            CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&references[i]), sizeof( int ) * TupleNum ) );
         }
      }
      cuInitializeReference<<<ThreadBlockNum, ThreadNum, 0, Device.Stream>>>( references[axis], TupleNum );
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
      node_type difference = a[axis] - b[axis];
      for (int i = 1; difference == 0 && i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         difference = a[r] - b[r];
      }
      return difference;
   }

   __device__
   node_type compareSuperKey(const node_type* a, const node_type* b, node_type delta, int axis, int dim)
   {
      node_type difference = a[axis] - (b[axis] + delta);
      for (int i = 1; difference == 0 && i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         difference = a[r] - (b[r] + delta);
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

      // Find the right place to put v among buffer in ascending order.
      // When it is inclusive, the place will be the rightmost among the same values with v.
      // When it is exclusive, the place will be the leftmost among the same values with v.
      // Local variable i points the index to put v, which means the number of values less than (or equal to) v.
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
      int size,
      int axis,
      int dim
   )
   {
      __shared__ int reference[SharedSize];
      __shared__ node_type buffer[SharedSize];

      const auto t = static_cast<int>(threadIdx.x);
      const int target_block_size = static_cast<int>(blockDim.x) * 2;
      const int index = static_cast<int>(blockIdx.x) * target_block_size + t;
      source_buffer += index;
      source_reference += index;
      target_buffer += index;
      target_reference += index;
      buffer[t] = source_buffer[0];
      reference[t] = source_reference[0];
      buffer[blockDim.x + t] = source_buffer[blockDim.x];
      reference[blockDim.x + t] = source_reference[blockDim.x];

      // Given S = SharedSize, for all threads, [base[i], base[i+step]] is
      // step 1: [0, 1] ... [S-2, S-1]
      // step 2: [0, 2] [1, 3] ... [S-4, S-2], [S-3, S-1]
      // step 4: [0, 4] [1, 5] [2, 6] [3, 7] ... [S-8, S-4] [S-7, S-3] [S-6, S-2] [S-5, S-1]
      //   ...
      // step S/2: [0, S/2] ... [S/2-1, S-1]
      for (int step = 1; step < target_block_size; step <<= 1) {
         const int i = t & (step - 1);
         const int offset = (t - i) * 2;
         int* reference_base = reference + offset;
         node_type* buffer_base = buffer + offset;

         // Merge the sorted array X, base[0] ~ base[step-1], and Y, base[step] ~ base[step*2-1]
         __syncthreads();
         const int reference_x = reference_base[i];
         const node_type buffer_x = buffer_base[i];
         const int x = search(
            reference_x, buffer_x, reference_base + step, buffer_base + step, coordinates, step, step, axis, dim, false
         ) + i;
         const int reference_y = reference_base[i + step];
         const node_type buffer_y = buffer_base[i + step];
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
      target_buffer[0] = buffer[t];
      target_reference[0] = reference[t];
      target_buffer[blockDim.x] = buffer[blockDim.x + t];
      target_reference[blockDim.x] = reference[blockDim.x + t];
   }

   __global__
   void cuSortLastBlock(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      int sorted_size,
      int size,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * sorted_size * 2;
      if (index >= size) return;

      const int mid = min( sorted_size, size - index );
      const int end = min( sorted_size * 2, size - index );
      source_buffer += index;
      source_reference += index;
      target_buffer += index;
      target_reference += index;
      int left = 0, right = mid;
      for (int i = 0; i < end; ++i) {
         const bool take_from_left = left < mid && (right >= end || compareSuperKey(
            coordinates + source_reference[left] * dim, coordinates + source_reference[right] * dim, axis, dim
         ) < 0);
         if (take_from_left) {
            target_buffer[i] = source_buffer[left];
            target_reference[i] = source_reference[left];
            left++;
         }
         else {
            target_buffer[i] = source_buffer[right];
            target_reference[i] = source_reference[right];
            right++;
         }
      }
   }

   __global__
   void cuGenerateSampleRanks(
      int* left_ranks,
      int* right_ranks,
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
      left_ranks += (index - i) * 2;
      right_ranks += (index - i) * 2;

      const int left_elements = sorted_size;
      const int right_elements = min( sorted_size, size - (segment_base + sorted_size) );
      const int left_sample_num = getSampleNum( left_elements );
      const int right_sample_num = getSampleNum( right_elements );
      if (i < left_sample_num) {
         left_ranks[i] = i * SampleStride;
         right_ranks[i] = search(
            reference[i * SampleStride], buffer[i * SampleStride],
            reference + sorted_size, buffer + sorted_size, coordinates,
            right_elements, getNextPowerOfTwo( right_elements ), axis, dim, false
         );
      }
      if (i < right_sample_num) {
         right_ranks[sorted_size / SampleStride + i] = i * SampleStride;
         left_ranks[sorted_size / SampleStride + i] = search(
            reference[sorted_size + i * SampleStride], buffer[sorted_size + i * SampleStride],
            reference, buffer, coordinates,
            left_elements, getNextPowerOfTwo( left_elements ), axis, dim, true
         );
      }
   }

   __global__
   void cuMergeRanksAndIndices(int* limits, const int* ranks, int sorted_size, int size, int total_thread_num)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= total_thread_num) return;

      const int i = index & (sorted_size / SampleStride - 1);
      const int segment_base = (index - i) * SampleStride * 2;
      ranks += (index - i) * 2;
      limits += (index - i) * 2;

      const int left_elements = sorted_size;
      const int right_elements = min( sorted_size, size - (segment_base + sorted_size) );
      const int left_sample_num = getSampleNum( left_elements );
      const int right_sample_num = getSampleNum( right_elements );
      if (i < left_sample_num) {
         int x = 0;
         if (right_sample_num > 0) {
            for (int s = getNextPowerOfTwo( right_sample_num ); s > 0; s >>= 1) {
               const int j = min( x + s, right_sample_num );
               if (ranks[left_sample_num + j - 1] < ranks[i]) x = j;
            }
         }
         limits[x + i] = ranks[i];
      }
      if (i < right_sample_num) {
         int x = 0;
         if (left_sample_num > 0) {
            for (int s = getNextPowerOfTwo( left_sample_num ); s > 0; s >>= 1) {
               const int j = min( x + s, left_sample_num );
               if (ranks[j - 1] <= ranks[left_sample_num + i]) x = j;
            }
         }
         limits[x + i] = ranks[left_sample_num + i];
      }
   }

   __device__
   void merge(
      int* reference,
      node_type* buffer,
      const node_type* coordinates,
      int left_length,
      int right_length,
      int axis,
      int dim
   )
   {
      const int* left_reference = reference;
      const int* right_reference = reference + SampleStride;
      const node_type* left_buffer = buffer;
      const node_type* right_buffer = buffer + SampleStride;

      int left_index, right_index, x, y;
      node_type left_value, right_value;
      if (threadIdx.x < left_length) {
         left_value = left_buffer[threadIdx.x];
         left_index = left_reference[threadIdx.x];
         x = static_cast<int>(threadIdx.x) + search(
            left_index, left_value, right_reference, right_buffer,
            coordinates, right_length, SampleStride, axis, dim, false
         );
      }
      if (threadIdx.x < right_length) {
         right_value = right_buffer[threadIdx.x];
         right_index = right_reference[threadIdx.x];
         y = static_cast<int>(threadIdx.x) + search(
            right_index, right_value, left_reference, left_buffer,
            coordinates, left_length, SampleStride, axis, dim, true
         );
      }

      __syncthreads();
      if (threadIdx.x < left_length) {
         buffer[x] = left_value;
         reference[x] = left_index;
      }
      if (threadIdx.x < right_length) {
         buffer[y] = right_value;
         reference[y] = right_index;
      }
   }

   __global__
   void cuMergeReferences(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      const int* left_limits,
      const int* right_limits,
      int sorted_size,
      int size,
      int axis,
      int dim
   )
   {
      const int i = static_cast<int>(blockIdx.x) & (sorted_size * 2 / SampleStride - 1);
      const int segment_base = (static_cast<int>(blockIdx.x) - i) * SampleStride;
      target_buffer += segment_base;
      target_reference += segment_base;
      source_buffer += segment_base;
      source_reference += segment_base;

      __shared__ int reference[SampleStride * 2];
      __shared__ node_type buffer[SampleStride * 2];
      __shared__ int left_start_source, right_start_source;
      __shared__ int left_start_target, right_start_target;
      __shared__ int left_length, right_length;

      if (threadIdx.x == 0) {
         const int left_elements = sorted_size;
         const int right_elements = min( sorted_size, size - (segment_base + sorted_size) );
         const int sample_num = getSampleNum( left_elements ) + getSampleNum( right_elements );
         const int left_end_source = i < sample_num - 1 ? left_limits[blockIdx.x + 1] : left_elements;
         const int right_end_source = i < sample_num - 1 ? right_limits[blockIdx.x + 1] : right_elements;
         left_start_source = left_limits[blockIdx.x];
         right_start_source = right_limits[blockIdx.x];
         left_length = left_end_source - left_start_source;
         right_length = right_end_source - right_start_source;
         left_start_target = left_start_source + right_start_source;
         right_start_target = left_start_target + left_length;
      }
      __syncthreads();

      if (threadIdx.x < left_length) {
         buffer[threadIdx.x] = source_buffer[left_start_source + threadIdx.x];
         reference[threadIdx.x] = source_reference[left_start_source + threadIdx.x];
      }
      if (threadIdx.x < right_length) {
         buffer[SampleStride + threadIdx.x] = source_buffer[sorted_size + right_start_source + threadIdx.x];
         reference[SampleStride + threadIdx.x] = source_reference[sorted_size + right_start_source + threadIdx.x];
      }
      __syncthreads();

      merge( reference, buffer, coordinates, left_length, right_length, axis, dim );
      __syncthreads();

      if (threadIdx.x < left_length) {
         target_buffer[left_start_target + threadIdx.x] = buffer[threadIdx.x];
         target_reference[left_start_target + threadIdx.x] = reference[threadIdx.x];
      }
      if (threadIdx.x < right_length) {
         target_buffer[right_start_target + threadIdx.x] = buffer[left_length + threadIdx.x];
         target_reference[right_start_target + threadIdx.x] = reference[left_length + threadIdx.x];
      }
   }

   void KdtreeCUDA::sortByAxis(int axis)
   {
      assert( Device.CoordinatesDevicePtr != nullptr );
      assert( Device.Reference[axis] != nullptr && Device.Reference[Dim] != nullptr );

      if (Device.Buffer[axis] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&Device.Buffer[axis]), sizeof( node_type ) * TupleNum )
         );
         cuCopyCoordinates<<<ThreadBlockNum, ThreadNum, 0, Device.Stream>>>(
            Device.Buffer[axis], Device.Reference[axis],
            Device.CoordinatesDevicePtr, TupleNum, axis, Dim
         );
         CHECK_KERNEL;
      }
      if (Device.Buffer[Dim] == nullptr) {
         CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Buffer[Dim]), sizeof( node_type ) * TupleNum ) );
      }

      int stage_num = 0;
      int* in_reference = nullptr;
      int* out_reference = nullptr;
      node_type* in_buffer = nullptr;
      node_type* out_buffer = nullptr;
      for (int step = SharedSize; step < TupleNum; step <<= 1) stage_num++;
      if (stage_num & 1) {
         in_buffer = Device.Sort.Buffer;
         in_reference = Device.Sort.Reference;
         out_buffer = Device.Buffer[Dim];
         out_reference = Device.Reference[Dim];
      }
      else {
         in_buffer = Device.Buffer[Dim];
         in_reference = Device.Reference[Dim];
         out_buffer = Device.Sort.Buffer;
         out_reference = Device.Sort.Reference;
      }

      assert( TupleNum <= SampleStride * Device.Sort.MaxSampleNum );

      int block_num = TupleNum / SharedSize;
      if (block_num > 0) {
         cuSortByBlock<<<block_num, SharedSize / 2, 0, Device.Stream>>>(
            in_reference, in_buffer,
            Device.Reference[axis], Device.Buffer[axis], Device.CoordinatesDevicePtr, TupleNum, axis, Dim
         );
         CHECK_KERNEL;
      }
      const int remained_size = TupleNum % SharedSize;
      if (remained_size > 0) {
         int buffer_index = 0;
         const int start_offset = TupleNum - remained_size;
         const std::array<node_type*, 2> buffers{ Device.Buffer[axis] + start_offset, in_buffer + start_offset };
         const std::array<int*, 2> references{ Device.Reference[axis] + start_offset, in_reference + start_offset };
         for (int sorted_size = 1; sorted_size < remained_size; sorted_size <<= 1) {
            cuSortLastBlock<<<1, divideUp( remained_size, sorted_size * 2 ), 0, Device.Stream>>>(
               references[buffer_index ^ 1], buffers[buffer_index ^ 1],
               references[buffer_index], buffers[buffer_index], Device.CoordinatesDevicePtr,
               sorted_size, remained_size, axis, Dim
            );
            CHECK_KERNEL;
            buffer_index ^= 1;
         }
         if (buffer_index == 0) {
            CHECK_CUDA(
               cudaMemcpyAsync(
                  buffers[1], buffers[0], sizeof( node_type ) * remained_size, cudaMemcpyDeviceToDevice, Device.Stream
               )
            );
            CHECK_CUDA(
               cudaMemcpyAsync(
                  references[1], references[0], sizeof( int ) * remained_size, cudaMemcpyDeviceToDevice, Device.Stream
               )
            );
         }
      }

      for (int sorted_size = SharedSize; sorted_size < TupleNum; sorted_size <<= 1) {
         constexpr int thread_num = SampleStride * 2;
         const int remained_threads = TupleNum % (sorted_size * 2);
         const int total_thread_num = remained_threads > sorted_size ?
            (TupleNum - remained_threads + sorted_size * 2) / thread_num : (TupleNum - remained_threads) / thread_num;
         block_num = divideUp( total_thread_num, thread_num );
         cuGenerateSampleRanks<<<block_num, thread_num, 0, Device.Stream>>>(
            Device.Sort.LeftRanks, Device.Sort.RightRanks,
            in_reference, in_buffer, Device.CoordinatesDevicePtr,
            sorted_size, TupleNum, axis, Dim, total_thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<block_num, thread_num, 0, Device.Stream>>>(
            Device.Sort.LeftLimits, Device.Sort.LeftRanks, sorted_size, TupleNum, total_thread_num
         );
         CHECK_KERNEL;

         cuMergeRanksAndIndices<<<block_num, thread_num, 0, Device.Stream>>>(
            Device.Sort.RightLimits, Device.Sort.RightRanks, sorted_size, TupleNum, total_thread_num
         );
         CHECK_KERNEL;

         const int merge_pairs = remained_threads > sorted_size ?
            getSampleNum( TupleNum ) : (TupleNum - remained_threads) / SampleStride;
         cuMergeReferences<<<merge_pairs, SampleStride, 0, Device.Stream>>>(
            out_reference, out_buffer,
            in_reference, in_buffer, Device.CoordinatesDevicePtr,
            Device.Sort.LeftLimits, Device.Sort.RightLimits,
            sorted_size, TupleNum, axis, Dim
         );
         CHECK_KERNEL;

         if (remained_threads <= sorted_size) {
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_reference + TupleNum - remained_threads, in_reference + TupleNum - remained_threads,
                  sizeof( int ) * remained_threads, cudaMemcpyDeviceToDevice, Device.Stream
               )
            );
            CHECK_CUDA(
               cudaMemcpyAsync(
                  out_buffer + TupleNum - remained_threads, in_buffer + TupleNum - remained_threads,
                  sizeof( node_type ) * remained_threads, cudaMemcpyDeviceToDevice, Device.Stream
               )
            );
         }

         std::swap( in_reference, out_reference );
         std::swap( in_buffer, out_buffer );
      }
   }

   __device__ int num_after_removal;
   __device__ int removal_error;

   __global__
   void cuRemoveDuplicates(
      int* unique_num_in_warp,
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const node_type* coordinates,
      int size_per_warp,
      int size,
      int axis,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int warp_index = index / warpSize;
      const int warp_lane = index & (warpSize - 1);
      const int offset = warp_index * size_per_warp;
      size_per_warp = min( size_per_warp, size - offset );

      __shared__ int reference[SharedSize];
      __shared__ node_type buffer[SharedSize];

      int* out_reference = target_reference + offset;
      node_type* out_buffer = target_buffer + offset;
      const int* in_reference = source_reference + offset;
      const node_type* in_buffer = source_buffer + offset;
      const int warps_per_block = warpSize / 2;
      const int shared_base = warpSize * 2 * (warp_index % warps_per_block);
      const int precede_mask = (1 << warp_lane) - 1;

      node_type t, v;
      int r, processed_size, unique_mask = 0;
      for (processed_size = 0; processed_size < size_per_warp && unique_mask == 0; processed_size += warpSize) {
         if (warp_lane < size_per_warp) {
            buffer[shared_base + warp_lane] = v = in_buffer[warp_lane];
            reference[shared_base + warp_lane] = r = in_reference[warp_lane];
            if (warp_lane > 0) {
               t = compareSuperKey(
                  v, buffer[shared_base + warp_lane - 1],
                  coordinates + r * dim, coordinates + reference[shared_base + warp_lane - 1] * dim,
                  axis, dim
               );
            }
            else if (warp_index > 0) {
               t = compareSuperKey(
                  v, *(in_buffer - 1),
                  coordinates + r * dim, coordinates + *(in_reference - 1) * dim,
                  axis, dim
               );
            }
            else t = 1;
         }
         else t = 0;

         if (t < 0) removal_error = -1;
         in_buffer += warpSize;
         in_reference += warpSize;

         unique_mask = static_cast<int>(__ballot_sync( 0xffffffff, t > 0 ));
         if (t > 0) {
            const int i = __popc( unique_mask & precede_mask );
            buffer[shared_base + i] = v;
            reference[shared_base + i] = r;
         }
      }

      int write_num = __popc( unique_mask );
      if (write_num == warpSize) {
         out_buffer[warp_lane] = buffer[shared_base + warp_lane];
         out_reference[warp_lane] = reference[shared_base + warp_lane];
      }

      const int shared_address_mask = warpSize * 2 - 1;
      for (; processed_size < size_per_warp; processed_size += warpSize) {
         if (processed_size + warp_lane < size_per_warp) {
            const int i = (write_num + warp_lane) & shared_address_mask;
            const int j = (write_num + warp_lane - 1) & shared_address_mask;
            buffer[shared_base + i] = v = in_buffer[warp_lane];
            reference[shared_base + i] = r = in_reference[warp_lane];
            t = compareSuperKey(
               v, buffer[shared_base + j],
               coordinates + r * dim, coordinates + reference[shared_base + j] * dim,
               axis, dim
            );
         }
         else t = 0;

         if (t < 0) removal_error = -1;
         in_buffer += warpSize;
         in_reference += warpSize;

         unique_mask = static_cast<int>(__ballot_sync( 0xffffffff, t > 0 ));
         if (t > 0) {
            const int i = (write_num + __popc( unique_mask & precede_mask )) & shared_address_mask;
            buffer[shared_base + i] = v;
            reference[shared_base + i] = r;
         }

         const int n = __popc( unique_mask );
         if (((write_num ^ (write_num + n)) & warpSize) != 0) {
            const int i = (write_num & ~(warpSize - 1)) + warp_lane;
            out_buffer[i] = buffer[shared_base + (write_num & warpSize) + warp_lane];
            out_reference[i] = reference[shared_base + (write_num & warpSize) + warp_lane];
         }
         write_num += n;
      }

      if (warp_lane < (write_num & (warpSize - 1))) {
         const int i = (write_num & ~(warpSize - 1)) + warp_lane;
         out_buffer[i] = buffer[shared_base + (write_num & warpSize) + warp_lane];
         out_reference[i] = reference[shared_base + (write_num & warpSize) + warp_lane];
      }

      if (warp_lane == 0 && unique_num_in_warp != nullptr) unique_num_in_warp[warp_index] = write_num;
   }

   __global__
   void cuRemoveGaps(
      int* target_reference,
      node_type* target_buffer,
      const int* source_reference,
      const node_type* source_buffer,
      const int* unique_num_in_warp,
      int size_per_warp,
      int size
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int warp_index = index / warpSize;
      const int warp_lane = index & (warpSize - 1);
      const int offset = warp_index * size_per_warp;

      int target_offset = 0, unique_num_in_this_warp = 0;
      if (warp_lane == 0) {
         for (int i = 0; i < warp_index; ++i) target_offset += unique_num_in_warp[i];
         unique_num_in_this_warp = unique_num_in_warp[warp_index];
      }
      target_offset = __shfl_sync( 0xffffffff, target_offset, 0 );
      unique_num_in_this_warp = __shfl_sync( 0xffffffff, unique_num_in_this_warp, 0 );

      source_buffer += offset;
      source_reference += offset;
      target_buffer += target_offset;
      target_reference += target_offset;
      for (int i = warp_lane; i < unique_num_in_this_warp; i += warpSize) {
         target_buffer[i] = source_buffer[i];
         target_reference[i] = source_reference[i];
      }

      if (warp_lane == 0 && offset + size_per_warp >= size) num_after_removal = target_offset + unique_num_in_this_warp;
   }

   int KdtreeCUDA::removeDuplicates(int axis) const
   {
      const int source_index = Dim;
      const int target_index = axis;

      assert( Device.Buffer[source_index] != nullptr && Device.Buffer[target_index] != nullptr );
      assert( Device.Reference[source_index] != nullptr && Device.Reference[target_index] != nullptr );

      int error = 0;
      CHECK_CUDA(
         cudaMemcpyToSymbolAsync(
            removal_error, &error, sizeof( removal_error ), 0,
            cudaMemcpyHostToDevice, Device.Stream
         )
      );

      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSize, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSize / 2 );
      constexpr int segment = total_thread_num / WarpSize;
      const int size_per_warp = divideUp( TupleNum, segment );

      int* unique_num_in_warp = nullptr;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&unique_num_in_warp), sizeof( int ) * segment ) );
      cuRemoveDuplicates<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
         unique_num_in_warp, Device.Sort.Reference, Device.Sort.Buffer,
         Device.Reference[source_index], Device.Buffer[source_index],
         Device.CoordinatesDevicePtr, size_per_warp, TupleNum, axis, Dim
      );
      CHECK_KERNEL;

      cuRemoveGaps<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
         Device.Reference[target_index], Device.Buffer[target_index],
         Device.Sort.Reference, Device.Sort.Buffer,
         unique_num_in_warp, size_per_warp, TupleNum
      );
      CHECK_KERNEL;

      CHECK_CUDA( cudaFree( unique_num_in_warp ) );

      CHECK_CUDA(
         cudaMemcpyFromSymbolAsync( &error, removal_error, sizeof( error ), 0, cudaMemcpyDeviceToHost, Device.Stream )
      );
      if (error != 0) {
         std::ostringstream buffer;
         buffer << "error in removeDuplicates(): " << error << "\n";
         throw std::runtime_error( buffer.str() );
      }

      int num = 0;
      CHECK_CUDA(
         cudaMemcpyFromSymbolAsync( &num, num_after_removal, sizeof( num ), 0, cudaMemcpyDeviceToHost, Device.Stream )
      );
      return num;
   }

   void KdtreeCUDA::sort(std::vector<int>& end)
   {
      const int max_sample_num = TupleNum / SampleStride + 1;
      Device.Sort.MaxSampleNum = max_sample_num;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.LeftRanks), sizeof( int ) * max_sample_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.RightRanks), sizeof( int ) * max_sample_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.LeftLimits), sizeof( int ) * max_sample_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.RightLimits), sizeof( int ) * max_sample_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.Reference), sizeof( int ) * TupleNum ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.Sort.Buffer), sizeof( node_type ) * TupleNum ) );

      for (int axis = 0; axis < Dim; ++axis) {
         initializeReference( axis );
         sortByAxis( axis );
         end[axis] = removeDuplicates( axis );
      }
      Device.TupleNum = end[0];
      CHECK_CUDA( cudaStreamSynchronize( Device.Stream ) );

      CHECK_CUDA( cudaFree( Device.Sort.LeftRanks ) );
      CHECK_CUDA( cudaFree( Device.Sort.RightRanks ) );
      CHECK_CUDA( cudaFree( Device.Sort.LeftLimits ) );
      CHECK_CUDA( cudaFree( Device.Sort.RightLimits ) );
      CHECK_CUDA( cudaFree( Device.Sort.Reference ) );
      CHECK_CUDA( cudaFree( Device.Sort.Buffer ) );
      for (int axis = 0; axis <= Dim; ++axis) CHECK_CUDA( cudaFree( Device.Buffer[axis] ) );
   }

   __device__
   void partition(
      int* target_left_reference,
      int* target_right_reference,
      int* left_child_num_in_warp,
      int* right_child_num_in_warp,
      const int* source_reference,
      const node_type* __restrict__ coordinates,
      int mid_reference,
      int size_per_warp,
      int partition_size,
      int axis,
      int dim,
      int warp_num_per_node
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int warp_index = (index / warpSize) & (warp_num_per_node - 1);
      const int warp_lane = index & (warpSize - 1);
      const int offset = warp_index * size_per_warp;
      size_per_warp = min( size_per_warp, partition_size - offset );

      __shared__ int left_reference[SharedSize];
      __shared__ int right_reference[SharedSize];

      int* out_left_reference = target_left_reference + offset;
      int* out_right_reference = target_right_reference + offset;
      const int* in_reference = source_reference + offset;
      const int warps_per_block = warpSize / 2;
      const int shared_base = warpSize * 2 * ((index / warpSize) % warps_per_block);
      const int shared_address_mask = warpSize * 2 - 1;
      const int precede_mask = (1 << warp_lane) - 1;

      node_type t;
      int r, left_write_num = 0, right_write_num = 0;
      for (int processed_size = 0; processed_size < size_per_warp; processed_size += warpSize) {
         if (processed_size + warp_lane < size_per_warp) {
            r = in_reference[warp_lane];
            t = compareSuperKey(
               coordinates[r * dim + axis], coordinates[mid_reference * dim + axis],
               coordinates + r * dim, coordinates + mid_reference * dim, axis, dim
            );
         }
         else t = 0;
         in_reference += warpSize;

         int unique_mask = static_cast<int>(__ballot_sync( 0xffffffff, t < 0 ));
         if (t < 0) {
            const int i = (left_write_num + __popc( unique_mask & precede_mask )) & shared_address_mask;
            left_reference[shared_base + i] = r;
         }

         int n = __popc( unique_mask );
         if (((left_write_num ^ (left_write_num + n)) & warpSize) != 0) {
            const int i = (left_write_num & ~(warpSize - 1)) + warp_lane;
            out_left_reference[i] = left_reference[shared_base + (left_write_num & warpSize) + warp_lane];
         }
         left_write_num += n;

         unique_mask = static_cast<int>(__ballot_sync( 0xffffffff, t > 0 ));
         if (t > 0) {
            const int i = (right_write_num + __popc( unique_mask & precede_mask )) & shared_address_mask;
            right_reference[shared_base + i] = r;
         }

         n = __popc( unique_mask );
         if (((right_write_num ^ (right_write_num + n)) & warpSize) != 0) {
            const int i = (right_write_num & ~(warpSize - 1)) + warp_lane;
            out_right_reference[i] = right_reference[shared_base + (right_write_num & warpSize) + warp_lane];
         }
         right_write_num += n;
      }

      if (warp_lane < (left_write_num & (warpSize - 1))) {
         const int i = (left_write_num & ~(warpSize - 1)) + warp_lane;
         out_left_reference[i] = left_reference[shared_base + (left_write_num & warpSize) + warp_lane];
      }
      if (warp_lane < (right_write_num & (warpSize - 1))) {
         const int i = (right_write_num & ~(warpSize - 1)) + warp_lane;
         out_right_reference[i] = right_reference[shared_base + (right_write_num & warpSize) + warp_lane];
      }

      if (warp_lane == 0 && left_child_num_in_warp != nullptr) left_child_num_in_warp[warp_index] = left_write_num;
      if (warp_lane == 0 && right_child_num_in_warp != nullptr) right_child_num_in_warp[warp_index] = right_write_num;
   }

   __global__
   void cuPartition(
      KdtreeNode* root,
      int* left_child_num_in_warp,
      int* right_child_num_in_warp,
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
      const auto total_warp_num = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_num_per_node = total_warp_num >> depth;
      const int warp_index = index / warpSize;
      const int warp_lane = index & (warpSize - 1);

      int mid = start + (end - start) / 2;
      for (int i = 1; i <= depth; ++i) {
         if (warp_index & (total_warp_num >> i)) start = mid + 1;
         else end = mid - 1;
         mid = start + (end - start) / 2;
      }

      const int partition_size = end - start + 1;
      const int size_per_warp = divideUp( partition_size, warp_num_per_node );
      const int mid_reference = primary_reference[mid];
      partition(
         target_left_reference + start, target_right_reference + start,
         left_child_num_in_warp + (warp_index & ~(warp_num_per_node - 1)),
         right_child_num_in_warp + (warp_index & ~(warp_num_per_node - 1)),
         source_reference + start, coordinates,
         mid_reference, size_per_warp, partition_size, axis, dim, warp_num_per_node
      );

      if (warp_lane == 0) {
         const int m = warp_index / warp_num_per_node;
         mid_references[m] = mid_reference;
         if (last_mid_references != nullptr) {
            if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
            else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
            root[mid_reference].ParentIndex = root[last_mid_references[m >> 1]].Index;
         }
      }
   }

   __global__
   void cuRemovePartitionGaps(
      int* target_reference,
      const int* source_left_reference,
      const int* source_right_reference,
      const int* left_child_num_in_warp,
      const int* right_child_num_in_warp,
      int start,
      int end,
      int depth
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto total_warp_num = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_num_per_node = total_warp_num >> depth;
      const int warp_lane = index & (warpSize - 1);
      const int warp_index = index / warpSize;

      int mid = start + (end - start) / 2;
      for (int i = 1; i <= depth; ++i) {
         if (warp_index & (total_warp_num >> i)) start = mid + 1;
         else end = mid - 1;
         mid = start + (end - start) / 2;
      }

      const int partition_size = end - start + 1;
      const int size_per_warp = divideUp( partition_size, warp_num_per_node );
      const int offset = start + size_per_warp * (warp_index - (warp_index & ~(warp_num_per_node - 1)));

      int target_offset = start, child_num_in_this_warp = 0;
      if (warp_lane == 0) {
         for (int i = warp_index & ~(warp_num_per_node - 1); i < warp_index; ++i) {
            target_offset += left_child_num_in_warp[i];
         }
         child_num_in_this_warp = left_child_num_in_warp[warp_index];
      }
      target_offset = __shfl_sync( 0xffffffff, target_offset, 0 );
      child_num_in_this_warp = __shfl_sync( 0xffffffff, child_num_in_this_warp, 0 );
      for (int i = warp_lane; i < child_num_in_this_warp; i += warpSize) {
         target_reference[target_offset + i] = source_left_reference[offset + i];
      }

      target_offset = mid + 1;
      if (warp_lane == 0) {
         for (int i = warp_index & ~(warp_num_per_node - 1); i < warp_index; ++i) {
            target_offset += right_child_num_in_warp[i];
         }
         child_num_in_this_warp = right_child_num_in_warp[warp_index];
      }
      target_offset = __shfl_sync( 0xffffffff, target_offset, 0 );
      child_num_in_this_warp = __shfl_sync( 0xffffffff, child_num_in_this_warp, 0 );
      for (int i = warp_lane; i < child_num_in_this_warp; i += warpSize) {
         target_reference[target_offset + i] = source_right_reference[offset + i];
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
      int max_controllable_depth_for_warp
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto total_warp_num = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_lane = index & (warpSize - 1);
      const int warp_index = index / warpSize;
      const int loop_levels = depth - max_controllable_depth_for_warp;
      for (int loop = 0; loop < (1 << loop_levels); ++loop) {
         int s = start, e = end, mid;
         for (int i = 1; i <= loop_levels; ++i) {
            mid = s + (e - s) / 2;
            if (loop & (1 << (loop_levels - i))) s = mid + 1;
            else e = mid - 1;
         }
         for (int i = 1; i <= max_controllable_depth_for_warp; ++i) {
            mid = s + (e - s) / 2;
            if (warp_index & (total_warp_num >> i)) s = mid + 1;
            else e = mid - 1;
         }
         mid = s + (e - s) / 2;

         const int partition_size = e - s + 1;
         const int mid_reference = primary_reference[mid];
         partition(
            target_reference + s, target_reference + mid + 1, nullptr, nullptr,
            source_reference + s, coordinates,
            mid_reference, partition_size, partition_size, axis, dim, 1
         );

         if (warp_lane == 0) {
            const int m = warp_index + total_warp_num * loop;
            mid_references[m] = mid_reference;
            if (last_mid_references != nullptr) {
               if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
               else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
               root[mid_reference].ParentIndex = root[last_mid_references[m >> 1]].Index;
            }
         }
      }
   }

   __global__
   void cuCopyReference(int* target_reference, const int* source_reference, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) target_reference[i] = source_reference[i];
   }

   void KdtreeCUDA::partitionDimension(int axis, int depth)
   {
      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSize, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSize / 2 );
      constexpr int warp_num = total_thread_num / WarpSize;
      const auto max_controllable_depth_for_warp =
         static_cast<int>(std::floor( std::log2( static_cast<double>(warp_num) ) ));

      int* mid_references = Device.MidReferences[depth & 1];
      const int* last_mid_references = depth == 0 ? nullptr : Device.MidReferences[(depth - 1) & 1];
      if (depth < max_controllable_depth_for_warp) {
         for (int i = 1; i < Dim; ++i) {
            int r = i + axis;
            r = r < Dim ? r : r - Dim;
            cuPartition<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
               Device.Root, Device.LeftChildNumInWarp, Device.RightChildNumInWarp,
               Device.Reference[Dim], Device.Reference[Dim + 1], mid_references,
               last_mid_references, Device.Reference[r], Device.Reference[axis],
               Device.CoordinatesDevicePtr, 0, Device.TupleNum - 1, axis, Dim, depth
            );
            CHECK_KERNEL;

            cuRemovePartitionGaps<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
               Device.Reference[r],
               Device.Reference[Dim], Device.Reference[Dim + 1],
               Device.LeftChildNumInWarp, Device.RightChildNumInWarp,
               0, Device.TupleNum - 1, depth
            );
            CHECK_KERNEL;
         }
      }
      else {
         for (int i = 1; i < Dim; ++i) {
            int r = i + axis;
            r = r < Dim ? r : r - Dim;
            cuPartition<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
               Device.Root, Device.Reference[Dim], mid_references,
               last_mid_references, Device.Reference[r], Device.Reference[axis],
               Device.CoordinatesDevicePtr, 0, Device.TupleNum - 1, axis, Dim, depth, max_controllable_depth_for_warp
            );
            CHECK_KERNEL;

            cuCopyReference<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
               Device.Reference[r], Device.Reference[Dim], Device.TupleNum
            );
            CHECK_KERNEL;
         }
      }

      if (depth == 0) {
         CHECK_CUDA(
            cudaMemcpyAsync(
               &Device.RootNode, Device.MidReferences[0], sizeof( int ), cudaMemcpyDeviceToHost, Device.Stream
            )
         );
         assert( Device.RootNode != -1 );
      }
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
      const auto total_warp_num = static_cast<int>(blockDim.x * gridDim.x / warpSize);
      const int warp_index = index / warpSize;

      for (int i = 1; i <= depth; ++i) {
         const int mid = start + (end - start) / 2;
         if (warp_index & (total_warp_num >> i)) start = mid + 1;
         else end = mid - 1;
      }

      int mid_reference = -1;
      if (end == start) mid_reference = primary_reference[end];
      else if (end == start + 1) {
         mid_reference = primary_reference[start];
         root[mid_reference].RightChildIndex = primary_reference[end];
         root[root[mid_reference].RightChildIndex].ParentIndex = root[mid_reference].Index;
      }
      else if (end == start + 2) {
         mid_reference = primary_reference[start + 1];
         root[mid_reference].LeftChildIndex = primary_reference[start];
         root[mid_reference].RightChildIndex = primary_reference[end];
         root[root[mid_reference].LeftChildIndex].ParentIndex = root[mid_reference].Index;
         root[root[mid_reference].RightChildIndex].ParentIndex = root[mid_reference].Index;
      }

      if (mid_reference != -1) {
         const int warp_num_per_node = total_warp_num >> depth;
         const int m = warp_index / warp_num_per_node;
         mid_references[m] = mid_reference;
         if (m & 1) root[last_mid_references[m >> 1]].RightChildIndex = mid_reference;
         else root[last_mid_references[m >> 1]].LeftChildIndex = mid_reference;
         root[mid_reference].ParentIndex = root[last_mid_references[m >> 1]].Index;
      }
   }

   void KdtreeCUDA::partitionDimensionFinal(int axis, int depth)
   {
      constexpr int total_thread_num = ThreadBlockNum * ThreadNum;
      constexpr int block_num = std::max( total_thread_num * 2 / SharedSize, 1 );
      constexpr int thread_num_per_block = std::min( total_thread_num, SharedSize / 2 );
      constexpr int warp_num = total_thread_num / WarpSize;
      const auto max_controllable_depth_for_warp =
         static_cast<int>(std::floor( std::log2( static_cast<double>(warp_num) ) ));
      const int loop_levels = std::max( depth - max_controllable_depth_for_warp, 0 );

      int* mid_references = Device.MidReferences[depth & 1];
      const int* last_mid_references = Device.MidReferences[(depth - 1) & 1];
      for (int loop = 0; loop < (1 << loop_levels); ++loop) {
         int start = 0, end = Device.TupleNum - 1;
         for (int i = 1; i <= loop_levels; ++i) {
            const int mid = start + (end - start) / 2;
            if (loop & (1 << (loop_levels - i))) start = mid + 1;
            else end = mid - 1;
         }

         cuPartitionFinal<<<block_num, thread_num_per_block, 0, Device.Stream>>>(
            Device.Root,
            mid_references + loop * warp_num,
            last_mid_references + loop * warp_num / 2,
            Device.Reference[axis],
            start, end, depth - loop_levels
         );
         CHECK_KERNEL;
      }
   }

   void KdtreeCUDA::build()
   {
      constexpr int warp_num = ThreadBlockNum * ThreadNum / WarpSize;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.LeftChildNumInWarp), sizeof( int ) * warp_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.RightChildNumInWarp), sizeof( int ) * warp_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.MidReferences[0]), sizeof( int ) * Device.TupleNum ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.MidReferences[1]), sizeof( int ) * Device.TupleNum ) );

      assert( !Device.Reference.empty() );
      for (int axis = 0; axis < Dim; ++axis) assert( Device.Reference[axis] != nullptr );

      if (Device.Reference[Dim] == nullptr) {
         CHECK_CUDA(
            cudaMalloc( reinterpret_cast<void**>(&Device.Reference[Dim]), sizeof( int ) * Device.TupleNum )
         );
      }

      const auto depth = static_cast<int>(std::floor( std::log2( static_cast<double>(Device.TupleNum) ) ));
      for (int i = 0; i < depth - 1; ++i) {
         partitionDimension( i % Dim, i );
      }
      partitionDimensionFinal( (depth - 1) % Dim, depth - 1 );

      CHECK_CUDA( cudaStreamSynchronize( Device.Stream ) );
      CHECK_CUDA( cudaFree( Device.LeftChildNumInWarp ) );
      CHECK_CUDA( cudaFree( Device.RightChildNumInWarp ) );
      CHECK_CUDA( cudaFree( Device.MidReferences[0] ) );
      CHECK_CUDA( cudaFree( Device.MidReferences[1] ) );
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
      __shared__ int sums[SharedSize];

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
            if (right >= 0) {
               if (compareSuperKey(
                     coordinates + root[right].Index * dim, coordinates + root[node].Index * dim, axis, dim
                   ) <= 0) {
                  verify_error = 1;
               }
               if (root[right].ParentIndex != root[node].Index) verify_error = 1;
            }

            const int left = root[node].LeftChildIndex;
            next_child[i * 2] = left;
            if (left >= 0) {
               if (compareSuperKey(
                     coordinates + root[left].Index * dim, coordinates + root[node].Index * dim, axis, dim
                   ) >= 0) {
                  verify_error = 1;
               }
               if (root[left].ParentIndex != root[node].Index) verify_error = 1;
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
         if (blockDim.x >= warpSize * 2) count += sums[id + warpSize];
         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            count += __shfl_down_sync( 0xffffffff, count, offset );
         }
      }

      if (id == 0) node_sums[blockIdx.x] += count;
   }

   __global__
   void cuSumNodeNum(int* node_sums)
   {
      __shared__ int sums[SharedSize];

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
         if (blockDim.x >= warpSize * 2) sum += sums[id + warpSize];
         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync( 0xffffffff, sum, offset );
         }
      }

      if (id == 0) node_sums[blockIdx.x] = sum;
   }

   int KdtreeCUDA::verify(int start_axis) const
   {
      const auto log_size = static_cast<int>(std::floor( std::log2( static_cast<double>(Device.TupleNum) ) ));
      CHECK_CUDA(
         cudaMemcpyAsync(
            Device.MidReferences[0], &Device.RootNode, sizeof( int ), cudaMemcpyHostToDevice, Device.Stream
         )
      );

      int error = 0;
      CHECK_CUDA(
         cudaMemcpyToSymbolAsync( verify_error, &error, sizeof( error ), 0, cudaMemcpyHostToDevice, Device.Stream )
      );

      int* child;
      int* next_child;
      for (int i = 0; i <= log_size; ++i) {
         const int needed_threads = 1 << i;
         const int block_num = std::clamp( needed_threads / ThreadNum, 1, ThreadBlockNum );
         const int axis = (i + start_axis) % Dim;
         child = Device.MidReferences[i & 1];
         next_child = Device.MidReferences[(i + 1) & 1];
         cuVerify<<<block_num, ThreadNum, 0, Device.Stream>>>(
            Device.NodeSums, next_child,
            child, Device.Root, Device.CoordinatesDevicePtr, needed_threads, axis, Dim
         );
         CHECK_KERNEL;

         CHECK_CUDA(
            cudaMemcpyFromSymbolAsync( &error, verify_error, sizeof( error ), 0, cudaMemcpyDeviceToHost, Device.Stream )
         );
         CHECK_CUDA( cudaStreamSynchronize( Device.Stream ) );
         if (error != 0) {
            std::ostringstream buffer;
            buffer << "error in verify(): " << error << "\n";
            throw std::runtime_error( buffer.str() );
         }
      }

      cuSumNodeNum<<<1, ThreadNum, 0, Device.Stream>>>( Device.NodeSums );
      CHECK_KERNEL;

      int node_num = 0;
      CHECK_CUDA( cudaMemcpyAsync( &node_num, Device.NodeSums, sizeof( int ), cudaMemcpyDeviceToHost, Device.Stream ) );
      return node_num;
   }

   int KdtreeCUDA::verify()
   {
      CHECK_CUDA(
         cudaMalloc( reinterpret_cast<void**>(&Device.MidReferences[0]), sizeof( int ) * 2 * Device.TupleNum )
      );
      CHECK_CUDA(
         cudaMalloc( reinterpret_cast<void**>(&Device.MidReferences[1]), sizeof( int ) * 2 * Device.TupleNum )
      );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.NodeSums), sizeof( int ) * ThreadBlockNum ) );
      CHECK_CUDA( cudaMemset( Device.NodeSums, 0, sizeof( int ) * ThreadBlockNum ) );

      const int node_num = verify( 0 );
      CHECK_CUDA( cudaStreamSynchronize( Device.Stream ) );
      CHECK_CUDA( cudaFree( Device.MidReferences[0] ) );
      CHECK_CUDA( cudaFree( Device.MidReferences[1] ) );
      CHECK_CUDA( cudaFree( Device.NodeSums ) );
      return node_num;
   }

   void KdtreeCUDA::create()
   {
      initialize( Coordinates, TupleNum );
      CHECK_CUDA( cudaStreamSynchronize( Device.Stream ) );

      auto start_time = std::chrono::steady_clock::now();
      std::vector<int> end(Dim);
      sort( end );
      auto end_time = std::chrono::steady_clock::now();
      const auto sort_time =
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

      for (int i = 0; i < Dim - 1; ++i) {
         assert( end[i] >= 0 );
         for (int j = i + 1; j < Dim; ++j) assert( end[i] == end[j] );
      }

      start_time = std::chrono::steady_clock::now();
      build();
      end_time = std::chrono::steady_clock::now();
      const auto build_time =
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;

      start_time = std::chrono::steady_clock::now();
      NodeNum = verify();
      end_time = std::chrono::steady_clock::now();
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

      const node_type* tuple = Coordinates + kd_nodes[index].Index * Dim;
      std::cout << "(" << tuple[0] << ",";
      for (int i = 1; i < Dim - 1; ++i) std::cout << tuple[i] << ",";
      std::cout << tuple[Dim - 1] << ")\n";

      if (kd_nodes[index].LeftChildIndex >= 0) print( kd_nodes, kd_nodes[index].LeftChildIndex, depth + 1 );
   }

   void KdtreeCUDA::print() const
   {
      if (Device.RootNode < 0 || Coordinates == nullptr) return;

      std::vector<KdtreeNode> kd_nodes(TupleNum);
      CHECK_CUDA(
         cudaMemcpyAsync(
            kd_nodes.data(), Device.Root, sizeof( KdtreeNode ) * TupleNum, cudaMemcpyDeviceToHost, Device.Stream
         )
      );

      print( kd_nodes, Device.RootNode, 0 );
   }

   void KdtreeCUDA::getResult(
      std::vector<node_type>& output,
      const std::vector<KdtreeNode>& kd_nodes,
      int index,
      int depth
   ) const
   {
      if (kd_nodes[index].RightChildIndex >= 0) {
         getResult( output, kd_nodes, kd_nodes[index].RightChildIndex, depth + 1 );
      }

      const node_type* tuple = Coordinates + kd_nodes[index].Index * Dim;
      output.emplace_back( tuple[0] );
      for (int i = 1; i < Dim - 1; ++i) output.emplace_back( tuple[i] );
      output.emplace_back( tuple[Dim - 1] );

      if (kd_nodes[index].LeftChildIndex >= 0) getResult( output, kd_nodes, kd_nodes[index].LeftChildIndex, depth + 1 );
   }

   void KdtreeCUDA::getResult(std::vector<node_type>& output) const
   {
      if (Device.RootNode < 0 || Coordinates == nullptr) return;

      std::vector<KdtreeNode> kd_nodes(TupleNum);
      CHECK_CUDA(
         cudaMemcpyAsync(
            kd_nodes.data(), Device.Root, sizeof( KdtreeNode ) * TupleNum, cudaMemcpyDeviceToHost, Device.Stream
         )
      );

      getResult( output, kd_nodes, Device.RootNode, 0 );
   }

   __device__
   void findQueryWithoutStack(
      int* lists,
      int* list_lengths,
      const KdtreeNode* root,
      const node_type* coordinates,
      const node_type* queries,
      node_type search_radius,
      int node_index,
      int query_num,
      int size,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      if (index >= query_num) return;

      int depth = 0;
      int prev = -1;
      int curr = node_index;
      while (curr >= 0) {
         const KdtreeNode* node = &root[curr];
         const int parent = node->ParentIndex;
         if (curr >= size) {
            prev = curr;
            curr = parent;
            continue;
         }

         const node_type* query = queries + index * dim;
         const node_type* point = coordinates + root[curr].Index * dim;
         const bool from_child = prev >= 0 && (prev == node->LeftChildIndex || prev == node->RightChildIndex);
         if (!from_child) {
            bool inside = true;
            for (int d = 0; d < dim; ++d) {
               if (query[d] - search_radius > point[d] || query[d] + search_radius < point[d]) {
                  inside = false;
                  break;
               }
            }
            if (inside) {
               lists[index * size + list_lengths[index]] = node->Index;
               list_lengths[index]++;
            }
         }

         const int axis = depth % dim;
         const bool search_left = node->LeftChildIndex >= 0 &&
            compareSuperKey( point, query, -search_radius, axis, dim ) >= 0;
         const bool search_right = node->RightChildIndex >= 0 &&
            compareSuperKey( point, query, search_radius, axis, dim ) <= 0;

         int next = -1;
         if (search_left && prev == node->LeftChildIndex) {
            if (search_right) {
               next = node->RightChildIndex;
               depth++;
            }
            else {
               next = parent;
               depth--;
            }
         }
         else if (search_right && prev == node->RightChildIndex) {
            next = parent;
            depth--;
         }
         else if (prev < 0 || prev == parent) {
            if (!search_left && !search_right) {
               next = parent;
               depth--;
            }
            else if (!search_left) {
               next = node->RightChildIndex;
               depth++;
            }
            else {
               next = node->LeftChildIndex;
               depth++;
            }
         }

         prev = curr;
         curr = next;
      }
   }

   __device__
   void findQuery(
      int* lists,
      int* list_lengths,
      const KdtreeNode* root,
      const node_type* coordinates,
      const node_type* queries,
      node_type search_radius,
      int node_index,
      int size,
      int dim,
      int mask
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int warp_lane = index & (warpSize - 1);
      const int max_node_num_in_stack = SharedSize * 8 + 1;

      int list_ptr = 0, depth = 0;
      int mask_list[2][max_node_num_in_stack];
      const KdtreeNode* visit_list[2][max_node_num_in_stack];
      mask_list[list_ptr][0] = mask;
      visit_list[list_ptr][0] = &root[node_index];
      visit_list[list_ptr][1] = nullptr;
      while (visit_list[list_ptr][0] != nullptr) {
         int child_num = 0;
         const int axis = depth % dim;
         for (int i = 0; visit_list[list_ptr][i] != nullptr; ++i) {
            bool search_left = false, search_right = false;
            const bool active = (mask_list[list_ptr][i] >> warp_lane) & 1;
            if (active) {
               bool inside = true;
               const node_type* query = queries + index * dim;
               const node_type* node = coordinates + visit_list[list_ptr][i]->Index * dim;
               for (int d = 0; d < dim; ++d) {
                  if (query[d] - search_radius > node[d] || query[d] + search_radius < node[d]) {
                     inside = false;
                     break;
                  }
               }
               if (inside) {
                  lists[index * size + list_lengths[index]] = visit_list[list_ptr][i]->Index;
                  list_lengths[index]++;
               }

               search_left = visit_list[list_ptr][i]->LeftChildIndex >= 0 &&
                  compareSuperKey( node, query, -search_radius, axis, dim ) >= 0;
               search_right = visit_list[list_ptr][i]->RightChildIndex >= 0 &&
                  compareSuperKey( node, query, search_radius, axis, dim ) <= 0;
            }

            const int left_mask = static_cast<int>(__ballot_sync( 0xffffffff, search_left ));
            if (__popc( left_mask ) != 0) {
               mask_list[list_ptr ^ 1][child_num] = left_mask;
               visit_list[list_ptr ^ 1][child_num++] = visit_list[list_ptr][i]->LeftChildIndex >= 0 ?
                  &root[visit_list[list_ptr][i]->LeftChildIndex] : &root[node_index];
            }
            const int right_mask = static_cast<int>(__ballot_sync( 0xffffffff, search_right ));
            if (__popc( right_mask ) != 0) {
               mask_list[list_ptr ^ 1][child_num] = right_mask;
               visit_list[list_ptr ^ 1][child_num++] = visit_list[list_ptr][i]->RightChildIndex >= 0 ?
                  &root[visit_list[list_ptr][i]->RightChildIndex] : &root[node_index];
            }
         }
         depth++;
         list_ptr ^= 1;
         visit_list[list_ptr][child_num] = nullptr;
      }
   }

   template<bool small_tree = true>
   __global__
   void cuSearch(
      int* lists,
      int* list_lengths,
      const KdtreeNode* root,
      const node_type* coordinates,
      const node_type* queries,
      node_type search_radius,
      int node_index,
      int query_num,
      int size,
      int dim
   )
   {
      auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      while (true) {
         const int mask = static_cast<int>(__ballot_sync( 0xffffffff, index < query_num ));
         if (__popc( mask ) == 0) break;

         if (small_tree) {
            findQuery( lists, list_lengths, root, coordinates, queries, search_radius, node_index, size, dim, mask );
         }
         else {
            findQueryWithoutStack(
               lists, list_lengths,
               root, coordinates, queries, search_radius, node_index, query_num, size, dim
            );
         }
         index += step;
      }
   }

   void KdtreeCUDA::search(
      std::vector<std::vector<int>>& founds,
      const node_type* queries,
      int query_num,
      node_type search_radius
   ) const
   {
      if (Device.RootNode < 0 || Coordinates == nullptr) return;

      int* lists = nullptr;
      int* list_lengths = nullptr;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&lists), sizeof( int ) * Device.TupleNum * query_num ) );
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&list_lengths), sizeof( int ) * query_num ) );
      CHECK_CUDA( cudaMemset( list_lengths, 0, sizeof( int ) * query_num ) );

      node_type* device_queries = nullptr;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device_queries), sizeof( node_type ) * query_num * Dim ) );
      CHECK_CUDA(
         cudaMemcpyAsync(
            device_queries, queries, sizeof( node_type ) * query_num * Dim,
            cudaMemcpyHostToDevice, Device.Stream
         )
      );

      const int max_depth = static_cast<int>(std::log2( static_cast<double>(Device.TupleNum) ));
      const int max_node_num = std::max( 1 << (max_depth - 1), Device.TupleNum - (1 << max_depth) + 1 );
      constexpr int max_node_num_in_stack = SharedSize * 8;
      if (max_node_num <= max_node_num_in_stack) {
         cuSearch<true><<<divideUp( query_num, WarpSize ), WarpSize, 0, Device.Stream>>>(
            lists, list_lengths,
            Device.Root, Device.CoordinatesDevicePtr, device_queries,
            search_radius, Device.RootNode, query_num, Device.TupleNum, Dim
         );
         CHECK_KERNEL;
      }
      else {
         cuSearch<false><<<divideUp( query_num, WarpSize ), WarpSize, 0, Device.Stream>>>(
            lists, list_lengths,
            Device.Root, Device.CoordinatesDevicePtr, device_queries,
            search_radius, Device.RootNode, query_num, Device.TupleNum, Dim
         );
         CHECK_KERNEL;
      }

      std::vector<int> lengths(query_num);
      CHECK_CUDA(
         cudaMemcpyAsync(
            lengths.data(), list_lengths, sizeof( int ) * query_num, cudaMemcpyDeviceToHost, Device.Stream
         )
      );

      founds.clear();
      founds.resize( query_num );
      for (int i = 0; i < query_num; ++i) {
         if (lengths[i] <= 0) continue;

         founds[i].resize( lengths[i] );
         CHECK_CUDA(
            cudaMemcpyAsync(
               founds[i].data(), lists + Device.TupleNum * i, sizeof( int ) * lengths[i],
               cudaMemcpyDeviceToHost, Device.Stream
            )
         );
      }

      CHECK_CUDA( cudaFree( lists ) );
      CHECK_CUDA( cudaFree( list_lengths ) );
      CHECK_CUDA( cudaFree( device_queries ) );
   }

   __global__
   void cuInitializeKNN(uint64_t* lists, int query_num, int neighbor_num)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      const uint64_t value =
            (static_cast<uint64_t>(__float_as_uint( INFINITY )) << 32) | static_cast<uint32_t>(-1);
      for (int i = index; i < query_num; i += step) {
         for (int j = 0; j < neighbor_num; ++j) lists[i * neighbor_num + j] = value;
      }
   }

   __device__
   float push(uint64_t* lists, int node_index, int neighbor_num, float squared_distance)
   {
      uint64_t value =
            (static_cast<uint64_t>(__float_as_uint( squared_distance )) << 32) | static_cast<uint32_t>(node_index);
      for (int i = 0; i < neighbor_num; ++i) {
         const uint64_t max_value = max( lists[i], value );
         const uint64_t min_value = min( lists[i], value );
         lists[i] = min_value;
         value = max_value;
      }
      return __uint_as_float( lists[neighbor_num - 1] >> 32 );
   }

   __device__
   void findNearestNeighbors(
      uint64_t* lists,
      const KdtreeNode* root,
      const node_type* coordinates,
      const node_type* query,
      int node_index,
      int neighbor_num,
      int size,
      int dim
   )
   {
      int depth = 0;
      int prev = -1;
      int curr = node_index;
      float max_squared_distance = INFINITY;
      while (curr >= 0) {
         const KdtreeNode* node = &root[curr];
         const int parent = node->ParentIndex;
         if (curr >= size) {
            prev = curr;
            curr = parent;
            continue;
         }

         const bool from_child = prev >= 0 && (prev == node->LeftChildIndex || prev == node->RightChildIndex);
         if (!from_child) {
            float squared_distance = 0.0f;
            for (int d = 0; d < dim; ++d) {
               const float x = query[d] - coordinates[node->Index * dim + d];
               squared_distance += x * x;
            }
            if (squared_distance <= max_squared_distance) {
               max_squared_distance = push( lists, curr, neighbor_num, squared_distance );
            }
         }

         const int axis = depth % dim;
         const node_type t = query[axis] - coordinates[node->Index * dim + axis];
         const bool right_priority = t > 0;
         const int far_child = right_priority ? node->LeftChildIndex : node->RightChildIndex;
         const int close_child = right_priority ? node->RightChildIndex : node->LeftChildIndex;

         int next = -1;
         if (prev >= 0 && prev == close_child) {
            if (far_child >= 0 && (t == 0 || t * t <= max_squared_distance)) {
               next = far_child;
               depth++;
            }
            else {
               next = parent;
               depth--;
            }
         }
         else if (prev >= 0 && prev == far_child) {
            next = parent;
            depth--;
         }
         else if (prev < 0 || prev == parent) {
            if (close_child < 0 && far_child < 0) {
               next = parent;
               depth--;
            }
            else if (close_child < 0) {
               next = far_child;
               depth++;
            }
            else {
               next = close_child;
               depth++;
            }
         }

         prev = curr;
         curr = next;
      }
   }

   __global__
   void cuFindNearestNeighbors(
      uint64_t* lists,
      const KdtreeNode* root,
      const node_type* coordinates,
      const node_type* queries,
      int node_index,
      int query_num,
      int neighbor_num,
      int size,
      int dim
   )
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < query_num; i += step) {
         const node_type* query = queries + i * dim;
         uint64_t* founds = lists + i * neighbor_num;
         findNearestNeighbors( founds, root, coordinates, query, node_index, neighbor_num, size, dim );
      }
   }

   void KdtreeCUDA::findNearestNeighbors(
      std::vector<std::vector<std::pair<float, int>>>& founds,
      const node_type* queries,
      int query_num,
      int neighbor_num
   ) const
   {
      if (Device.RootNode < 0 || Coordinates == nullptr) return;

      uint64_t* lists = nullptr;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&lists), sizeof( uint64_t ) * neighbor_num * query_num ) );

      node_type* device_queries = nullptr;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&device_queries), sizeof( node_type ) * query_num * Dim ) );
      CHECK_CUDA(
         cudaMemcpyAsync(
            device_queries, queries, sizeof( node_type ) * query_num * Dim,
            cudaMemcpyHostToDevice, Device.Stream
         )
      );

      const int block_num = divideUp( query_num, WarpSize );
      cuInitializeKNN<<<block_num, WarpSize, 0, Device.Stream>>>( lists, query_num, neighbor_num );
      CHECK_KERNEL;

      cuFindNearestNeighbors<<<block_num, WarpSize, 0, Device.Stream>>>(
         lists, Device.Root, Device.CoordinatesDevicePtr, device_queries,
         Device.RootNode, query_num, neighbor_num, Device.TupleNum, Dim
      );
      CHECK_KERNEL;

      founds.clear();
      founds.resize( query_num );
      std::vector<uint64_t> neighbors(neighbor_num);
      for (int i = 0; i < query_num; ++i) {
         CHECK_CUDA(
            cudaMemcpyAsync(
               neighbors.data(), lists + i * neighbor_num, sizeof( uint64_t ) * neighbor_num,
               cudaMemcpyDeviceToHost, Device.Stream
            )
         );

         for (int j = 0; j < neighbor_num; ++j) {
            const auto index = static_cast<int>(neighbors[j] & 0xffffffff);
            if (index < 0) break;

            const auto float_part = static_cast<int>(neighbors[j] >> 32);
            const float squared_distance = *reinterpret_cast<const float*>(&float_part);
            founds[i].emplace_back( squared_distance, index );
         }
      }

      CHECK_CUDA( cudaFree( lists ) );
      CHECK_CUDA( cudaFree( device_queries ) );
   }
}
#endif