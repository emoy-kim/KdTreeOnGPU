#include "kdtree.h"
#include "cuda/kdtree.cuh"

void testMultithreading(std::vector<float>& output, const std::vector<glm::vec3>& coordinates, bool print_result)
{
   std::cout << "\n================ MultiThreading Test ================\n";
   Kdtree kdtree(coordinates);
   if (print_result) kdtree.print();
   kdtree.getResult( output );

   constexpr float search_radius = 2.0f;
   const glm::vec3 query(4.0f, 3.0f, 1.0f);

   auto start_time = std::chrono::steady_clock::now();
   const auto list = kdtree.search( query, search_radius );
   auto end_time = std::chrono::steady_clock::now();
   const auto search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << "\n>> " << list.size() << " nodes within " << search_radius << " units of (" <<
      query.x << ", " << query.y << ", " << query.z << ") in all dimensions, Search Time: " << search_time << "\n";
   std::cout << ">> List of k-d nodes within " << search_radius << "-unit search radius\n   ";
   for (const auto& p : list) std::cout << "(" << p->Tuple[0] << ", " << p->Tuple[1] << ", " << p->Tuple[2] << ") ";
   std::cout << "\n";

   start_time = std::chrono::steady_clock::now();
   const auto nn_list = kdtree.findNearestNeighbors( query, 5 );
   end_time = std::chrono::steady_clock::now();
   const auto nn_search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << "\n>> " << std::distance( nn_list.begin(), nn_list.end() ) << " nearest neighbors of (" <<
      query.x << ", " << query.y << ", " << query.z << ") in all dimensions, Search Time: " << nn_search_time << "\n";
   std::cout << ">> List of nearest neighbors\n";
   for (const auto& p : nn_list) {
      std::cout << "   (" << p.second->Tuple[0] << ", " << p.second->Tuple[1] << ", " << p.second->Tuple[2] << ") in "
         << p.first << "\n";
   }
}

#ifdef USE_CUDA
void testCUDA(std::vector<float>& output, const std::vector<glm::vec3>& coordinates, bool print_result)
{
   std::cout << "\n================ CUDA Test ================\n";
   cuda::KdtreeCUDA kdtree(glm::value_ptr( coordinates[0] ), static_cast<int>(coordinates.size()), 3);
   if (print_result) kdtree.print();
   kdtree.getResult( output );

   constexpr float search_radius = 2.0f;
   const glm::vec3 query(4.0f, 3.0f, 1.0f);

   auto start_time = std::chrono::steady_clock::now();
   const auto list = kdtree.search( glm::value_ptr( query ), search_radius );
   auto end_time = std::chrono::steady_clock::now();
   const auto search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << "\n>> " << list.size() << " nodes within " << search_radius << " units of (" <<
      query.x << ", " << query.y << ", " << query.z << ") in all dimensions, Search Time: " << search_time << "\n";
   std::cout << ">> List of k-d nodes within " << search_radius << "-unit search radius\n   ";
   for (const auto& r : list) {
      const glm::vec3& p = coordinates[r];
      std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
   }
   std::cout << "\n";
}
#endif

int main()
{
   auto start_time = std::chrono::steady_clock::now();
   //const std::vector<glm::vec3> coordinates = {
   //   { 2.0f, 3.0f, 3.0f }, { 5.0f, 4.0f, 2.0f }, { 9.0f, 6.0f, 7.0f }, { 4.0f, 7.0f, 9.0f }, { 8.0f, 1.0f, 5.0f },
   //   { 7.0f, 2.0f, 6.0f }, { 9.0f, 4.0f, 1.0f }, { 8.0f, 4.0f, 2.0f }, { 9.0f, 7.0f, 8.0f }, { 6.0f, 3.0f, 1.0f },
   //   { 3.0f, 4.0f, 5.0f }, { 1.0f, 6.0f, 8.0f }, { 9.0f, 5.0f, 3.0f }, { 2.0f, 1.0f, 3.0f }, { 8.0f, 7.0f, 6.0f },
   //   { 5.0f, 4.0f, 2.0f }, { 6.0f, 3.0f, 1.0f }, { 8.0f, 7.0f, 6.0f }, { 9.0f, 6.0f, 7.0f }, { 2.0f, 1.0f, 3.0f },
   //   { 7.0f, 2.0f, 6.0f }, { 4.0f, 7.0f, 9.0f }, { 1.0f, 6.0f, 8.0f }, { 3.0f, 4.0f, 5.0f }, { 9.0f, 4.0f, 1.0f }
   //};
   constexpr int n = 1024 * 1000;
   std::vector<glm::vec3> coordinates;
   for (int i = 0; i < n; ++i) {
      coordinates.emplace_back(
         getRandomValue( 0.0f, 100.0f ),
         getRandomValue( 0.0f, 100.0f ),
         getRandomValue( 0.0f, 100.0f )
      );
   }
   auto end_time = std::chrono::steady_clock::now();
   const auto generation_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << ">> Coordinates generated: " << generation_time << " sec.\n";

   const bool print_result = false;

   std::vector<float> mt_output;
   testMultithreading( mt_output, coordinates, print_result );

#ifdef USE_CUDA
   std::vector<float> cuda_output;
   testCUDA( cuda_output, coordinates, print_result );

   assert( mt_output.size() == cuda_output.size() );
   for (size_t i = 0; i < mt_output.size(); ++i) {
      assert( mt_output[i] == cuda_output[i] );
   }
   std::cout << " ====================== Correct Kd-Tree! ======================\n";
#endif
   return 0;
}