/*
 * Author: Jeesun Kim
 * E-mail: emoy.kim_AT_gmail.com
 *
 */

#include "kdtree.h"
#include "cuda/kdtree.cuh"

constexpr bool PrintResult = false;
constexpr float SearchRadius = 2.0f;

void testMultithreading(
   std::vector<float>& output,
   const std::vector<glm::vec3>& coordinates,
   const std::vector<glm::vec3>& queries
)
{
   std::cout << "\n================ MultiThreading Test ================\n";
   Kdtree kdtree(coordinates);
   if (PrintResult) kdtree.print();
   kdtree.getResult( output );

   /*double search_time = 0.0;
   std::vector<std::list<const Kdtree<float, 3>::KdtreeNode*>> founds;
   for (const auto& query : queries) {
      const auto start_time = std::chrono::steady_clock::now();
      founds.emplace_back( kdtree.search( query, SearchRadius ) );
      const auto end_time = std::chrono::steady_clock::now();
      search_time +=
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   }
   std::cout << ">> Search " << queries.size() << " queries\n";
   for (size_t i = 0; i < founds.size(); ++i) {
      std::cout << ">> [" << i + 1 << "] " << founds[i].size() << " nodes within " << SearchRadius << " units of ("
         << queries[i].x << ", " << queries[i].y << ", " << queries[i].z << ") in all dimensions\n";
      if (!founds[i].empty()) {
         std::cout << "\t* List of k-d nodes within " << SearchRadius << "-unit search radius:\t";
         for (const auto& p : founds[i]) {
            std::cout << "(" << p->Tuple[0] << ", " << p->Tuple[1] << ", " << p->Tuple[2] << ") ";
         }
         std::cout << "\n";
      }
   }
   std::cout << ">> Total Search Time: " << search_time << " sec.\n";*/

   double nn_search_time = 0.0;
   std::vector<std::forward_list<std::pair<float, const Kdtree<float, 3>::KdtreeNode*>>> nn_founds;
   for (const auto& query : queries) {
      const auto start_time = std::chrono::steady_clock::now();
      nn_founds.emplace_back( kdtree.findNearestNeighbors( query, 5 ) );
      const auto end_time = std::chrono::steady_clock::now();
      nn_search_time +=
         static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   }
   for (size_t i = 0; i < nn_founds.size(); ++i) {
      std::cout << "\n>> " << std::distance( nn_founds[i].begin(), nn_founds[i].end() ) << " nearest neighbors of ("
         << queries[i].x << ", " << queries[i].y << ", " << queries[i].z << ") in all dimensions\n";
      std::cout << ">> List of nearest neighbors\n";
      for (const auto& p : nn_founds[i]) {
         std::cout << "   (" << p.second->Tuple[0] << ", " << p.second->Tuple[1] << ", " << p.second->Tuple[2]
            << ") in " << p.first << "\n";
      }
   }
   std::cout << ">> Total Nearest Neighbors Search Time: " << nn_search_time << " sec.\n";
}

#ifdef USE_CUDA
void testCUDA(
   std::vector<float>& output,
   const std::vector<glm::vec3>& coordinates,
   const std::vector<glm::vec3>& queries
)
{
   std::cout << "\n================ CUDA Test ================\n";
   cuda::KdtreeCUDA kdtree(glm::value_ptr( coordinates[0] ), static_cast<int>(coordinates.size()), 3);
   if (PrintResult) kdtree.print();
   kdtree.getResult( output );

   std::vector<std::vector<int>> founds;
   auto start_time = std::chrono::steady_clock::now();
   kdtree.search( founds, glm::value_ptr( queries[0] ), static_cast<int>(queries.size()), SearchRadius );
   auto end_time = std::chrono::steady_clock::now();
   const auto search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   /*std::cout << ">> Search " << queries.size() << " queries\n";
   for (size_t i = 0; i < founds.size(); ++i) {
      std::cout << ">> [" << i + 1 << "] " << founds[i].size() << " nodes within " << SearchRadius << " units of ("
         << queries[i].x << ", " << queries[i].y << ", " << queries[i].z << ") in all dimensions\n";
      if (!founds[i].empty()) {
         std::cout << "\t* List of k-d nodes within " << SearchRadius << "-unit search radius:\t";
         for (const auto& r : founds[i]) {
            const glm::vec3& p = coordinates[r];
            std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
         }
         std::cout << "\n";
      }
   }
   std::cout << ">> Total Search Time: " << search_time << " sec.\n";*/

   std::vector<std::vector<int>> nn_founds;
   start_time = std::chrono::steady_clock::now();
   kdtree.findNearestNeighbors(
      nn_founds, glm::value_ptr( queries[0] ), static_cast<int>(queries.size()), SearchRadius
   );
   end_time = std::chrono::steady_clock::now();
   const auto nn_search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   for (size_t i = 0; i < nn_founds.size(); ++i) {
      std::cout << "\n>> " << std::distance( nn_founds[i].begin(), nn_founds[i].end() ) << " nearest neighbors of ("
         << queries[i].x << ", " << queries[i].y << ", " << queries[i].z << ") in all dimensions\n";
      std::cout << ">> List of nearest neighbors\n";
      for (const auto& r : nn_founds[i]) {
         const glm::vec3& p = coordinates[r];
         std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
      }
      std::cout << "\n";
   }
   std::cout << ">> Total Nearest Neighbors Search Time: " << nn_search_time << " sec.\n";
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
   constexpr int n = 1024 * 32;
   std::vector<glm::vec3> coordinates;
   coordinates.reserve( n );
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

   std::vector<glm::vec3> queries;
   for (int i = 0; i < 10; ++i) {
      queries.emplace_back(
         getRandomValue( 0.0f, 100.0f ),
         getRandomValue( 0.0f, 100.0f ),
         getRandomValue( 0.0f, 100.0f )
      );
   }

   std::vector<float> mt_output;
   testMultithreading( mt_output, coordinates, queries );

#ifdef USE_CUDA
   std::vector<float> cuda_output;
   testCUDA( cuda_output, coordinates, queries );

   assert( mt_output.size() == cuda_output.size() );
   for (size_t i = 0; i < mt_output.size(); ++i) {
      assert( mt_output[i] == cuda_output[i] );
   }
#endif
   return 0;
}