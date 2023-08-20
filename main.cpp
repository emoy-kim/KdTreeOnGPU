#include "kdtree.h"

int main()
{
   const std::vector<glm::vec3> coordinates = {
      { 2.0f, 3.0f, 3.0f }, { 5.0f, 4.0f, 2.0f }, { 9.0f, 6.0f, 7.0f }, { 4.0f, 7.0f, 9.0f }, { 8.0f, 1.0f, 5.0f },
      { 7.0f, 2.0f, 6.0f }, { 9.0f, 4.0f, 1.0f }, { 8.0f, 4.0f, 2.0f }, { 9.0f, 7.0f, 8.0f }, { 6.0f, 3.0f, 1.0f },
      { 3.0f, 4.0f, 5.0f }, { 1.0f, 6.0f, 8.0f }, { 9.0f, 5.0f, 3.0f }, { 2.0f, 1.0f, 3.0f }, { 8.0f, 7.0f, 6.0f },
      { 5.0f, 4.0f, 2.0f }, { 6.0f, 3.0f, 1.0f }, { 8.0f, 7.0f, 6.0f }, { 9.0f, 6.0f, 7.0f }, { 2.0f, 1.0f, 3.0f },
      { 7.0f, 2.0f, 6.0f }, { 4.0f, 7.0f, 9.0f }, { 1.0f, 6.0f, 8.0f }, { 3.0f, 4.0f, 5.0f }, { 9.0f, 4.0f, 1.0f }
   };

   Kdtree kdtree(coordinates);
   kdtree.print();

   constexpr float search_radius = 2.0f;
   const glm::vec3 query(4.0f, 3.0f, 1.0f);
   const glm::vec3 lower = query - search_radius;
   const glm::vec3 upper = query + search_radius;
   auto start_time = std::chrono::system_clock::now();
   const auto list = kdtree.search( query, search_radius );
   auto end_time = std::chrono::system_clock::now();
   const auto search_time =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()) * 1e-9;
   std::cout << "\n>> " << list.size() << " nodes within " << search_radius << " units of (" <<
      query.x << ", " << query.y << ", " << query.z << ") in all dimensions, Search Time: " << search_time << "\n";
   std::cout << ">> List of k-d nodes within " << search_radius << "-unit search radius\n   ";
   for (const auto& p : list) std::cout << "(" << p->Tuple[0] << ", " << p->Tuple[1] << ", " << p->Tuple[2] << ") ";
   std::cout << "\n";
   return 0;
}