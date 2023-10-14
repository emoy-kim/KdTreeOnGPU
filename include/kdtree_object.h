#pragma once

#include "object.h"

struct KdtreeNodeGL
{
   alignas(4) int Index;
   alignas(4) int ParentIndex;
   alignas(4) int LeftChildIndex;
   alignas(4) int RightChildIndex;

   KdtreeNodeGL() : Index( -1 ), ParentIndex( -1 ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
   explicit KdtreeNodeGL(int index) : Index( index ), ParentIndex( -1 ), LeftChildIndex( -1 ), RightChildIndex( -1 ) {}
};

class KdtreeGL final : public ObjectGL
{
public:
   static constexpr int WarpSize = 32;
   static constexpr int ThreadNum = 512;
   static constexpr int ThreadBlockNum = 32;
   static constexpr int SampleStride = 128;
   static constexpr int SharedSize = WarpSize * WarpSize;

   KdtreeGL();
   ~KdtreeGL() override = default;

   void setObject(GLenum draw_mode, const std::string& obj_file_path) override;
   void initialize();
   void prepareSorting();
   void releaseSorting();
   void prepareBuilding();
   void releaseBuilding();
   void prepareVerifying();
   void releaseVerifying();
   void prepareSearching(const std::vector<glm::vec3>& queries);
   void releaseSearching();
   void prepareKNN(const std::vector<glm::vec3>& queries, int neighbor_num);
   void releaseKNN();
   void setUniqueNum(int unique_num) { UniqueNum = unique_num; }
   void setRootNode(int root_node) { RootNode = root_node; }
   void setNodeNum(int node_num) { NodeNum = node_num; }
   [[nodiscard]] int getDimension() const { return Dim; }
   [[nodiscard]] int getUniqueNum() const { return UniqueNum; }
   [[nodiscard]] int getRootNode() const { return RootNode; }
   [[nodiscard]] int getNodeNum() const { return NodeNum; }
   [[nodiscard]] int getSize() const { return static_cast<int>(Vertices.size()); }
   [[nodiscard]] int getMaxSampleNum() const { return Sort.MaxSampleNum; }
   [[nodiscard]] GLuint getRoot() const { return Root; }
   [[nodiscard]] GLuint getCoordinates() const { return Coordinates; }
   [[nodiscard]] GLuint getReference(int index) const { return Reference[index]; }
   [[nodiscard]] GLuint getBuffer(int index) const { return Buffer[index]; }
   [[nodiscard]] GLuint getSortReference() const { return Sort.Reference; }
   [[nodiscard]] GLuint getSortBuffer() const { return Sort.Buffer; }
   [[nodiscard]] GLuint getLeftRanks() const { return Sort.LeftRanks; }
   [[nodiscard]] GLuint getRightRanks() const { return Sort.RightRanks; }
   [[nodiscard]] GLuint getLeftLimits() const { return Sort.LeftLimits; }
   [[nodiscard]] GLuint getRightLimits() const { return Sort.RightLimits; }
   [[nodiscard]] GLuint getLeftChildNumInWarp() const { return LeftChildNumInWarp; }
   [[nodiscard]] GLuint getRightChildNumInWarp() const { return RightChildNumInWarp; }
   [[nodiscard]] GLuint getNodeSums() const { return NodeSums; }
   [[nodiscard]] GLuint getMidReferences(int index) const { return MidReferences[index]; }
   [[nodiscard]] GLuint getSearchLists() const { return Search.Lists; }
   [[nodiscard]] GLuint getSearchListLengths() const { return Search.ListLengths; }
   [[nodiscard]] GLuint getQueries() const { return Search.Queries; }

private:
   struct SortGL
   {
      int MaxSampleNum;
      GLuint LeftRanks;
      GLuint RightRanks;
      GLuint LeftLimits;
      GLuint RightLimits;
      GLuint Reference;
      GLuint Buffer;

      SortGL() :
         MaxSampleNum( 0 ), LeftRanks( 0 ), RightRanks( 0 ), LeftLimits( 0 ), RightLimits( 0 ), Reference( 0 ),
         Buffer( 0 ) {}
   };

   struct SearchGL
   {
      GLuint Lists;
      GLuint ListLengths;
      GLuint Queries;

      SearchGL() : Lists( 0 ), ListLengths( 0 ), Queries( 0 ) {}
   };

   const int Dim;
   int UniqueNum;
   int RootNode;
   int NodeNum;
   SortGL Sort;
   SearchGL Search;
   GLuint Root;
   GLuint Coordinates;
   GLuint LeftChildNumInWarp;
   GLuint RightChildNumInWarp;
   GLuint NodeSums;
   std::array<GLuint, 2> MidReferences;
   std::vector<GLuint> Reference;
   std::vector<GLuint> Buffer;
   std::vector<glm::vec3> Vertices;
};