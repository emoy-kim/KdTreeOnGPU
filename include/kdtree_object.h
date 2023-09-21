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
   KdtreeGL();
   ~KdtreeGL() override;

   void setObject(GLenum draw_mode, const std::string& obj_file_path) override;
   void initialize();
   void prepareSorting();
   void releaseSorting();
   [[nodiscard]] int getSize() const { return static_cast<int>(Vertices.size()); }
   [[nodiscard]] GLuint getRoot() const { return Root; }
   [[nodiscard]] GLuint getCoordinates() const { return Coordinates; }
   [[nodiscard]] GLuint getReference(int index) const { return Reference[index]; }


private:
   static constexpr int SampleStride = 128;

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

   const int Dim;
   int TupleNum;
   int RootNode;
   SortGL Sort;
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