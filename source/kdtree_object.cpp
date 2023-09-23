#include "kdtree_object.h"

KdtreeGL::KdtreeGL() :
   Dim( 3 ), UniqueNum( 0 ), RootNode( -1 ), Sort(), Root( 0 ), Coordinates( 0 ), LeftChildNumInWarp( 0 ),
   RightChildNumInWarp( 0 ), NodeSums( 0 ), MidReferences{ 0, 0 }, Reference( Dim + 2, 0 ), Buffer( Dim + 2, 0 )
{
}

KdtreeGL::~KdtreeGL()
{

}

void KdtreeGL::setObject(GLenum draw_mode, const std::string& obj_file_path)
{
   Vertices.clear();
   DrawMode = draw_mode;
   std::vector<glm::vec3> normals;
   std::vector<glm::vec2> textures;
   if (!readObjectFile( Vertices, normals, textures, obj_file_path )) return;

   const bool normals_exist = !normals.empty();
   const bool textures_exist = !textures.empty();
   for (uint i = 0; i < Vertices.size(); ++i) {
      DataBuffer.push_back( Vertices[i].x );
      DataBuffer.push_back( Vertices[i].y );
      DataBuffer.push_back( Vertices[i].z );
      if (normals_exist) {
         DataBuffer.push_back( normals[i].x );
         DataBuffer.push_back( normals[i].y );
         DataBuffer.push_back( normals[i].z );
      }
      if (textures_exist) {
         DataBuffer.push_back( textures[i].x );
         DataBuffer.push_back( textures[i].y );
      }
      VerticesCount++;
   }
   int n = 3;
   if (normals_exist) n += 3;
   if (textures_exist) n += 2;
   const auto n_bytes_per_vertex = static_cast<int>(n * sizeof( GLfloat ));
   prepareVertexBuffer( n_bytes_per_vertex );
   if (normals_exist) prepareNormal();
   if (textures_exist) prepareTexture( normals_exist );
   prepareIndexBuffer();
}

void KdtreeGL::initialize()
{
   const auto size = static_cast<int>(Vertices.size());
   Coordinates = addCustomBufferObject<float>( "Coordinates", Dim * (size + 1) );
   glNamedBufferSubData(
      Coordinates, 0,
      static_cast<int>(Dim * size * sizeof( float )),
      glm::value_ptr( Vertices[0] )
   );

   float max_value[Dim];
   for (int i = 0; i < Dim; ++i) max_value[i] = std::numeric_limits<float>::max();
   glNamedBufferSubData(
      Coordinates,
      static_cast<int>(Dim * size * sizeof( float )),
      static_cast<int>(Dim * sizeof( float )),
      max_value
   );

   Root = addCustomBufferObject<KdtreeNodeGL>( "Root", size );
}

void KdtreeGL::prepareSorting()
{
   const auto size = static_cast<int>(Vertices.size());
   const int max_sample_num = size / SampleStride + 1;
   Sort.MaxSampleNum = max_sample_num;
   Sort.LeftRanks = addCustomBufferObject<int>( "LeftRanks", max_sample_num );
   Sort.RightRanks = addCustomBufferObject<int>( "RightRanks", max_sample_num );
   Sort.LeftLimits = addCustomBufferObject<int>( "LeftLimits", max_sample_num );
   Sort.RightLimits = addCustomBufferObject<int>( "RightLimits", max_sample_num );
   Sort.Reference = addCustomBufferObject<int>( "SortReference", size );
   Sort.Buffer = addCustomBufferObject<float>( "SortBuffer", size );

   for (int i = 0; i <= Dim + 1; ++i) {
      Reference[i] = addCustomBufferObject<int>( "Reference", size );
   }
   for (int i = 0; i <= Dim; ++i) {
      Buffer[i] = addCustomBufferObject<float>( "Buffer", size );
   }
}

void KdtreeGL::releaseSorting()
{
   releaseCustomBuffer( "LeftRanks" );
   releaseCustomBuffer( "RightRanks" );
   releaseCustomBuffer( "LeftLimits" );
   releaseCustomBuffer( "RightLimits" );
   releaseCustomBuffer( "SortReference" );
   releaseCustomBuffer( "SortBuffer" );
   Sort.LeftRanks = 0;
   Sort.RightRanks = 0;
   Sort.LeftLimits = 0;
   Sort.RightLimits = 0;
   Sort.Reference = 0;
   Sort.Buffer = 0;
}