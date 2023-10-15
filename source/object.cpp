#include "object.h"

ObjectGL::ObjectGL() :
   VAO( 0 ), VBO( 0 ), IBO( 0 ), DrawMode( 0 ), VerticesCount( 0 ), EmissionColor( 0.1f, 0.1f, 0.1f, 1.0f ),
   AmbientReflectionColor( 0.2f, 0.2f, 0.2f, 1.0f ), DiffuseReflectionColor( 0.8f, 0.8f, 0.8f, 1.0f ),
   SpecularReflectionColor( 0.8f, 0.8f, 0.8f, 1.0f ), SpecularReflectionExponent( 16.0f )
{
}

ObjectGL::~ObjectGL()
{
   if (IBO != 0) glDeleteBuffers( 1, &IBO );
   if (VBO != 0) glDeleteBuffers( 1, &VBO );
   if (VAO != 0) glDeleteVertexArrays( 1, &VAO );
   for (const auto& texture_id : TextureID) {
      if (texture_id != 0) glDeleteTextures( 1, &texture_id );
   }
   for (const auto& buffer : CustomBuffers) {
      if (buffer != 0) glDeleteBuffers( 1, &buffer );
   }
}

void ObjectGL::setEmissionColor(const glm::vec4& emission_color)
{
   EmissionColor = emission_color;
}

void ObjectGL::setAmbientReflectionColor(const glm::vec4& ambient_reflection_color)
{
   AmbientReflectionColor = ambient_reflection_color;
}

void ObjectGL::setDiffuseReflectionColor(const glm::vec4& diffuse_reflection_color)
{
   DiffuseReflectionColor = diffuse_reflection_color;
}

void ObjectGL::setSpecularReflectionColor(const glm::vec4& specular_reflection_color)
{
   SpecularReflectionColor = specular_reflection_color;
}

void ObjectGL::setSpecularReflectionExponent(const float& specular_reflection_exponent)
{
   SpecularReflectionExponent = specular_reflection_exponent;
}

bool ObjectGL::prepareTexture2DUsingFreeImage(const std::string& file_path, bool is_grayscale) const
{
   const FREE_IMAGE_FORMAT format = FreeImage_GetFileType( file_path.c_str(), 0 );
   FIBITMAP* texture = FreeImage_Load( format, file_path.c_str() );
   if (!texture) return false;

   FIBITMAP* texture_converted;
   const uint n_bits_per_pixel = FreeImage_GetBPP( texture );
   const uint n_bits = is_grayscale ? 8 : 32;
   if (is_grayscale) {
      texture_converted = n_bits_per_pixel == n_bits ? texture : FreeImage_GetChannel( texture, FICC_RED );
   }
   else {
      texture_converted = n_bits_per_pixel == n_bits ? texture : FreeImage_ConvertTo32Bits( texture );
   }

   const auto width = static_cast<GLsizei>(FreeImage_GetWidth( texture_converted ));
   const auto height = static_cast<GLsizei>(FreeImage_GetHeight( texture_converted ));
   GLvoid* data = FreeImage_GetBits( texture_converted );
   glTextureStorage2D( TextureID.back(), 1, is_grayscale ? GL_R8 : GL_RGBA8, width, height );
   glTextureSubImage2D( TextureID.back(), 0, 0, 0, width, height, is_grayscale ? GL_RED : GL_BGRA, GL_UNSIGNED_BYTE, data );

   FreeImage_Unload( texture_converted );
   if (n_bits_per_pixel != n_bits) FreeImage_Unload( texture );
   return true;
}

int ObjectGL::addTexture(const std::string& texture_file_path, bool is_grayscale)
{
   GLuint texture_id = 0;
   glCreateTextures( GL_TEXTURE_2D, 1, &texture_id );
   TextureID.emplace_back( texture_id );
   if (!prepareTexture2DUsingFreeImage( texture_file_path, is_grayscale )) {
      glDeleteTextures( 1, &texture_id );
      TextureID.erase( TextureID.end() - 1 );
      std::cerr << "Could not read image file " << texture_file_path.c_str() << "\n";
      return -1;
   }

   glTextureParameteri( texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_S, GL_REPEAT );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_T, GL_REPEAT );
   glGenerateTextureMipmap( texture_id );
   return static_cast<int>(TextureID.size() - 1);
}

void ObjectGL::addTexture(int width, int height, bool is_grayscale)
{
   GLuint texture_id = 0;
   glCreateTextures( GL_TEXTURE_2D, 1, &texture_id );
   glTextureStorage2D(
      texture_id, 1,
      is_grayscale ? GL_R8 : GL_RGBA8,
      width, height
   );
   glTextureParameteri( texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_S, GL_REPEAT );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_T, GL_REPEAT );
   glGenerateTextureMipmap( texture_id );
   TextureID.emplace_back( texture_id );
}

int ObjectGL::addTexture(const uint8_t* image_buffer, int width, int height, bool is_grayscale)
{
   addTexture( width, height, is_grayscale );
   glTextureSubImage2D(
      TextureID.back(), 0, 0, 0,
      width, height,
      is_grayscale ? GL_RED : GL_RGBA,
      GL_UNSIGNED_BYTE,
      image_buffer
   );
   return static_cast<int>(TextureID.size() - 1);
}

void ObjectGL::prepareTexture(bool normals_exist) const
{
   const uint offset = normals_exist ? 6 : 3;
   glVertexArrayAttribFormat( VAO, TextureLoc, 2, GL_FLOAT, GL_FALSE, offset * sizeof( GLfloat ) );
   glEnableVertexArrayAttrib( VAO, TextureLoc );
   glVertexArrayAttribBinding( VAO, TextureLoc, 0 );
}

void ObjectGL::prepareNormal() const
{
   glVertexArrayAttribFormat( VAO, NormalLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof( GLfloat ) );
   glEnableVertexArrayAttrib( VAO, NormalLoc );
   glVertexArrayAttribBinding( VAO, NormalLoc, 0 );
}

void ObjectGL::prepareVertexBuffer(int n_bytes_per_vertex)
{
   glCreateBuffers( 1, &VBO );
   glNamedBufferStorage( VBO, sizeof( GLfloat ) * DataBuffer.size(), DataBuffer.data(), GL_DYNAMIC_STORAGE_BIT );

   glCreateVertexArrays( 1, &VAO );
   glVertexArrayVertexBuffer( VAO, 0, VBO, 0, n_bytes_per_vertex );
   glVertexArrayAttribFormat( VAO, VertexLoc, 3, GL_FLOAT, GL_FALSE, 0 );
   glEnableVertexArrayAttrib( VAO, VertexLoc );
   glVertexArrayAttribBinding( VAO, VertexLoc, 0 );
}

void ObjectGL::prepareIndexBuffer()
{
   assert( VAO != 0 );

   if (IBO != 0) glDeleteBuffers( 1, &IBO );

   glCreateBuffers( 1, &IBO );
   glNamedBufferStorage( IBO, sizeof( GLuint ) * IndexBuffer.size(), IndexBuffer.data(), GL_DYNAMIC_STORAGE_BIT );
   glVertexArrayElementBuffer( VAO, IBO );
}

void ObjectGL::getSquareObject(
   std::vector<glm::vec3>& vertices,
   std::vector<glm::vec3>& normals,
   std::vector<glm::vec2>& textures
)
{
   vertices = {
      { 1.0f, 0.0f, 0.0f },
      { 1.0f, 1.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f },

      { 1.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f },
      { 0.0f, 0.0f, 0.0f }
   };
   normals = {
      { 0.0f, 0.0f, 1.0f },
      { 0.0f, 0.0f, 1.0f },
      { 0.0f, 0.0f, 1.0f },

      { 0.0f, 0.0f, 1.0f },
      { 0.0f, 0.0f, 1.0f },
      { 0.0f, 0.0f, 1.0f }
   };
   textures = {
      { 1.0f, 0.0f },
      { 1.0f, 1.0f },
      { 0.0f, 1.0f },

      { 1.0f, 0.0f },
      { 0.0f, 1.0f },
      { 0.0f, 0.0f }
   };
}

void ObjectGL::setObject(GLenum draw_mode, const std::vector<glm::vec3>& vertices)
{
   DrawMode = draw_mode;
   VerticesCount = 0;
   DataBuffer.clear();
   for (auto& vertex : vertices) {
      DataBuffer.emplace_back( vertex.x );
      DataBuffer.emplace_back( vertex.y );
      DataBuffer.emplace_back( vertex.z );
      VerticesCount++;
   }
   const int n_bytes_per_vertex = 3 * sizeof( GLfloat );
   prepareVertexBuffer( n_bytes_per_vertex );
}

void ObjectGL::setObject(
   GLenum draw_mode,
   const std::vector<glm::vec3>& vertices,
   const std::vector<glm::vec3>& normals
)
{
   DrawMode = draw_mode;
   VerticesCount = 0;
   DataBuffer.clear();
   for (size_t i = 0; i < vertices.size(); ++i) {
      DataBuffer.emplace_back( vertices[i].x );
      DataBuffer.emplace_back( vertices[i].y );
      DataBuffer.emplace_back( vertices[i].z );
      DataBuffer.emplace_back( normals[i].x );
      DataBuffer.emplace_back( normals[i].y );
      DataBuffer.emplace_back( normals[i].z );
      VerticesCount++;
   }
   const int n_bytes_per_vertex = 6 * sizeof( GLfloat );
   prepareVertexBuffer( n_bytes_per_vertex );
   prepareNormal();
}

void ObjectGL::setObject(
   GLenum draw_mode,
   const std::vector<glm::vec3>& vertices,
   const std::vector<glm::vec3>& normals,
   const std::vector<glm::vec2>& textures
)
{
   DrawMode = draw_mode;
   VerticesCount = 0;
   DataBuffer.clear();
   for (size_t i = 0; i < vertices.size(); ++i) {
      DataBuffer.emplace_back( vertices[i].x );
      DataBuffer.emplace_back( vertices[i].y );
      DataBuffer.emplace_back( vertices[i].z );
      DataBuffer.emplace_back( normals[i].x );
      DataBuffer.emplace_back( normals[i].y );
      DataBuffer.emplace_back( normals[i].z );
      DataBuffer.emplace_back( textures[i].x );
      DataBuffer.emplace_back( textures[i].y );
      VerticesCount++;
   }
   const int n_bytes_per_vertex = 8 * sizeof( GLfloat );
   prepareVertexBuffer( n_bytes_per_vertex );
   prepareNormal();
   prepareTexture( true );
}

void ObjectGL::setObject(
   GLenum draw_mode,
   const std::vector<glm::vec3>& vertices,
   const std::vector<glm::vec3>& normals,
   const std::vector<glm::vec2>& textures,
   const std::string& texture_file_path,
   bool is_grayscale
)
{
   setObject( draw_mode, vertices, normals, textures );
   addTexture( texture_file_path, is_grayscale );
}

void ObjectGL::findNormals(
   std::vector<glm::vec3>& normals,
   const std::vector<glm::vec3>& vertices,
   const std::vector<GLuint>& vertex_indices
)
{
   normals.resize( vertices.size(), glm::vec3(0.0f) );
   const auto size = static_cast<int>(vertex_indices.size());
   for (int i = 0; i < size; i += 3) {
      const GLuint n0 = vertex_indices[i];
      const GLuint n1 = vertex_indices[i + 1];
      const GLuint n2 = vertex_indices[i + 2];
      const glm::vec3 normal = glm::cross( vertices[n1] - vertices[n0], vertices[n2] - vertices[n0] );
      normals[n0] += normal;
      normals[n1] += normal;
      normals[n2] += normal;
   }
   for (auto& n : normals) n = glm::normalize( n );
}

bool ObjectGL::readObjectFile(
   std::vector<glm::vec3>& vertices,
   std::vector<glm::vec3>& normals,
   std::vector<glm::vec2>& textures,
   const std::string& file_path
)
{
   std::ifstream file(file_path);
   if (!file.is_open()) {
      std::cout << "The object file is not correct.\n";
      return false;
   }

   bool found_normals = false, found_textures = false;
   std::vector<glm::vec3> vertex_buffer, normal_buffer;
   std::vector<glm::vec2> texture_buffer;
   std::vector<GLuint> vertex_indices, normal_indices, texture_indices;
   while (!file.eof()) {
      std::string word;
      file >> word;

      if (word == "v") {
         glm::vec3 vertex;
         file >> vertex.x >> vertex.y >> vertex.z;
         vertex_buffer.emplace_back( vertex );
      }
      else if (word == "vt") {
         glm::vec2 uv;
         file >> uv.x >> uv.y;
         texture_buffer.emplace_back( uv );
         found_textures = true;
      }
      else if (word == "vn") {
         glm::vec3 normal;
         file >> normal.x >> normal.y >> normal.z;
         normal_buffer.emplace_back( normal );
         found_normals = true;
      }
      else if (word == "f") {
         std::string face;
         const std::regex delimiter("[/]");
         for (int i = 0; i < 3; ++i) {
            file >> face;
            const std::sregex_token_iterator it(face.begin(), face.end(), delimiter, -1);
            const std::vector<std::string> vtn(it, std::sregex_token_iterator());
            vertex_indices.emplace_back( std::stoi( vtn[0] ) - 1 );
            if (found_textures) texture_indices.emplace_back( std::stoi( vtn[1] ) - 1 );
            if (found_normals) normal_indices.emplace_back( std::stoi( vtn[2] ) - 1 );
         }
      }
      else std::getline( file, word );
   }

   if (!found_normals) findNormals( normal_buffer, vertex_buffer, vertex_indices );

   vertices = std::move( vertex_buffer );
   normals = std::move( normal_buffer );
   if (found_textures && vertex_indices.size() == texture_indices.size()) textures = std::move( texture_buffer );
   IndexBuffer = std::move( vertex_indices );
   return true;
}

void ObjectGL::setObject(GLenum draw_mode, const std::string& obj_file_path)
{
   DrawMode = draw_mode;
   std::vector<glm::vec3> vertices, normals;
   std::vector<glm::vec2> textures;
   if (!readObjectFile( vertices, normals, textures, obj_file_path )) return;

   const bool normals_exist = !normals.empty();
   const bool textures_exist = !textures.empty();
   for (uint i = 0; i < vertices.size(); ++i) {
      DataBuffer.emplace_back( vertices[i].x );
      DataBuffer.emplace_back( vertices[i].y );
      DataBuffer.emplace_back( vertices[i].z );
      if (normals_exist) {
         DataBuffer.emplace_back( normals[i].x );
         DataBuffer.emplace_back( normals[i].y );
         DataBuffer.emplace_back( normals[i].z );
      }
      if (textures_exist) {
         DataBuffer.emplace_back( textures[i].x );
         DataBuffer.emplace_back( textures[i].y );
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

void ObjectGL::setSquareObject(GLenum draw_mode, bool use_texture)
{
   std::vector<glm::vec3> square_vertices, square_normals;
   std::vector<glm::vec2> square_textures;
   getSquareObject( square_vertices, square_normals, square_textures );
   if (use_texture) setObject( draw_mode, square_vertices, square_normals, square_textures );
   else setObject( draw_mode, square_vertices, square_normals );
}

void ObjectGL::setSquareObject(
   GLenum draw_mode,
   const std::string& texture_file_path,
   bool is_grayscale
)
{
   std::vector<glm::vec3> square_vertices, square_normals;
   std::vector<glm::vec2> square_textures;
   getSquareObject( square_vertices, square_normals, square_textures );
   setObject( draw_mode, square_vertices, square_normals, square_textures, texture_file_path, is_grayscale );
}

void ObjectGL::replaceVertices(
   const std::vector<glm::vec3>& vertices,
   bool normals_exist,
   bool textures_exist
)
{
   assert( VBO != 0 );

   VerticesCount = 0;
   int step = 3;
   if (normals_exist) step += 3;
   if (textures_exist) step += 2;
   for (size_t i = 0; i < vertices.size(); ++i) {
      DataBuffer[i * step] = vertices[i].x;
      DataBuffer[i * step + 1] = vertices[i].y;
      DataBuffer[i * step + 2] = vertices[i].z;
      VerticesCount++;
   }
   glNamedBufferSubData( VBO, 0, static_cast<int>(sizeof( GLfloat ) * VerticesCount * step), DataBuffer.data() );
}

void ObjectGL::replaceVertices(
   const std::vector<float>& vertices,
   bool normals_exist,
   bool textures_exist
)
{
   assert( VBO != 0 );

   VerticesCount = 0;
   int step = 3;
   if (normals_exist) step += 3;
   if (textures_exist) step += 2;
   for (size_t i = 0, j = 0; i < vertices.size(); i += 3, ++j) {
      DataBuffer[j * step] = vertices[i];
      DataBuffer[j * step + 1] = vertices[i + 1];
      DataBuffer[j * step + 2] = vertices[i + 2];
      VerticesCount++;
   }
   glNamedBufferSubData( VBO, 0, static_cast<int>(sizeof( GLfloat ) * VerticesCount * step), DataBuffer.data() );
}