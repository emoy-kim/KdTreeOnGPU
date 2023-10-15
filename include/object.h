#pragma once

#include "base.h"

class ObjectGL
{
public:
   enum LayoutLocation { VertexLoc = 0, NormalLoc, TextureLoc };

   ObjectGL();
   virtual ~ObjectGL();

   void setEmissionColor(const glm::vec4& emission_color);
   void setAmbientReflectionColor(const glm::vec4& ambient_reflection_color);
   void setDiffuseReflectionColor(const glm::vec4& diffuse_reflection_color);
   void setSpecularReflectionColor(const glm::vec4& specular_reflection_color);
   void setSpecularReflectionExponent(const float& specular_reflection_exponent);
   void setObject(GLenum draw_mode, const std::vector<glm::vec3>& vertices);
   void setObject(
      GLenum draw_mode,
      const std::vector<glm::vec3>& vertices,
      const std::vector<glm::vec3>& normals
   );
   void setObject(
      GLenum draw_mode,
      const std::vector<glm::vec3>& vertices,
      const std::vector<glm::vec3>& normals,
      const std::vector<glm::vec2>& textures
   );
   void setObject(
      GLenum draw_mode,
      const std::vector<glm::vec3>& vertices,
      const std::vector<glm::vec3>& normals,
      const std::vector<glm::vec2>& textures,
      const std::string& texture_file_path,
      bool is_grayscale = false
   );
   virtual void setObject(GLenum draw_mode, const std::string& obj_file_path);
   void setSquareObject(GLenum draw_mode, bool use_texture = true);
   void setSquareObject(
      GLenum draw_mode,
      const std::string& texture_file_path,
      bool is_grayscale = false
   );
   int addTexture(const std::string& texture_file_path, bool is_grayscale = false);
   void addTexture(int width, int height, bool is_grayscale = false);
   int addTexture(const uint8_t* image_buffer, int width, int height, bool is_grayscale = false);
   void replaceVertices(const std::vector<glm::vec3>& vertices, bool normals_exist, bool textures_exist);
   void replaceVertices(const std::vector<float>& vertices, bool normals_exist, bool textures_exist);
   void releaseCustomBuffer(GLuint& id)
   {
      const auto it = std::find_if( CustomBuffers.begin(), CustomBuffers.end(), [id](auto v) { return id == v; } );
      if (it != CustomBuffers.end()) {
         glDeleteBuffers( 1, &(*it) );
         CustomBuffers.erase( it );
         id = 0;
      }
   }
   [[nodiscard]] GLuint getVAO() const { return VAO; }
   [[nodiscard]] GLuint getVBO() const { return VBO; }
   [[nodiscard]] GLuint getIBO() const { return IBO; }
   [[nodiscard]] GLenum getDrawMode() const { return DrawMode; }
   [[nodiscard]] GLsizei getVertexNum() const { return VerticesCount; }
   [[nodiscard]] GLsizei getIndexNum() const { return static_cast<GLsizei>(IndexBuffer.size()); }
   [[nodiscard]] GLuint getTextureID(int index) const { return TextureID[index]; }
   [[nodiscard]] int getTextureNum() const { return static_cast<int>(TextureID.size()); }
   [[nodiscard]] glm::vec4 getEmissionColor() const { return EmissionColor; }
   [[nodiscard]] glm::vec4 getAmbientReflectionColor() const { return AmbientReflectionColor; }
   [[nodiscard]] glm::vec4 getDiffuseReflectionColor() const { return DiffuseReflectionColor; }
   [[nodiscard]] glm::vec4 getSpecularReflectionColor() const { return SpecularReflectionColor; }
   [[nodiscard]] float getSpecularReflectionExponent() const { return SpecularReflectionExponent; }

   template<typename T>
   [[nodiscard]] GLuint addCustomBufferObject(int data_size)
   {
      GLuint buffer = 0;
      glCreateBuffers( 1, &buffer );
      glNamedBufferStorage( buffer, sizeof( T ) * data_size, nullptr, GL_DYNAMIC_STORAGE_BIT );
      CustomBuffers.emplace_back( buffer );
      return buffer;
   }

protected:
   GLuint VAO;
   GLuint VBO;
   GLuint IBO;
   GLenum DrawMode;
   GLsizei VerticesCount;
   std::vector<GLuint> TextureID;
   std::vector<GLfloat> DataBuffer;
   std::vector<GLuint> IndexBuffer;
   std::vector<GLuint> CustomBuffers;
   glm::vec4 EmissionColor;
   glm::vec4 AmbientReflectionColor; // It is usually set to the same color with DiffuseReflectionColor.
                                     // Otherwise, it should be in balance with DiffuseReflectionColor.
   glm::vec4 DiffuseReflectionColor; // the intrinsic color
   glm::vec4 SpecularReflectionColor;
   float SpecularReflectionExponent;

   [[nodiscard]] bool prepareTexture2DUsingFreeImage(const std::string& file_path, bool is_grayscale) const;
   void prepareNormal() const;
   void prepareTexture(bool normals_exist) const;
   void prepareVertexBuffer(int n_bytes_per_vertex);
   void prepareIndexBuffer();
   static void getSquareObject(
      std::vector<glm::vec3>& vertices,
      std::vector<glm::vec3>& normals,
      std::vector<glm::vec2>& textures
   );
   static void findNormals(
      std::vector<glm::vec3>& normals,
      const std::vector<glm::vec3>& vertices,
      const std::vector<GLuint>& vertex_indices
   );
   [[nodiscard]] bool readObjectFile(
      std::vector<glm::vec3>& vertices,
      std::vector<glm::vec3>& normals,
      std::vector<glm::vec2>& textures,
      const std::string& file_path
   );
};