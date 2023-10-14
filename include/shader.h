#pragma once

#include "base.h"

class ShaderGL
{
public:
   ShaderGL();
   virtual ~ShaderGL();

   void setShader(
      const char* vertex_shader_path,
      const char* fragment_shader_path,
      const char* geometry_shader_path = nullptr,
      const char* tessellation_control_shader_path = nullptr,
      const char* tessellation_evaluation_shader_path = nullptr
   );
   void setComputeShader(const char* compute_shader_path);
   void uniform1i(int location, int value) const
   {
      glProgramUniform1i( ShaderProgram, location, value );
   }
   void uniform1ui(int location, uint value) const
   {
      glProgramUniform1ui( ShaderProgram, location, value );
   }
   void uniform1iv(int location, int count, const int* value) const
   {
      glProgramUniform1iv( ShaderProgram, location, count, value );
   }
   void uniform1f(int location, float value) const
   {
      glProgramUniform1f( ShaderProgram, location, value );
   }
   void uniform1fv(int location, int count, const float* value) const
   {
      glProgramUniform1fv( ShaderProgram, location, count, value );
   }
   void uniform2iv(int location, const glm::ivec2& value) const
   {
      glProgramUniform2iv( ShaderProgram, location, 1, &value[0] );
   }
   void uniform2fv(int location, const glm::vec2& value) const
   {
      glProgramUniform2fv( ShaderProgram, location, 1, &value[0] );
   }
   void uniform2fv(int location, int count, const glm::vec2* value) const
   {
      glProgramUniform2fv( ShaderProgram, location, count, glm::value_ptr( *value ) );
   }
   void uniform2fv(int location, int count, const float* value) const
   {
      glProgramUniform2fv( ShaderProgram, location, count, value );
   }
   void uniform3fv(int location, const glm::vec3& value) const
   {
      glProgramUniform3fv( ShaderProgram, location, 1, &value[0] );
   }
   void uniform3fv(int location, int count, const glm::vec3* value) const
   {
      glProgramUniform3fv( ShaderProgram, location, count, glm::value_ptr( *value ) );
   }
   void uniform3fv(int location, int count, const float* value) const
   {
      glProgramUniform3fv( ShaderProgram, location, count, value );
   }
   void uniform4fv(int location, const glm::vec4& value) const
   {
      glProgramUniform4fv( ShaderProgram, location, 1, &value[0] );
   }
   void uniform4fv(int location, int count, const float* value) const
   {
      glProgramUniform4fv( ShaderProgram, location, count, value );
   }
   void uniformMat3fv(int location, const glm::mat3& value) const
   {
      glProgramUniformMatrix3fv( ShaderProgram, location, 1, GL_FALSE, glm::value_ptr( value ) );
   }
   void uniformMat4fv(int location, const glm::mat4& value) const
   {
      glProgramUniformMatrix4fv( ShaderProgram, location, 1, GL_FALSE, glm::value_ptr( value ) );
   }
   void uniformMat4fv(int location, int count, const glm::mat4* value) const
   {
      glProgramUniformMatrix4fv( ShaderProgram, location, count, GL_FALSE, glm::value_ptr( *value ) );
   }
   void uniformMat43fv(int location, const glm::mat<3, 4, float, glm::highp>& value) const
   {
      glProgramUniformMatrix4x3fv( ShaderProgram, location, 1, GL_FALSE, glm::value_ptr( value ) );
   }
   [[nodiscard]] GLuint getShaderProgram() const { return ShaderProgram; }

protected:
   GLuint ShaderProgram;

   static void readShaderFile(std::string& shader_contents, const char* shader_path);
   [[nodiscard]] static std::string getShaderTypeString(GLenum shader_type);
   [[nodiscard]] static bool checkCompileError(GLenum shader_type, const GLuint& shader);
   [[nodiscard]] static GLuint getCompiledShader(GLenum shader_type, const char* shader_path);
};

class SceneShaderGL final : public ShaderGL
{
public:
   enum UNIFORM {
      WorldMatrix = 0,
      ViewMatrix,
      ModelViewProjectionMatrix,
      Lights,
      Material = 291,
      UseTexture = 296,
      UseLight,
      LightNum,
      GlobalAmbient,

      LightSwitch = 0,
      LightPosition,
      LightAmbientColor,
      LightDiffuseColor,
      LightSpecularColor,
      SpotlightDirection,
      SpotlightCutoffAngle,
      SpotlightFeather,
      FallOffRadius,
      LightUniformNum,

      MaterialEmissionColor = 0,
      MaterialAmbientColor,
      MaterialDiffuseColor,
      MaterialSpecularColor,
      MaterialSpecularExponent
   };

   SceneShaderGL() = default;
   ~SceneShaderGL() override = default;

   SceneShaderGL(const SceneShaderGL&) = delete;
   SceneShaderGL(const SceneShaderGL&&) = delete;
   SceneShaderGL& operator=(const SceneShaderGL&) = delete;
   SceneShaderGL& operator=(const SceneShaderGL&&) = delete;
};

class PointShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { ModelViewProjectionMatrix = 0 };

   PointShaderGL() = default;
   ~PointShaderGL() override = default;

   PointShaderGL(const PointShaderGL&) = delete;
   PointShaderGL(const PointShaderGL&&) = delete;
   PointShaderGL& operator=(const PointShaderGL&) = delete;
   PointShaderGL& operator=(const PointShaderGL&&) = delete;
};

class TextShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { ModelViewProjectionMatrix = 0, TextScale };

   TextShaderGL() = default;
   ~TextShaderGL() override = default;

   TextShaderGL(const TextShaderGL&) = delete;
   TextShaderGL(const TextShaderGL&&) = delete;
   TextShaderGL& operator=(const TextShaderGL&) = delete;
   TextShaderGL& operator=(const TextShaderGL&&) = delete;
};