﻿#pragma once

#include "base.h"
#include "camera.h"

class ShaderGL
{
public:
   struct LightLocationSet
   {
      GLint LightSwitch, LightPosition;
      GLint LightAmbient, LightDiffuse, LightSpecular, LightFallOffRadius;
      GLint SpotlightDirection, SpotlightCutoffAngle, SpotlightFeather;

      LightLocationSet() : LightSwitch( 0 ), LightPosition( 0 ), LightAmbient( 0 ), LightDiffuse( 0 ),
      LightSpecular( 0 ), LightFallOffRadius( 0 ), SpotlightDirection( 0 ), SpotlightCutoffAngle( 0 ),
      SpotlightFeather( 0 ) {}
   };

   struct LocationSet
   {
      GLint World, View, Projection, ModelViewProjection;
      GLint MaterialEmission, MaterialAmbient, MaterialDiffuse, MaterialSpecular, MaterialSpecularExponent;
      std::map<GLint, GLint> Texture; // <binding point, texture id>
      GLint UseLight, LightNum, GlobalAmbient;
      std::vector<LightLocationSet> Lights;

      LocationSet() : World( 0 ), View( 0 ), Projection( 0 ), ModelViewProjection( 0 ), MaterialEmission( 0 ),
      MaterialAmbient( 0 ), MaterialDiffuse( 0 ), MaterialSpecular( 0 ), MaterialSpecularExponent( 0 ), UseLight( 0 ),
      LightNum( 0 ), GlobalAmbient( 0 ) {}
   };

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
   void setTextUniformLocations();
   void setPointUniformLocations();
   void setSceneUniformLocations(int light_num);
   virtual void setUniformLocations() {}
   void addUniformLocation(const std::string& name)
   {
      CustomLocations[name] = glGetUniformLocation( ShaderProgram, name.c_str() );
   }
   void transferBasicTransformationUniforms(const glm::mat4& to_world, const CameraGL* camera) const;
   void uniform1i(const char* name, int value) const
   {
      glProgramUniform1i( ShaderProgram, CustomLocations.find( name )->second, value );
   }
   void uniform1ui(const char* name, uint value) const
   {
      glProgramUniform1ui( ShaderProgram, CustomLocations.find( name )->second, value );
   }
   void uniform1f(const char* name, float value) const
   {
      glProgramUniform1f( ShaderProgram, CustomLocations.find( name )->second, value );
   }
   void uniform1fv(const char* name, int count, const float* value) const
   {
      glProgramUniform1fv( ShaderProgram, CustomLocations.find( name )->second, count, value );
   }
   void uniform2iv(const char* name, const glm::ivec2& value) const
   {
      glProgramUniform2iv( ShaderProgram, CustomLocations.find( name )->second, 1, &value[0] );
   }
   void uniform2fv(const char* name, const glm::vec2& value) const
   {
      glProgramUniform2fv( ShaderProgram, CustomLocations.find( name )->second, 1, &value[0] );
   }
   void uniform2fv(const char* name, int count, const float* value) const
   {
      glProgramUniform2fv( ShaderProgram, CustomLocations.find( name )->second, count, value );
   }
   void uniform3fv(const char* name, const glm::vec3& value) const
   {
      glProgramUniform3fv( ShaderProgram, CustomLocations.find( name )->second, 1, &value[0] );
   }
   void uniform4fv(const char* name, const glm::vec4& value) const
   {
      glProgramUniform4fv( ShaderProgram, CustomLocations.find( name )->second, 1, &value[0] );
   }
   void uniformMat3fv(const char* name, const glm::mat3& value) const
   {
      glProgramUniformMatrix3fv( ShaderProgram, CustomLocations.find( name )->second, 1, GL_FALSE, &value[0][0] );
   }
   void uniformMat4fv(const char* name, const glm::mat4& value) const
   {
      glProgramUniformMatrix4fv( ShaderProgram, CustomLocations.find( name )->second, 1, GL_FALSE, &value[0][0] );
   }
   void uniformMat4fv(const char* name, const std::vector<glm::mat4>& value) const
   {
      glProgramUniformMatrix4fv( ShaderProgram, CustomLocations.find( name )->second, value.size(), GL_FALSE, &value[0][0][0] );
   }
   void uniformMat4fv(const char* name, int count, const glm::mat4* value) const
   {
      glProgramUniformMatrix4fv( ShaderProgram, CustomLocations.find( name )->second, count, GL_FALSE, glm::value_ptr( *value ) );
   }
   [[nodiscard]] GLuint getShaderProgram() const { return ShaderProgram; }
   [[nodiscard]] GLint getLocation(const std::string& name) const { return CustomLocations.find( name )->second; }
   [[nodiscard]] GLint getMaterialEmissionLocation() const { return Location.MaterialEmission; }
   [[nodiscard]] GLint getMaterialAmbientLocation() const { return Location.MaterialAmbient; }
   [[nodiscard]] GLint getMaterialDiffuseLocation() const { return Location.MaterialDiffuse; }
   [[nodiscard]] GLint getMaterialSpecularLocation() const { return Location.MaterialSpecular; }
   [[nodiscard]] GLint getMaterialSpecularExponentLocation() const { return Location.MaterialSpecularExponent; }
   [[nodiscard]] GLint getLightAvailabilityLocation() const { return Location.UseLight; }
   [[nodiscard]] GLint getLightNumLocation() const { return Location.LightNum; }
   [[nodiscard]] GLint getGlobalAmbientLocation() const { return Location.GlobalAmbient; }
   [[nodiscard]] GLint getLightSwitchLocation(int light_index) const
   {
      return Location.Lights[light_index].LightSwitch;
   }
   [[nodiscard]] GLint getLightPositionLocation(int light_index) const
   {
      return Location.Lights[light_index].LightPosition;
   }
   [[nodiscard]] GLint getLightAmbientLocation(int light_index) const
   {
      return Location.Lights[light_index].LightAmbient;
   }
   [[nodiscard]] GLint getLightDiffuseLocation(int light_index) const
   {
      return Location.Lights[light_index].LightDiffuse;
   }
   [[nodiscard]] GLint getLightSpecularLocation(int light_index) const
   {
      return Location.Lights[light_index].LightSpecular;
   }
   [[nodiscard]] GLint getLightSpotlightDirectionLocation(int light_index) const
   {
      return Location.Lights[light_index].SpotlightDirection;
   }
   [[nodiscard]] GLint getLightSpotlightCutoffAngleLocation(int light_index) const
   {
      return Location.Lights[light_index].SpotlightCutoffAngle;
   }
   [[nodiscard]] GLint getLightSpotlightFeatherLocation(int light_index) const
   {
      return Location.Lights[light_index].SpotlightFeather;
   }
   [[nodiscard]] GLint getLightFallOffRadiusLocation(int light_index) const
   {
      return Location.Lights[light_index].LightFallOffRadius;
   }

protected:
   GLuint ShaderProgram;
   LocationSet Location;
   std::unordered_map<std::string, GLint> CustomLocations;

   static void readShaderFile(std::string& shader_contents, const char* shader_path);
   [[nodiscard]] static std::string getShaderTypeString(GLenum shader_type);
   [[nodiscard]] static bool checkCompileError(GLenum shader_type, const GLuint& shader);
   [[nodiscard]] static GLuint getCompiledShader(GLenum shader_type, const char* shader_path);
   void setBasicTransformationUniforms();
};