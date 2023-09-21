#pragma once

#include "shader.h"

class LightGL final
{
public:
   LightGL();
   ~LightGL() = default;

   [[nodiscard]] bool isLightOn() const;
   void toggleLightSwitch();
   void addLight(
      const glm::vec4& light_position,
      const glm::vec4& ambient_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
      const glm::vec4& diffuse_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
      const glm::vec4& specular_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
      const glm::vec3& spotlight_direction = glm::vec3(0.0f, 0.0f, -1.0f),
      float spotlight_cutoff_angle_in_degree = 180.0f,
      float spotlight_feather = 0.0f,
      float falloff_radius = 1000.0f
   );
   void activateLight(const int& light_index);
   void deactivateLight(const int& light_index);
   void transferUniformsToShader(const ShaderGL* shader);
   [[nodiscard]] int getTotalLightNum() const { return TotalLightNum; }
   [[nodiscard]] glm::vec4 getLightPosition(int light_index) { return Positions[light_index]; }
   [[nodiscard]] glm::vec3 getSpotlightDirection(int light_index) { return SpotlightDirections[light_index]; }

private:
   bool TurnLightOn;
   int TotalLightNum;
   glm::vec4 GlobalAmbientColor;
   std::vector<bool> IsActivated;
   std::vector<glm::vec4> Positions;
   std::vector<glm::vec4> AmbientColors;
   std::vector<glm::vec4> DiffuseColors;
   std::vector<glm::vec4> SpecularColors;
   std::vector<glm::vec3> SpotlightDirections;
   std::vector<float> SpotlightCutoffAngles;
   std::vector<float> SpotlightFeathers;
   std::vector<float> FallOffRadii;
};