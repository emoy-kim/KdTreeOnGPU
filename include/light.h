#pragma once

#include "base.h"

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
   [[nodiscard]] int getTotalLightNum() const { return TotalLightNum; }
   [[nodiscard]] glm::vec4 getGlobalAmbientColor() { return GlobalAmbientColor; }
   [[nodiscard]] bool isActivated(int light_index) { return IsActivated[light_index]; }
   [[nodiscard]] glm::vec4 getPosition(int light_index) { return Positions[light_index]; }
   [[nodiscard]] glm::vec4 getAmbientColors(int light_index) { return AmbientColors[light_index]; }
   [[nodiscard]] glm::vec4 getDiffuseColors(int light_index) { return DiffuseColors[light_index]; }
   [[nodiscard]] glm::vec4 getSpecularColors(int light_index) { return SpecularColors[light_index]; }
   [[nodiscard]] glm::vec3 getSpotlightDirections(int light_index) { return SpotlightDirections[light_index]; }
   [[nodiscard]] float getSpotlightCutoffAngles(int light_index) { return SpotlightCutoffAngles[light_index]; }
   [[nodiscard]] float getSpotlightFeathers(int light_index) { return SpotlightFeathers[light_index]; }
   [[nodiscard]] float getFallOffRadii(int light_index) { return FallOffRadii[light_index]; }

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