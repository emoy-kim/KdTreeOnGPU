#include "light.h"

LightGL::LightGL() :
   TurnLightOn( true ), GlobalAmbientColor( 0.2f, 0.2f, 0.2f, 1.0f ), TotalLightNum( 0 )
{
}

bool LightGL::isLightOn() const
{
   return TurnLightOn;
}

void LightGL::toggleLightSwitch()
{
   TurnLightOn = !TurnLightOn;
}

void LightGL::addLight(
   const glm::vec4& light_position,
   const glm::vec4& ambient_color,
   const glm::vec4& diffuse_color,
   const glm::vec4& specular_color,
   const glm::vec3& spotlight_direction,
   float spotlight_cutoff_angle_in_degree,
   float spotlight_feather,
   float falloff_radius
)
{
   Positions.emplace_back( light_position );

   AmbientColors.emplace_back( ambient_color );
   DiffuseColors.emplace_back( diffuse_color );
   SpecularColors.emplace_back( specular_color );

   SpotlightDirections.emplace_back( spotlight_direction );
   SpotlightCutoffAngles.emplace_back( spotlight_cutoff_angle_in_degree );
   SpotlightFeathers.emplace_back( spotlight_feather );
   FallOffRadii.emplace_back( falloff_radius );

   IsActivated.emplace_back( true );

   TotalLightNum = static_cast<int>(Positions.size());
}

void LightGL::activateLight(const int& light_index)
{
   if (light_index >= TotalLightNum) return;
   IsActivated[light_index] = true;
}

void LightGL::deactivateLight(const int& light_index)
{
   if (light_index >= TotalLightNum) return;
   IsActivated[light_index] = false;
}

void LightGL::transferUniformsToShader(const ShaderGL* shader)
{
   glUniform1i( shader->getLightAvailabilityLocation(), TurnLightOn ? 1 : 0 );
   glUniform1i( shader->getLightNumLocation(), static_cast<GLint>(TotalLightNum) );
   glUniform4fv( shader->getGlobalAmbientLocation(), 1, &GlobalAmbientColor[0] );

   for (int i = 0; i < TotalLightNum; ++i) {
      glUniform1i( shader->getLightSwitchLocation( i ), IsActivated[0] ? 1 : 0 );
      glUniform4fv( shader->getLightPositionLocation( i ), 1, &Positions[i][0] );
      glUniform4fv( shader->getLightAmbientLocation( i ), 1, &AmbientColors[i][0] );
      glUniform4fv( shader->getLightDiffuseLocation( i ), 1, &DiffuseColors[i][0] );
      glUniform4fv( shader->getLightSpecularLocation( i ), 1, &SpecularColors[i][0] );
      glUniform3fv( shader->getLightSpotlightDirectionLocation( i ), 1, &SpotlightDirections[i][0] );
      glUniform1f( shader->getLightSpotlightCutoffAngleLocation( i ), SpotlightCutoffAngles[i] );
      glUniform1f( shader->getLightSpotlightFeatherLocation( i ), SpotlightFeathers[i] );
      glUniform1f( shader->getLightFallOffRadiusLocation( i ), FallOffRadii[i] );
   }
}