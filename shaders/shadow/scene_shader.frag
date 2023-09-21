#version 460

#define MAX_LIGHTS 32

struct LightInfo
{
   int LightSwitch;
   vec4 Position;
   vec4 AmbientColor;
   vec4 DiffuseColor;
   vec4 SpecularColor;
   vec3 SpotlightDirection;
   float SpotlightCutoffAngle;
   float SpotlightFeather;
   float FallOffRadius;
};
uniform LightInfo Lights[MAX_LIGHTS];

struct MateralInfo {
   vec4 EmissionColor;
   vec4 AmbientColor;
   vec4 DiffuseColor;
   vec4 SpecularColor;
   float SpecularExponent;
};
uniform MateralInfo Material;

layout (binding = 0) uniform sampler2D BaseTexture;
layout (binding = 1) uniform sampler2DArrayShadow DepthMap;

uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 LightViewMatrix[2];
uniform mat4 LightViewProjectionMatrix[2];
uniform int UseTexture;
uniform int UseLight;
uniform int LightNum;
uniform vec4 GlobalAmbient;
uniform vec4 ShadowColor;

in vec4 position_in_wc;
in vec3 position_in_ec;
in vec3 normal_in_ec;
in vec2 tex_coord;

layout (location = 0) out vec4 final_color;

const float zero = 0.0f;
const float one = 1.0f;
const float half_pi = 1.57079632679489661923132169163975144f;

bool IsPointLight(in vec4 light_position)
{
   return light_position.w != zero;
}

float getAttenuation(in vec3 light_vector, in int light_index)
{
   float squared_distance = dot( light_vector, light_vector );
   float distance = sqrt( squared_distance );
   float radius = Lights[light_index].FallOffRadius;
   if (distance <= radius) return one;

   return clamp( radius * radius / squared_distance, zero, one );
}

float getSpotlightFactor(in vec3 normalized_light_vector, in int light_index)
{
   if (Lights[light_index].SpotlightCutoffAngle >= 180.0f) return one;

   vec4 direction_in_ec = transpose( inverse( ViewMatrix ) ) * vec4(Lights[light_index].SpotlightDirection, zero);
   vec3 normalized_direction = normalize( direction_in_ec.xyz );
   float factor = dot( -normalized_light_vector, normalized_direction );
   float cutoff_angle = radians( clamp( Lights[light_index].SpotlightCutoffAngle, zero, 90.0f ) );
   if (factor >= cos( cutoff_angle )) {
      float normalized_angle = acos( factor ) * half_pi / cutoff_angle;
      float threshold = half_pi * (one - Lights[light_index].SpotlightFeather);
      return normalized_angle <= threshold ? one :
         cos( half_pi * (normalized_angle - threshold) / (half_pi - threshold) );
   }
   return zero;
}

float getShadowWithPCF(in int light_index)
{
   const float bias_for_shadow_acne = 0.005f;
   vec4 position_in_light_cc = LightViewProjectionMatrix[light_index] * position_in_wc;
   vec4 depth_map_coord = vec4(
      0.5f * position_in_light_cc.xyz / position_in_light_cc.w + 0.5f,
      position_in_light_cc.w
   );
   depth_map_coord.z -= bias_for_shadow_acne;

   vec2 dx = dFdx( depth_map_coord.xy );
   vec2 dy = dFdy( depth_map_coord.xy );

   const float epsilon = 1e-2f;
   const vec2 min_size = vec2(one);
   const vec2 max_size = vec2(32.0f);
   if (epsilon <= depth_map_coord.x && depth_map_coord.x <= one - epsilon &&
       epsilon <= depth_map_coord.y && depth_map_coord.y <= one - epsilon &&
       zero < depth_map_coord.w) {
      vec2 shadow_size = vec2(textureSize( DepthMap, 0 ));
      vec2 normalizer = one / shadow_size;

      // this filter is designed to reduce artifacts.
      // the filter size gets increased when the camera is closer to hide artifacts.
      vec2 filter_size = smoothstep( 1.05f - abs( dx ) - abs( dy ), vec2(zero), vec2(one) );
      filter_size = round( clamp( filter_size * shadow_size, min_size, shadow_size * 0.1f ) );

      ivec2 window_size = ivec2(filter_size);
      filter_size *= normalizer;
      vec2 lower_left = depth_map_coord.xy - 0.5f * filter_size;
      float shadow = zero;
      for (int y = 0; y < window_size.y; ++y) {
         for (int x = 0; x < window_size.x; ++x) {
            vec4 tex_coord = vec4(lower_left + vec2(x, y) * normalizer, float(light_index), depth_map_coord.z);
            shadow += texture( DepthMap, tex_coord );
         }
      }
      return shadow / float(window_size.x * window_size.y);
   }
   return one;
}

float calculateSuperellipseShaping(in vec3 position_in_light, in bool barn_shaping)
{
   if (!barn_shaping) return one;

   const float width = 128.0f;
   const float height = 128.0f;
   const float width_with_edge = 150.0f;
   const float height_with_edge = 150.0f;
   const float round = 16.5f;
   vec2 projected_onto_z_is_one = position_in_light.xy / position_in_light.z;
   vec2 p = abs( projected_onto_z_is_one );
   float a = 2.0f / round;
   float b = -round * 0.5f;
   float inner = width * height * pow( pow( height * p.x, a ) + pow( width * p.y, a ), b );
   float outer = width_with_edge * height_with_edge *
      pow( pow( height_with_edge * p.x, a ) + pow( width_with_edge * p.y, a ), b );
   return one - smoothstep( inner, outer, one );
}

float calculateDistanceShaping(in vec3 position_in_light, in bool barn_shaping)
{
   const float near = 100.0f;
   const float far = 1000.0f;
   const float near_edge = 5.0f;
   const float far_edge = 5.0f;
   float depth = barn_shaping ? -position_in_light.z : abs( position_in_light.z );
   return smoothstep( near - near_edge, near, depth ) * (one - smoothstep( far, far + far_edge, depth ));
}

vec4 calculateUberlight()
{
   vec4 color = Material.EmissionColor + GlobalAmbient * Material.AmbientColor;

   for (int i = 0; i < LightNum; ++i) {
      if (!bool(Lights[i].LightSwitch)) continue;

      vec4 light_position_in_ec = ViewMatrix * Lights[i].Position;
      vec4 position_in_light = LightViewMatrix[i] * position_in_wc;
      float final_effect_factor =
         calculateSuperellipseShaping( position_in_light.xyz, true ) *
         calculateDistanceShaping( position_in_light.xyz, true );
      vec3 light_vector = light_position_in_ec.xyz - position_in_ec;
      if (IsPointLight( light_position_in_ec )) {
         float attenuation = getAttenuation( light_vector, i );

         light_vector = normalize( light_vector );
         float spotlight_factor = getSpotlightFactor( light_vector, i );
         final_effect_factor *= attenuation * spotlight_factor;
      }
      else light_vector = normalize( light_position_in_ec.xyz );

      if (final_effect_factor <= zero) continue;

      float shadow = getShadowWithPCF( i );
      vec4 local_color = mix( ShadowColor, Lights[i].AmbientColor, shadow ) * Material.AmbientColor;

      float diffuse_intensity = max( dot( normal_in_ec, light_vector ), zero );
      local_color += diffuse_intensity * mix( ShadowColor, Lights[i].DiffuseColor, shadow ) * Material.DiffuseColor;

      vec3 halfway_vector = normalize( light_vector - normalize( position_in_ec ) );
      float specular_intensity = max( dot( normal_in_ec, halfway_vector ), zero );
      local_color +=
         pow( specular_intensity, Material.SpecularExponent ) *
         mix( ShadowColor, Lights[i].SpecularColor, shadow ) * Material.SpecularColor;

      color += local_color * final_effect_factor;
   }
   return color;
}

void main()
{
   final_color = bool(UseTexture) ? texture( BaseTexture, tex_coord ) : vec4(one);
   final_color *= bool(UseLight) ? calculateUberlight() : Material.DiffuseColor;
}