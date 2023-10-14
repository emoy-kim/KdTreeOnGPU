#version 460

layout (location = 0) uniform mat4 WorldMatrix;
layout (location = 1) uniform mat4 ViewMatrix;
layout (location = 2) uniform mat4 ModelViewProjectionMatrix;

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_tex_coord;

out vec3 position_in_ec;
out vec3 normal_in_ec;
out vec2 tex_coord;

void main()
{   
   vec4 e_position = ViewMatrix * WorldMatrix * vec4(v_position, 1.0f);
   vec4 e_normal = transpose( inverse( ViewMatrix * WorldMatrix ) ) * vec4(v_normal, 0.0f);
   position_in_ec = e_position.xyz;
   normal_in_ec = normalize( e_normal.xyz );

   tex_coord = v_tex_coord;

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}