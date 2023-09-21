#version 460

uniform mat4 WorldMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 ModelViewProjectionMatrix;

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_tex_coord;

out vec3 position_in_ec;
out vec3 normal_in_ec;
out vec2 tex_coord;

void main()
{   
   vec4 e_position = ViewMatrix * WorldMatrix * vec4(v_position, 1.0f);
   // As ViewMatrix * WorldMatrix is rigid body transformation,
   // transpose( inverse( ViewMatrix * WorldMatrix ) ) is equal to ViewMatrix * WorldMatrix.
   // So it is possible to avoid the costly operation to calculate the tranformation for normals.
   vec4 e_normal = ViewMatrix * WorldMatrix * vec4(v_normal, 0.0f);
   position_in_ec = e_position.xyz;
   normal_in_ec = normalize( e_normal.xyz );

   tex_coord = v_tex_coord;

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}