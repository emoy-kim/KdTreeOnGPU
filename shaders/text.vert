#version 460

uniform mat4 WorldMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 LightModelViewProjectionMatrix;
uniform vec2 TextScale;

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_tex_coord;

out vec2 tex_coord;

void main()
{
   tex_coord = v_tex_coord * TextScale;

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}