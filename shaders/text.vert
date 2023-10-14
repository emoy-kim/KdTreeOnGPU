#version 460

layout (location = 0) uniform mat4 ModelViewProjectionMatrix;
layout (location = 1) uniform vec2 TextScale;

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_tex_coord;

out vec2 tex_coord;

void main()
{
   tex_coord = v_tex_coord * TextScale;

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}