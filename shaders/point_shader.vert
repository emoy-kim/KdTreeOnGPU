#version 460

uniform mat4 WorldMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 ModelViewProjectionMatrix;

layout (location = 0) in vec3 v_position;

out vec3 color;

void main()
{
   color = gl_VertexID == 0 ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 0.0f, 1.0f);

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}