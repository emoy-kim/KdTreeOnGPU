#version 460

layout (location = 0) uniform mat4 ModelViewProjectionMatrix;

layout (location = 0) in vec3 v_position;

out vec3 color;

void main()
{
   color = gl_VertexID == 0 ? vec3(0.957f, 0.263f, 0.212f) : vec3(0.161f, 0.525f, 0.8f);

   gl_Position = ModelViewProjectionMatrix * vec4(v_position, 1.0f);
}