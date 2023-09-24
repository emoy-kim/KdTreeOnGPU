#version 460

in vec3 color;

layout (location = 0) out vec4 final_color;

void main()
{
   final_color = vec4(color, 1.0f);
}