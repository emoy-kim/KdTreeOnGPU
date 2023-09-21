/*
 * Author: Jeesun Kim
 * E-mail: emoy.kim_AT_gmail.com
 *
 */

#pragma once

#include "base.h"
#include "text.h"
#include "light.h"
#include "kdtree.h"

class RendererGL final
{
public:
   RendererGL();
   ~RendererGL();

   RendererGL(const RendererGL&) = delete;
   RendererGL(const RendererGL&&) = delete;
   RendererGL& operator=(const RendererGL&) = delete;
   RendererGL& operator=(const RendererGL&&) = delete;

   void play();

private:
   inline static RendererGL* Renderer = nullptr;
   GLFWwindow* Window;
   bool Pause;
   int FrameWidth;
   int FrameHeight;
   int ShadowMapSize;
   float BoxHalfSide;
   GLuint DepthFBO;
   GLuint DepthTextureArrayID;
   glm::ivec2 ClickedPoint;
   CameraGL* ActiveCamera;
   std::unique_ptr<TextGL> Texter;
   std::unique_ptr<CameraGL> MainCamera;
   std::unique_ptr<CameraGL> TextCamera;
   std::vector<std::unique_ptr<CameraGL>> LightCameras;
   std::unique_ptr<ShaderGL> TextShader;
   std::unique_ptr<ShaderGL> PCFSceneShader;
   std::unique_ptr<ShaderGL> LightViewDepthShader;
   std::unique_ptr<LightGL> Lights;
   std::unique_ptr<ObjectGL> Object;
   std::unique_ptr<ObjectGL> WallObject;
   std::array<int, 2> WallTextureIndices;
   std::vector<glm::mat4> LightViewMatrices;
   std::vector<glm::mat4> LightViewProjectionMatrices;

   // 16 and 32 do well, anything in between or below is bad.
   // 32 seems to do well on laptop/desktop Windows Intel and on NVidia/AMD as well.
   // further hardware-specific tuning might be needed for optimal performance.
   static constexpr int ThreadGroupSize = 32;
   [[nodiscard]] static int getGroupSize(int size)
   {
      return (size + ThreadGroupSize - 1) / ThreadGroupSize;
   }

   void registerCallbacks() const;
   void initialize();
   void writeFrame() const;
   void writeDepthTextureArray() const;
   static void printOpenGLInformation();
   static void cleanup(GLFWwindow* window);
   static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
   static void cursor(GLFWwindow* window, double xpos, double ypos);
   static void mouse(GLFWwindow* window, int button, int action, int mods);

   void setLights();
   void setObject() const;
   void setWallObject();
   void setLightViewFrameBuffers();
   void drawObject(ShaderGL* shader, CameraGL* camera) const;
   void drawBoxObject(ShaderGL* shader, const CameraGL* camera) const;
   void drawDepthMapFromLightView();
   void drawShadow() const;
   void drawText(const std::string& text, glm::vec2 start_position) const;
   void render();
};