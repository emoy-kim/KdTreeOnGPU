/*
 * Author: Jeesun Kim
 * E-mail: emoy.kim_AT_gmail.com
 *
 */

#pragma once

#include "base.h"
#include "text.h"
#include "light.h"
#include "kdtree_object.h"
#include "kdtree_shader.h"
#include "cuda/kdtree.cuh"

class RendererGL final
{
public:
   RendererGL();
   ~RendererGL() = default;

   RendererGL(const RendererGL&) = delete;
   RendererGL(const RendererGL&&) = delete;
   RendererGL& operator=(const RendererGL&) = delete;
   RendererGL& operator=(const RendererGL&&) = delete;

   void play();

private:
   struct TimeCheck
   {
      double ObjectLoad;
      double Sort;
      double Build;
      double Verify;

      TimeCheck() = default;
   };

   struct KdtreeBuild
   {
      std::unique_ptr<InitializeShaderGL> Initialize;
      std::unique_ptr<InitializeReferenceShaderGL> InitializeReference;
      std::unique_ptr<CopyCoordinatesShaderGL> CopyCoordinates;
      std::unique_ptr<SortByBlockShaderGL> SortByBlock;
      std::unique_ptr<SortLastBlockShaderGL> SortLastBlock;
      std::unique_ptr<GenerateSampleRanksShaderGL> GenerateSampleRanks;
      std::unique_ptr<MergeRanksAndIndicesShaderGL> MergeRanksAndIndices;
      std::unique_ptr<MergeReferencesShaderGL> MergeReferences;
      std::unique_ptr<RemoveDuplicatesShaderGL> RemoveDuplicates;
      std::unique_ptr<RemoveGapsShaderGL> RemoveGaps;
      std::unique_ptr<PartitionShaderGL> Partition;
      std::unique_ptr<RemovePartitionGapsShaderGL> RemovePartitionGaps;
      std::unique_ptr<SmallPartitionShaderGL> SmallPartition;
      std::unique_ptr<CopyReferenceShaderGL> CopyReference;
      std::unique_ptr<PartitionFinalShaderGL> PartitionFinal;
      std::unique_ptr<VerifyShaderGL> Verify;
      std::unique_ptr<SumNodeNumShaderGL> SumNodeNum;

      KdtreeBuild() :
         Initialize( std::make_unique<InitializeShaderGL>() ),
         InitializeReference( std::make_unique<InitializeReferenceShaderGL>() ),
         CopyCoordinates( std::make_unique<CopyCoordinatesShaderGL>() ),
         SortByBlock( std::make_unique<SortByBlockShaderGL>() ),
         SortLastBlock( std::make_unique<SortLastBlockShaderGL>() ),
         GenerateSampleRanks( std::make_unique<GenerateSampleRanksShaderGL>() ),
         MergeRanksAndIndices( std::make_unique<MergeRanksAndIndicesShaderGL>() ),
         MergeReferences( std::make_unique<MergeReferencesShaderGL>() ),
         RemoveDuplicates( std::make_unique<RemoveDuplicatesShaderGL>() ),
         RemoveGaps( std::make_unique<RemoveGapsShaderGL>() ),
         Partition( std::make_unique<PartitionShaderGL>() ),
         RemovePartitionGaps( std::make_unique<RemovePartitionGapsShaderGL>() ),
         SmallPartition( std::make_unique<SmallPartitionShaderGL>() ),
         CopyReference( std::make_unique<CopyReferenceShaderGL>() ),
         PartitionFinal( std::make_unique<PartitionFinalShaderGL>() ),
         Verify( std::make_unique<VerifyShaderGL>() ),
         SumNodeNum( std::make_unique<SumNodeNumShaderGL>() )
         {}
   };

   inline static RendererGL* Renderer = nullptr;
   GLFWwindow* Window;
   bool Pause;
   int FrameWidth;
   int FrameHeight;
   glm::ivec2 ClickedPoint;
   std::unique_ptr<TextGL> Texter;
   std::unique_ptr<LightGL> Lights;
   std::unique_ptr<KdtreeGL> Object;
   std::unique_ptr<CameraGL> MainCamera;
   std::unique_ptr<CameraGL> TextCamera;
   std::unique_ptr<ShaderGL> TextShader;
   std::unique_ptr<ShaderGL> SceneShader;
   std::unique_ptr<TimeCheck> Timer;
   KdtreeBuild KdtreeBuilder;

   // 16 and 32 do well, anything in between or below is bad.
   // 32 seems to do well on laptop/desktop Windows Intel and on NVidia/AMD as well.
   // further hardware-specific tuning might be needed for optimal performance.
   static constexpr int ThreadGroupSize = 32;
   [[nodiscard]] static int getGroupSize(int size)
   {
      return (size + ThreadGroupSize - 1) / ThreadGroupSize;
   }

   static constexpr int divideUp(int a, int b) { return (a + b - 1) / b; }

   void registerCallbacks() const;
   void initialize();
   void writeFrame() const;
   static void printOpenGLInformation();
   static void cleanup(GLFWwindow* window);
   static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
   static void cursor(GLFWwindow* window, double xpos, double ypos);
   static void mouse(GLFWwindow* window, int button, int action, int mods);

   void setLights();
   void setObject() const;
   void setShaders() const;
   void sortByAxis(int axis) const;
   void removeDuplicates(int axis) const;
   void sort() const;
   void partitionDimension(int axis, int depth) const;
   void build() const;
   void verify() const;
   void buildKdtree() const;
   void drawObject() const;
   void drawText(const std::string& text, glm::vec2 start_position) const;
   void render();
};