#include "renderer.h"

RendererGL::RendererGL() :
   Window( nullptr ), Pause( false ), FrameWidth( 1024 ), FrameHeight( 1024 ), ClickedPoint( -1, -1 ),
   Texter( std::make_unique<TextGL>() ), Lights( std::make_unique<LightGL>() ), Object( std::make_unique<KdtreeGL>() ),
   MainCamera( std::make_unique<CameraGL>() ), TextCamera( std::make_unique<CameraGL>() ),
   TextShader( std::make_unique<ShaderGL>() ), SceneShader( std::make_unique<ShaderGL>() ),
   Timer( std::make_unique<TimeCheck>() ), KdtreeBuilder()

{
   Renderer = this;

   initialize();
   printOpenGLInformation();
}

void RendererGL::printOpenGLInformation()
{
   std::cout << "**************************************************************************\n";
   std::cout << " - GLFW version supported: " << glfwGetVersionString() << "\n";
   std::cout << " - OpenGL renderer: " << glGetString( GL_RENDERER ) << "\n";
   std::cout << " - OpenGL version supported: " << glGetString( GL_VERSION ) << "\n";
   std::cout << " - OpenGL shader version supported: " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << "\n";

   int work_group_count = 0;
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_group_count );
   std::cout << " - OpenGL maximum number of work groups: " <<  work_group_count << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_group_count );
   std::cout << work_group_count << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_group_count );
   std::cout << work_group_count << "\n";

   int work_group_size = 0;
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_group_size );
   std::cout << " - OpenGL maximum work group size: " <<  work_group_size << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_group_size );
   std::cout << work_group_size << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_group_size );
   std::cout << work_group_size << "\n";

   int max_ssbo_num = 0;
   glGetIntegerv( GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &max_ssbo_num );
   std::cout << " - OpenGL maximum number of shader storage blocks: " << max_ssbo_num << "\n";
   std::cout << "**************************************************************************\n\n";
}

void RendererGL::initialize()
{
   if (!glfwInit()) {
      std::cout << "Cannot Initialize OpenGL...\n";
      return;
   }
   glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
   glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
   glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE );
   glfwWindowHint( GLFW_RESIZABLE, GLFW_FALSE );
   glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

   Window = glfwCreateWindow( FrameWidth, FrameHeight, "K-d Tree", nullptr, nullptr );
   glfwMakeContextCurrent( Window );

   if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cout << "Failed to initialize GLAD" << std::endl;
      return;
   }

   registerCallbacks();

   glEnable( GL_CULL_FACE );
   glEnable( GL_DEPTH_TEST );
   glClearColor( 0.97, 0.47f, 0.7f, 1.0f );

   Texter->initialize( 30.0f );

   TextCamera->update2DCamera( FrameWidth, FrameHeight );
   MainCamera->updatePerspectiveCamera( FrameWidth, FrameHeight );
}

void RendererGL::writeFrame() const
{
   const int size = FrameWidth * FrameHeight * 3;
   auto* buffer = new uint8_t[size];
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glNamedFramebufferReadBuffer( 0, GL_COLOR_ATTACHMENT0 );
   glReadPixels( 0, 0, FrameWidth, FrameHeight, GL_BGR, GL_UNSIGNED_BYTE, buffer );
   FIBITMAP* image = FreeImage_ConvertFromRawBits(
      buffer, FrameWidth, FrameHeight, FrameWidth * 3, 24,
      FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
   );
   FreeImage_Save( FIF_PNG, image, "../result.png" );
   FreeImage_Unload( image );
   delete [] buffer;
}

void RendererGL::cleanup(GLFWwindow* window)
{
   glfwSetWindowShouldClose( window, GLFW_TRUE );
}

void RendererGL::keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
   if (action != GLFW_PRESS) return;

   switch (key) {
      case GLFW_KEY_C:
         Renderer->writeFrame();
         std::cout << ">> Framebuffer Captured\n";
         break;
      case GLFW_KEY_L:
         Renderer->Lights->toggleLightSwitch();
         std::cout << ">> Light Turned " << (Renderer->Lights->isLightOn() ? "On!\n" : "Off!\n");
         break;
      case GLFW_KEY_P: {
         const glm::vec3 pos = Renderer->MainCamera->getCameraPosition();
         std::cout << ">> Camera Position: " << pos.x << ", " << pos.y << ", " << pos.z << "\n";
      } break;
      case GLFW_KEY_SPACE:
         Renderer->Pause = !Renderer->Pause;
         break;
      case GLFW_KEY_Q:
      case GLFW_KEY_ESCAPE:
         cleanup( window );
         break;
      default:
         return;
   }
}

void RendererGL::cursor(GLFWwindow* window, double xpos, double ypos)
{
   if (Renderer->Pause) return;

   if (Renderer->MainCamera->getMovingState()) {
      const auto x = static_cast<int>(std::round( xpos ));
      const auto y = static_cast<int>(std::round( ypos ));
      const int dx = x - Renderer->ClickedPoint.x;
      const int dy = y - Renderer->ClickedPoint.y;
      Renderer->MainCamera->moveForward( -dy );
      Renderer->MainCamera->rotateAroundWorldY( -dx );

      if (glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS) {
         Renderer->MainCamera->pitch( -dy );
      }

      Renderer->ClickedPoint.x = x;
      Renderer->ClickedPoint.y = y;
   }
}

void RendererGL::mouse(GLFWwindow* window, int button, int action, int mods)
{
   if (Renderer->Pause) return;

   if (button == GLFW_MOUSE_BUTTON_LEFT) {
      const bool moving_state = action == GLFW_PRESS;
      if (moving_state) {
         double x, y;
         glfwGetCursorPos( window, &x, &y );
         Renderer->ClickedPoint.x = static_cast<int>(std::round( x ));
         Renderer->ClickedPoint.y = static_cast<int>(std::round( y ));
      }
      Renderer->MainCamera->setMovingState( moving_state );
   }
}

void RendererGL::registerCallbacks() const
{
   glfwSetWindowCloseCallback( Window, cleanup );
   glfwSetKeyCallback( Window, keyboard );
   glfwSetCursorPosCallback( Window, cursor );
   glfwSetMouseButtonCallback( Window, mouse );
}

void RendererGL::setLights()
{
   glm::vec4 light_position(0.0f, 500.0f, 500.0f, 1.0f);
   glm::vec4 ambient_color(0.1f, 0.1f, 0.1f, 1.0f);
   glm::vec4 diffuse_color(0.7f, 0.7f, 0.7f, 1.0f);
   glm::vec4 specular_color(0.7f, 0.7f, 0.7f, 1.0f);
   const glm::vec3 reference_position(0.0f, 150.0f, 0.0f);
   Lights->addLight(
      light_position, ambient_color, diffuse_color, specular_color,
      reference_position - glm::vec3(light_position),
      25.0f,
      0.5f,
      1000.0f
   );

   light_position = glm::vec4(0.0f, 0.0f, 500.0f, 1.0f);
   ambient_color = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
   diffuse_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
   specular_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
   Lights->addLight( light_position, ambient_color, diffuse_color, specular_color );
}

void RendererGL::setObject() const
{
   std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   Object->setObject(
      GL_TRIANGLES,
      std::string(sample_directory_path + "/Bunny/bunny.obj")
   );
   Object->setDiffuseReflectionColor( { 0.3f, 0.3f, 0.5f, 1.0f } );
   std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
   Timer->ObjectLoad =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-9;
}

void RendererGL::setShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   TextShader->setShader(
      std::string(shader_directory_path + "/text.vert").c_str(),
      std::string(shader_directory_path + "/text.frag").c_str()
   );
   SceneShader->setShader(
      std::string(shader_directory_path + "/scene_shader.vert").c_str(),
      std::string(shader_directory_path + "/scene_shader.frag").c_str()
   );
   TextShader->setTextUniformLocations();
   SceneShader->setSceneUniformLocations( Lights->getTotalLightNum() );

   KdtreeBuilder.Initialize->setComputeShader( std::string(shader_directory_path + "/kdtree/initialize.comp").c_str() );
   KdtreeBuilder.Initialize->setUniformLocations();

   KdtreeBuilder.InitializeReference->setComputeShader(
      std::string(shader_directory_path + "/kdtree/initialize_reference.comp").c_str()
   );
   KdtreeBuilder.InitializeReference->setUniformLocations();

   KdtreeBuilder.CopyCoordinates->setComputeShader(
      std::string(shader_directory_path + "/kdtree/copy_coordinates.comp").c_str()
   );
   KdtreeBuilder.CopyCoordinates->setUniformLocations();

   KdtreeBuilder.SortByBlock->setComputeShader(
      std::string(shader_directory_path + "/kdtree/sort_by_block.comp").c_str()
   );
   KdtreeBuilder.SortByBlock->setUniformLocations();

   KdtreeBuilder.SortLastBlock->setComputeShader(
      std::string(shader_directory_path + "/kdtree/sort_last_block.comp").c_str()
   );
   KdtreeBuilder.SortLastBlock->setUniformLocations();

   KdtreeBuilder.GenerateSampleRanks->setComputeShader(
      std::string(shader_directory_path + "/kdtree/generate_sample_ranks.comp").c_str()
   );
   KdtreeBuilder.GenerateSampleRanks->setUniformLocations();

   KdtreeBuilder.MergeRanksAndIndices->setComputeShader(
      std::string(shader_directory_path + "/kdtree/merge_ranks_and_indices.comp").c_str()
   );
   KdtreeBuilder.MergeRanksAndIndices->setUniformLocations();

   KdtreeBuilder.MergeReferences->setComputeShader(
      std::string(shader_directory_path + "/kdtree/merge_references.comp").c_str()
   );
   KdtreeBuilder.MergeReferences->setUniformLocations();

   KdtreeBuilder.RemoveDuplicates->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_duplicates.comp").c_str()
   );
   KdtreeBuilder.RemoveDuplicates->setUniformLocations();

   KdtreeBuilder.RemoveGaps->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_gaps.comp").c_str()
   );
   KdtreeBuilder.RemoveGaps->setUniformLocations();

   KdtreeBuilder.Partition->setComputeShader( std::string(shader_directory_path + "/kdtree/partition.comp").c_str() );
   KdtreeBuilder.Partition->setUniformLocations();

   KdtreeBuilder.RemovePartitionGaps->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_partition_gaps.comp").c_str()
   );
   KdtreeBuilder.RemovePartitionGaps->setUniformLocations();
}

void RendererGL::sortByAxis(int axis) const
{
   const int size = Object->getSize();
   const int dim = Object->getDimension();
   const GLuint coordinates = Object->getCoordinates();
   glUseProgram( KdtreeBuilder.CopyCoordinates->getShaderProgram() );
   KdtreeBuilder.CopyCoordinates->uniform1i( "Size", size );
   KdtreeBuilder.CopyCoordinates->uniform1i( "Axis", axis );
   KdtreeBuilder.CopyCoordinates->uniform1i( "Dim", dim );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getBuffer( axis ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getReference( axis ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, coordinates );
   glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   int stage_num = 0;
   GLuint in_reference, in_buffer;
   GLuint out_reference, out_buffer;
   for (int step = KdtreeGL::SharedSize; step < size; step <<= 1) stage_num++;
   if (stage_num & 1) {
      in_buffer = Object->getSortBuffer();
      in_reference = Object->getSortReference();
      out_buffer = Object->getBuffer( dim );
      out_reference = Object->getReference( dim );
   }
   else {
      in_buffer = Object->getBuffer( dim );
      in_reference = Object->getReference( dim );
      out_buffer = Object->getSortBuffer();
      out_reference = Object->getSortReference();
   }

   assert( size <= KdtreeGL::SampleStride * Object->getMaxSampleNum() );

   int block_num = size / KdtreeGL::SharedSize;
   if (block_num > 0) {
      glUseProgram( KdtreeBuilder.SortByBlock->getShaderProgram() );
      KdtreeBuilder.SortByBlock->uniform1i( "Size", size );
      KdtreeBuilder.SortByBlock->uniform1i( "Axis", axis );
      KdtreeBuilder.SortByBlock->uniform1i( "Dim", dim );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getReference( axis ) );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getBuffer( axis ) );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
   }
   const int remained_size = size % KdtreeGL::SharedSize;
   if (remained_size > 0) {
      int buffer_index = 0;
      const int start_offset = size - remained_size;
      const std::array<GLuint, 2> buffers{ Object->getBuffer( axis ), in_buffer };
      const std::array<GLuint, 2> references{ Object->getReference( axis ), in_reference };
      glUseProgram( KdtreeBuilder.SortLastBlock->getShaderProgram() );
      KdtreeBuilder.SortLastBlock->uniform1i( "StartOffset", start_offset );
      KdtreeBuilder.SortLastBlock->uniform1i( "Size", remained_size );
      KdtreeBuilder.SortLastBlock->uniform1i( "Axis", axis );
      KdtreeBuilder.SortLastBlock->uniform1i( "Dim", dim );
      for (int sorted_size = 1; sorted_size < remained_size; sorted_size <<= 1) {
         KdtreeBuilder.SortLastBlock->uniform1i( "SortedSize", sorted_size );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, references[buffer_index ^ 1] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, buffers[buffer_index ^ 1] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, references[buffer_index] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, buffers[buffer_index] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
         glDispatchCompute( divideUp( remained_size, KdtreeGL::ThreadNum ), 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
         buffer_index ^= 1;
      }
      if (buffer_index == 0) {
         glCopyNamedBufferSubData(
            buffers[0], buffers[1],
            static_cast<int>(sizeof( float ) * start_offset),
            static_cast<int>(sizeof( float ) * start_offset),
            static_cast<int>(sizeof( float ) * remained_size)
         );
         glCopyNamedBufferSubData(
            references[0], references[1],
            static_cast<int>(sizeof( int ) * start_offset),
            static_cast<int>(sizeof( int ) * start_offset),
            static_cast<int>(sizeof( int ) * remained_size)
         );
      }
   }

   for (int sorted_size = KdtreeGL::SharedSize; sorted_size < size; sorted_size <<= 1) {
      constexpr int thread_num = KdtreeGL::SampleStride * 2;
      const int remained_threads = size % (sorted_size * 2);
      const int total_thread_num = remained_threads > sorted_size ?
         (size - remained_threads + sorted_size * 2) / thread_num : (size - remained_threads) / thread_num;
      block_num = divideUp( total_thread_num, thread_num );
      glUseProgram( KdtreeBuilder.GenerateSampleRanks->getShaderProgram() );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( "SortedSize", sorted_size );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( "Size", size );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( "Axis", axis );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( "Dim", dim );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( "TotalThreadNum", total_thread_num );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getLeftRanks() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getRightRanks() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      glUseProgram( KdtreeBuilder.MergeRanksAndIndices->getShaderProgram() );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( "SortedSize", sorted_size );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( "Size", size );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( "TotalThreadNum", total_thread_num );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getLeftLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getLeftRanks() );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getRightLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getRightRanks() );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      const int merge_pairs = remained_threads > sorted_size ?
         divideUp( size, KdtreeGL::SampleStride ) : (size - remained_threads) / KdtreeGL::SampleStride;
      glUseProgram( KdtreeBuilder.MergeReferences->getShaderProgram() );
      KdtreeBuilder.MergeReferences->uniform1i( "SortedSize", sorted_size );
      KdtreeBuilder.MergeReferences->uniform1i( "Size", size );
      KdtreeBuilder.MergeReferences->uniform1i( "Axis", axis );
      KdtreeBuilder.MergeReferences->uniform1i( "Dim", dim );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, out_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, out_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, Object->getLeftLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, Object->getRightLimits() );
      glDispatchCompute( merge_pairs, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      if (remained_threads <= sorted_size) {
         glCopyNamedBufferSubData(
            in_reference, out_reference,
            static_cast<int>(sizeof( int ) * (size - remained_threads)),
            static_cast<int>(sizeof( int ) * (size - remained_threads)),
            static_cast<int>(sizeof( int ) * remained_threads)
         );
         glCopyNamedBufferSubData(
            in_buffer, out_buffer,
            static_cast<int>(sizeof( float ) * (size - remained_threads)),
            static_cast<int>(sizeof( float ) * (size - remained_threads)),
            static_cast<int>(sizeof( float ) * remained_threads)
         );
      }

      std::swap( in_reference, out_reference );
      std::swap( in_buffer, out_buffer );
   }
}

void RendererGL::removeDuplicates(int axis) const
{
   constexpr int total_thread_num = KdtreeGL::ThreadBlockNum * KdtreeGL::ThreadNum;

   assert( total_thread_num > KdtreeGL::SharedSize / 2  );

   const int size = Object->getSize();
   const int dim = Object->getDimension();
   const int source_index = dim;
   const int target_index = axis;
   constexpr int block_num = total_thread_num * 2 / KdtreeGL::SharedSize;
   constexpr int segment = total_thread_num / KdtreeGL::WarpSize;
   const int size_per_warp = divideUp( size, segment );
   const GLuint coordinates = Object->getCoordinates();
   const GLuint num_after_removal = Object->addCustomBufferObject<int>( "NumAfterRemoval", 1 );
   const GLuint unique_num_in_warp = Object->addCustomBufferObject<int>( "UniqueNumInWarp", segment );
   glUseProgram( KdtreeBuilder.RemoveDuplicates->getShaderProgram() );
   KdtreeBuilder.RemoveDuplicates->uniform1i( "SizePerWarp", size_per_warp );
   KdtreeBuilder.RemoveDuplicates->uniform1i( "Size", size );
   KdtreeBuilder.RemoveDuplicates->uniform1i( "Axis", axis );
   KdtreeBuilder.RemoveDuplicates->uniform1i( "Dim", dim );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, unique_num_in_warp );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getSortReference() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getSortBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getReference( source_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getBuffer( source_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, coordinates );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   glUseProgram( KdtreeBuilder.RemoveGaps->getShaderProgram() );
   KdtreeBuilder.RemoveGaps->uniform1i( "SizePerWarp", size_per_warp );
   KdtreeBuilder.RemoveGaps->uniform1i( "Size", size );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getReference( target_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getBuffer( target_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getSortReference() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getSortBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, unique_num_in_warp );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, num_after_removal );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   int num = 0;
   glGetNamedBufferSubData( num_after_removal, 0, sizeof( int ), &num );
   Object->setUniqueNum( num );

   Object->releaseCustomBuffer( "NumAfterRemoval" );
   Object->releaseCustomBuffer( "UniqueNumInWarp" );
}

void RendererGL::sort() const
{
   Object->prepareSorting();
   KdtreeBuilder.InitializeReference->uniform1i( "Size", Object->getSize() );
   for (int axis = 0; axis < Object->getDimension(); ++axis) {
      glUseProgram( KdtreeBuilder.InitializeReference->getShaderProgram() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getReference( axis ) );
      glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      sortByAxis( axis );
      removeDuplicates( axis );
   }
   Object->releaseSorting();
}

void RendererGL::partitionDimension(int axis, int depth) const
{
   constexpr int total_thread_num = KdtreeGL::ThreadBlockNum * KdtreeGL::ThreadNum;

   assert( total_thread_num > KdtreeGL::SharedSize / 2  );

   constexpr int block_num = total_thread_num * 2 / KdtreeGL::SharedSize;
   constexpr int warp_num = total_thread_num / KdtreeGL::WarpSize;
   const auto max_controllable_depth_for_warp =
         static_cast<int>(std::floor( std::log2( static_cast<double>(warp_num) ) ));
   const int dim = Object->getDimension();
   const int size = Object->getUniqueNum();
   const GLuint coordinates = Object->getCoordinates();
   const GLuint mid_reference = Object->getMidReferences( depth & 1 );
   const GLuint last_mid_reference = depth == 0 ? 0 : Object->getMidReferences( (depth - 1) & 1 );
   if (depth < max_controllable_depth_for_warp) {
      for (int i = 1; i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         glUseProgram( KdtreeBuilder.Partition->getShaderProgram() );
         KdtreeBuilder.Partition->uniform1i( "Start", 0 );
         KdtreeBuilder.Partition->uniform1i( "End", size - 1 );
         KdtreeBuilder.Partition->uniform1i( "Axis", axis );
         KdtreeBuilder.Partition->uniform1i( "Dim", dim );
         KdtreeBuilder.Partition->uniform1i( "Depth", depth );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getRoot() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getLeftChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getRightChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getReference( dim + 1 ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, last_mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 7, Object->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 8, Object->getReference( axis ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 9, coordinates );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

         glUseProgram( KdtreeBuilder.RemovePartitionGaps->getShaderProgram() );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( "Start", 0 );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( "End", size - 1 );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( "Depth", depth );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getReference( dim + 1 ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getLeftChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getRightChildNumInWarp() );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
      }
   }
   else {
      for (int i = 1; i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         glUseProgram( KdtreeBuilder.SmallPartition->getShaderProgram() );
         KdtreeBuilder.SmallPartition->uniform1i( "Start", 0 );
         KdtreeBuilder.SmallPartition->uniform1i( "End", size - 1 );
         KdtreeBuilder.SmallPartition->uniform1i( "Axis", axis );
         KdtreeBuilder.SmallPartition->uniform1i( "Dim", dim );
         KdtreeBuilder.SmallPartition->uniform1i( "Depth", depth );
         KdtreeBuilder.SmallPartition->uniform1i( "MaxControllableDepthForWarp", max_controllable_depth_for_warp );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getRoot() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, last_mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, Object->getReference( axis ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, coordinates );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      }
   }
}

void RendererGL::build() const
{
   return;
   Object->prepareBuilding();
   const int dim = Object->getDimension();
   const auto depth = static_cast<int>(std::floor( std::log2( static_cast<double>(Object->getUniqueNum()) ) ));
   for (int i = 0; i < depth - 1; ++i) {
      partitionDimension( i % dim, i );
   }

   Object->releaseBuilding();
}

void RendererGL::buildKdtree() const
{
   //const auto& vert = Object->getVertices();
   //cuda::KdtreeCUDA kdtree(glm::value_ptr( vert[0] ), static_cast<int>(vert.size()), 3);

   Object->initialize();
   glUseProgram( KdtreeBuilder.Initialize->getShaderProgram() );
   KdtreeBuilder.Initialize->uniform1i( "Size", Object->getSize() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getRoot() );
   glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
   sort();
   std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
   Timer->Sort =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-9;

   start = std::chrono::steady_clock::now();
   build();
   end = std::chrono::steady_clock::now();
   Timer->Build =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-9;
}

void RendererGL::drawObject() const
{
   glViewport( 0, 0, FrameWidth, FrameHeight );
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glUseProgram( SceneShader->getShaderProgram() );
   Lights->transferUniformsToShader( SceneShader.get() );
   Object->transferUniformsToShader( SceneShader.get() );
   SceneShader->transferBasicTransformationUniforms( glm::mat4(1.0f), MainCamera.get() );
   SceneShader->uniform1i( "UseTexture", 0 );
   glBindVertexArray( Object->getVAO() );
   glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, Object->getIBO() );
   glDrawElements( Object->getDrawMode(), Object->getIndexNum(), GL_UNSIGNED_INT, nullptr );
}

void RendererGL::drawText(const std::string& text, glm::vec2 start_position) const
{
   std::vector<TextGL::Glyph*> glyphs;
   Texter->getGlyphsFromText( glyphs, text );

   glViewport( 0, 0, FrameWidth, FrameHeight );
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glUseProgram( TextShader->getShaderProgram() );

   glEnable( GL_BLEND );
   glBlendFunc( GL_SRC_ALPHA, GL_ONE );
   glDisable( GL_DEPTH_TEST );

   glm::vec2 text_position = start_position;
   const ObjectGL* glyph_object = Texter->getGlyphObject();
   glBindVertexArray( glyph_object->getVAO() );
   for (const auto& glyph : glyphs) {
      if (glyph->IsNewLine) {
         text_position.x = start_position.x;
         text_position.y -= Texter->getFontSize();
         continue;
      }

      const glm::vec2 position(
         std::round( text_position.x + glyph->Bearing.x ),
         std::round( text_position.y + glyph->Bearing.y - glyph->Size.y )
      );
      const glm::mat4 to_world =
         glm::translate( glm::mat4(1.0f), glm::vec3(position, 0.0f) ) *
         glm::scale( glm::mat4(1.0f), glm::vec3(glyph->Size.x, glyph->Size.y, 1.0f) );
      TextShader->transferBasicTransformationUniforms( to_world, TextCamera.get() );
      TextShader->uniform2fv( "TextScale", glyph->TopRightTextureCoord );
      glBindTextureUnit( 0, glyph_object->getTextureID( glyph->TextureIDIndex ) );
      glDrawArrays( glyph_object->getDrawMode(), 0, glyph_object->getVertexNum() );

      text_position.x += glyph->Advance.x;
      text_position.y -= glyph->Advance.y;
   }
   glEnable( GL_DEPTH_TEST );
   glDisable( GL_BLEND );
}

void RendererGL::render()
{
   glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

   std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
   drawObject();
   std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
   const auto fps = 1E+6 / static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

   std::stringstream text;
   text << std::fixed << std::setprecision( 2 ) << fps << " fps";
   drawText( text.str(), { 80.0f, 100.0f } );
}

void RendererGL::play()
{
   if (glfwWindowShouldClose( Window )) initialize();

   setLights();
   setObject();
   setShaders();
   buildKdtree();

   while (!glfwWindowShouldClose( Window )) {
      if (!Pause) render();

      glfwSwapBuffers( Window );
      glfwPollEvents();
   }
   glfwDestroyWindow( Window );
}