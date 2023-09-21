#include "renderer.h"

RendererGL::RendererGL() :
   Window( nullptr ), Pause( false ), FrameWidth( 1024 ), FrameHeight( 1024 ), ShadowMapSize( 1024 ),
   BoxHalfSide( 300.0f ), DepthFBO( 0 ), DepthTextureArrayID( 0 ), ClickedPoint( -1, -1 ), ActiveCamera( nullptr ),
   Texter( std::make_unique<TextGL>() ), MainCamera( std::make_unique<CameraGL>() ),
   TextCamera( std::make_unique<CameraGL>() ), TextShader( std::make_unique<ShaderGL>() ),
   PCFSceneShader( std::make_unique<ShaderGL>() ), LightViewDepthShader( std::make_unique<ShaderGL>() ),
   Lights( std::make_unique<LightGL>() ), Object( std::make_unique<ObjectGL>() ),
   WallObject( std::make_unique<ObjectGL>() )
{
   Renderer = this;

   initialize();
   printOpenGLInformation();
}

RendererGL::~RendererGL()
{
   if (DepthTextureArrayID != 0) glDeleteTextures( 1, &DepthTextureArrayID );
   if (DepthFBO != 0) glDeleteFramebuffers( 1, &DepthFBO );
}

void RendererGL::printOpenGLInformation()
{
   std::cout << "****************************************************************\n";
   std::cout << " - GLFW version supported: " << glfwGetVersionString() << "\n";
   std::cout << " - OpenGL renderer: " << glGetString( GL_RENDERER ) << "\n";
   std::cout << " - OpenGL version supported: " << glGetString( GL_VERSION ) << "\n";
   std::cout << " - OpenGL shader version supported: " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << "\n";
   std::cout << "****************************************************************\n\n";
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

   Window = glfwCreateWindow( FrameWidth, FrameHeight, "Cinematic Relighting", nullptr, nullptr );
   glfwMakeContextCurrent( Window );

   if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cout << "Failed to initialize GLAD" << std::endl;
      return;
   }

   registerCallbacks();

   glEnable( GL_CULL_FACE );
   glEnable( GL_DEPTH_TEST );
   glClearColor( 0.094, 0.07f, 0.17f, 1.0f );

   Texter->initialize( 30.0f );

   TextCamera->update2DCamera( FrameWidth, FrameHeight );
   MainCamera->updatePerspectiveCamera( FrameWidth, FrameHeight );

   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   TextShader->setShader(
      std::string(shader_directory_path + "/text.vert").c_str(),
      std::string(shader_directory_path + "/text.frag").c_str()
   );
   PCFSceneShader->setShader(
      std::string(shader_directory_path + "/shadow/scene_shader.vert").c_str(),
      std::string(shader_directory_path + "/shadow/scene_shader.frag").c_str()
   );
   LightViewDepthShader->setShader(
      std::string(shader_directory_path + "/shadow/light_view_depth_generator.vert").c_str(),
      std::string(shader_directory_path + "/shadow/light_view_depth_generator.frag").c_str()
   );
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

void RendererGL::writeDepthTextureArray() const
{
   const int size = ShadowMapSize * ShadowMapSize;
   auto* buffer = new uint8_t[size];
   auto* raw_buffer = new GLfloat[size];
   glBindFramebuffer( GL_FRAMEBUFFER, DepthFBO );
   for (int s = 0; s < Lights->getTotalLightNum(); ++s) {
      glNamedFramebufferReadBuffer( DepthFBO, GL_DEPTH_ATTACHMENT );
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, s );
      glReadPixels( 0, 0, ShadowMapSize, ShadowMapSize, GL_DEPTH_COMPONENT, GL_FLOAT, raw_buffer );

      for (int i = 0; i < size; ++i) {
         buffer[i] = static_cast<uint8_t>(LightCameras[s]->linearizeDepthValue( raw_buffer[i] ) * 255.0f);
      }

      FIBITMAP* image = FreeImage_ConvertFromRawBits(
         buffer, ShadowMapSize, ShadowMapSize, ShadowMapSize, 8,
         FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
      );
      FreeImage_Save( FIF_PNG, image, std::string("../depth" + std::to_string( s ) + ".png").c_str() );
      FreeImage_Unload( image );
   }
   delete [] raw_buffer;
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
      case GLFW_KEY_0:
         Renderer->ActiveCamera = Renderer->MainCamera.get();
         std::cout << ">> Main Camera Selected\n";
         break;
      case GLFW_KEY_1:
         Renderer->ActiveCamera = Renderer->LightCameras[0].get();
         std::cout << ">> Light-0 Selected\n";
         break;
      case GLFW_KEY_2:
         Renderer->ActiveCamera = Renderer->LightCameras[1].get();
         std::cout << ">> Light-1 Selected\n";
         break;
      case GLFW_KEY_C:
         Renderer->writeFrame();
         std::cout << ">> Framebuffer Captured\n";
         break;
      case GLFW_KEY_D:
         Renderer->writeDepthTextureArray();
         std::cout << ">> Depth Array Captured\n";
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

   if (Renderer->ActiveCamera->getMovingState()) {
      const auto x = static_cast<int>(std::round( xpos ));
      const auto y = static_cast<int>(std::round( ypos ));
      const int dx = x - Renderer->ClickedPoint.x;
      const int dy = y - Renderer->ClickedPoint.y;
      Renderer->ActiveCamera->moveForward( -dy );
      Renderer->ActiveCamera->rotateAroundWorldY( -dx );

      if (glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS) {
         Renderer->ActiveCamera->pitch( -dy );
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
      Renderer->ActiveCamera->setMovingState( moving_state );
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

   const std::vector<glm::vec3> reference_points = {
      glm::vec3(Lights->getLightPosition( 0 )) + Lights->getSpotlightDirection( 0 ),
      glm::vec3(0.0f)
   };

   const int light_num = Lights->getTotalLightNum();
   LightViewMatrices.resize( light_num );
   LightViewProjectionMatrices.resize( light_num );
   LightCameras.resize( light_num );
   for (int i = 0; i < light_num; ++i) {
      LightCameras[i] = std::make_unique<CameraGL>();
      LightCameras[i]->updatePerspectiveCamera( ShadowMapSize, ShadowMapSize );
      LightCameras[i]->updateNearFarPlanes( 100.0f, 1000.0f );
      LightCameras[i]->updateCameraView(
         glm::vec3(Lights->getLightPosition( i )),
         reference_points[i],
         glm::vec3(0.0f, 1.0f, 0.0f)
      );
   }
   ActiveCamera = LightCameras[0].get();
}

void RendererGL::setObject() const
{
   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   Object->setObject(
      GL_TRIANGLES,
      std::string(sample_directory_path + "/Tiger/tiger.obj")
   );
   Object->setDiffuseReflectionColor( { 1.0f, 1.0f, 1.0f, 1.0f } );
}

void RendererGL::setWallObject()
{
   std::vector<glm::vec3> wall_vertices;
   wall_vertices.emplace_back( BoxHalfSide, 0.0f, BoxHalfSide );
   wall_vertices.emplace_back( BoxHalfSide, 0.0f, -BoxHalfSide );
   wall_vertices.emplace_back( -BoxHalfSide, 0.0f, -BoxHalfSide );

   wall_vertices.emplace_back( -BoxHalfSide, 0.0f, BoxHalfSide );
   wall_vertices.emplace_back( BoxHalfSide, 0.0f, BoxHalfSide );
   wall_vertices.emplace_back( -BoxHalfSide, 0.0f, -BoxHalfSide );

   std::vector<glm::vec3> wall_normals;
   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );
   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );
   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );

   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );
   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );
   wall_normals.emplace_back( 0.0f, 1.0f, 0.0f );

   std::vector<glm::vec2> wall_textures;
   wall_textures.emplace_back( 1.0f, 1.0f );
   wall_textures.emplace_back( 1.0f, 0.0f );
   wall_textures.emplace_back( 0.0f, 0.0f );

   wall_textures.emplace_back( 0.0f, 1.0f );
   wall_textures.emplace_back( 1.0f, 1.0f );
   wall_textures.emplace_back( 0.0f, 0.0f );

   WallObject->setObject( GL_TRIANGLES, wall_vertices, wall_normals, wall_textures );
   WallObject->setDiffuseReflectionColor( { 1.0f, 1.0f, 1.0f, 1.0f } );

   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   WallTextureIndices[0] = WallObject->addTexture( std::string(sample_directory_path + "/Wall/sand.jpg") );
   WallTextureIndices[1] = WallObject->addTexture( std::string(sample_directory_path + "/Wall/brick.jpg") );
}

void RendererGL::setLightViewFrameBuffers()
{
   glCreateTextures( GL_TEXTURE_2D_ARRAY, 1, &DepthTextureArrayID );
   glTextureStorage3D( DepthTextureArrayID, 1, GL_DEPTH_COMPONENT32F, ShadowMapSize, ShadowMapSize, Lights->getTotalLightNum() );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL );

   glCreateFramebuffers( 1, &DepthFBO );
   for (int i = 0; i < Lights->getTotalLightNum(); ++i) {
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, i );
   }

   if (glCheckNamedFramebufferStatus( DepthFBO, GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "DepthFBO Setup Error\n";
   }
}

void RendererGL::drawObject(ShaderGL* shader, CameraGL* camera) const
{
   glBindVertexArray( Object->getVAO() );
   glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, Object->getIBO() );
   Object->transferUniformsToShader( shader );

   const glm::mat4 to_object =
      glm::translate( glm::mat4(1.0f), glm::vec3(0.0f, 50.0f, 0.0f) ) *
      glm::scale( glm::mat4(1.0f), glm::vec3(0.3f) );
   glm::mat4 to_world = glm::translate( glm::mat4(1.0f), glm::vec3(200.0f, 0.0f, 0.0f) ) * to_object;
   shader->transferBasicTransformationUniforms( to_world, camera );
   glDrawElements( Object->getDrawMode(), Object->getIndexNum(), GL_UNSIGNED_INT, nullptr );

   to_world = glm::translate( glm::mat4(1.0f), glm::vec3(-150.0f, 0.0f, 0.0f) ) * to_object;
   shader->transferBasicTransformationUniforms( to_world, camera );
   glDrawElements( Object->getDrawMode(), Object->getIndexNum(), GL_UNSIGNED_INT, nullptr );

   to_world = glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 0.0f, -100.0f) ) * to_object;
   shader->transferBasicTransformationUniforms( to_world, camera );
   glDrawElements( Object->getDrawMode(), Object->getIndexNum(), GL_UNSIGNED_INT, nullptr );

   to_world =
      glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 30.0f, 200.0f) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( -30.0f ), glm::vec3(1.0f, 0.0f, 0.0f) ) * to_object;
   shader->transferBasicTransformationUniforms( to_world, camera );
   glDrawElements( Object->getDrawMode(), Object->getIndexNum(), GL_UNSIGNED_INT, nullptr );
}

void RendererGL::drawBoxObject(ShaderGL* shader, const CameraGL* camera) const
{
   glBindVertexArray( WallObject->getVAO() );

   shader->transferBasicTransformationUniforms( glm::mat4(1.0f), camera );
   WallObject->transferUniformsToShader( shader );
   glBindTextureUnit( 0, WallObject->getTextureID( WallTextureIndices[0] ) );
   glDrawArrays( WallObject->getDrawMode(), 0, WallObject->getVertexNum() );

   glm::mat4 to_world =
      glm::translate( glm::mat4(1.0f), glm::vec3(0.0f, BoxHalfSide, -BoxHalfSide) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( 90.0f ), glm::vec3(1.0f, 0.0f, 0.0f) );
   shader->transferBasicTransformationUniforms( to_world, camera );
   WallObject->transferUniformsToShader( shader );
   glBindTextureUnit( 0, WallObject->getTextureID( WallTextureIndices[1] ) );
   glDrawArrays( WallObject->getDrawMode(), 0, WallObject->getVertexNum() );

   to_world =
      glm::translate( glm::mat4(1.0f), glm::vec3(-BoxHalfSide, BoxHalfSide, 0.0f) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( 90.0f ), glm::vec3(0.0f, 1.0f, 0.0f) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( 90.0f ), glm::vec3(1.0f, 0.0f, 0.0f) );
   shader->transferBasicTransformationUniforms( to_world, camera );
   WallObject->transferUniformsToShader( shader );
   glDrawArrays( WallObject->getDrawMode(), 0, WallObject->getVertexNum() );

   to_world =
      glm::translate( glm::mat4(1.0f), glm::vec3(BoxHalfSide, BoxHalfSide, 0.0f) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( -90.0f ), glm::vec3(0.0f, 1.0f, 0.0f) ) *
      glm::rotate( glm::mat4(1.0f), glm::radians( 90.0f ), glm::vec3(1.0f, 0.0f, 0.0f) );
   shader->transferBasicTransformationUniforms( to_world, camera );
   WallObject->transferUniformsToShader( shader );
   glDrawArrays( WallObject->getDrawMode(), 0, WallObject->getVertexNum() );
}

void RendererGL::drawDepthMapFromLightView()
{
   glViewport( 0, 0, ShadowMapSize, ShadowMapSize );
   glBindFramebuffer( GL_FRAMEBUFFER, DepthFBO );
   glUseProgram( LightViewDepthShader->getShaderProgram() );
   glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

   for (int i = 0; i < Lights->getTotalLightNum(); ++i) {
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, i );

      constexpr GLfloat one = 1.0f;
      glClearNamedFramebufferfv( DepthFBO, GL_DEPTH, 0, &one );

      drawObject( LightViewDepthShader.get(), LightCameras[i].get() );
      drawBoxObject( LightViewDepthShader.get(), LightCameras[i].get() );

      LightViewMatrices[i] = LightCameras[i]->getViewMatrix();
      LightViewProjectionMatrices[i] = LightCameras[i]->getProjectionMatrix() * LightCameras[i]->getViewMatrix();
   }
   glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
}

void RendererGL::drawShadow() const
{
   glViewport( 0, 0, FrameWidth, FrameHeight );
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glUseProgram( PCFSceneShader->getShaderProgram() );

   Lights->transferUniformsToShader( PCFSceneShader.get() );

   PCFSceneShader->uniform4fv( "ShadowColor", { 0.24, 0.16f, 0.13f, 1.0f } );
   PCFSceneShader->uniformMat4fv( "LightViewMatrix", Lights->getTotalLightNum(), LightViewMatrices.data() );
   PCFSceneShader->uniformMat4fv( "LightViewProjectionMatrix", Lights->getTotalLightNum(), LightViewProjectionMatrices.data() );

   glBindTextureUnit( 1, DepthTextureArrayID );

   PCFSceneShader->uniform1i( "UseTexture", 0 );
   drawObject( PCFSceneShader.get(), MainCamera.get() );

   PCFSceneShader->uniform1i( "UseTexture", 1 );
   drawBoxObject( PCFSceneShader.get(), MainCamera.get() );
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

   std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

   drawDepthMapFromLightView();
   drawShadow();

   std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
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
   setWallObject();
   setLightViewFrameBuffers();
   TextShader->setTextUniformLocations();
   LightViewDepthShader->setLightViewUniformLocations();
   PCFSceneShader->setSceneUniformLocations( Lights->getTotalLightNum() );

   while (!glfwWindowShouldClose( Window )) {
      if (!Pause) render();

      glfwSwapBuffers( Window );
      glfwPollEvents();
   }
   glfwDestroyWindow( Window );
}