#include "camera.h"

CameraGL::CameraGL() :
   CameraGL(
      glm::vec3(0.0f, 0.0f, 700.0f),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 1.0f, 0.0f)
   )
{
}

CameraGL::CameraGL(
   const glm::vec3& cam_position,
   const glm::vec3& view_reference_position,
   const glm::vec3& view_up_vector,
   float fov,
   float near_plane,
   float far_plane
) : 
   IsPerspective( false ), IsMoving( false ), Width( 0 ), Height( 0 ), FOV( fov ), InitFOV( fov ),
   NearPlane( near_plane ), FarPlane( far_plane ), AspectRatio( 0.0f ), ZoomSensitivity( 1.0f ),
   MoveSensitivity( 0.5f ), RotationSensitivity( 0.005f ), InitCamPos( cam_position ),
   InitRefPos( view_reference_position ), InitUpVec( view_up_vector ), CamPos( cam_position ),
   ViewMatrix( glm::lookAt( InitCamPos, InitRefPos, InitUpVec ) ), ProjectionMatrix( glm::mat4(1.0f) )
{
}

void CameraGL::updateCamera()
{
   const glm::mat4 inverse_view = glm::inverse( ViewMatrix );
   CamPos.x = inverse_view[3][0];
   CamPos.y = inverse_view[3][1];
   CamPos.z = inverse_view[3][2];
}

void CameraGL::pitch(int angle)
{
   const glm::vec3 u_axis(ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
   ViewMatrix = glm::rotate( ViewMatrix, static_cast<float>(-angle) * RotationSensitivity, u_axis );
   updateCamera();
}

void CameraGL::yaw(int angle)
{
   const glm::vec3 v_axis(ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);
   ViewMatrix = glm::rotate( ViewMatrix, static_cast<float>(-angle) * RotationSensitivity, v_axis );
   updateCamera();
}

void CameraGL::rotateAroundWorldY(int angle)
{
   const glm::vec3 world_y(0.0f, 1.0f, 0.0f);
   ViewMatrix = glm::rotate(glm::mat4(1.0f), static_cast<float>(-angle) * RotationSensitivity, world_y ) * ViewMatrix;
   updateCamera();
}

void CameraGL::moveForward(int delta)
{
   const glm::vec3 n_axis(ViewMatrix[0][2], ViewMatrix[1][2], ViewMatrix[2][2]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) *  -n_axis );
   updateCamera();
}

void CameraGL::moveBackward(int delta)
{
   const glm::vec3 n_axis(ViewMatrix[0][2], ViewMatrix[1][2], ViewMatrix[2][2]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) * n_axis );
   updateCamera();
}

void CameraGL::moveLeft(int delta)
{
   const glm::vec3 u_axis(ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) * -u_axis );
   updateCamera();
}

void CameraGL::moveRight(int delta)
{
   const glm::vec3 u_axis(ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) * u_axis );
   updateCamera();
}

void CameraGL::moveUp(int delta)
{
   const glm::vec3 v_axis(ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) * v_axis );
   updateCamera();
}

void CameraGL::moveDown(int delta)
{
   const glm::vec3 v_axis(ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);
   ViewMatrix = glm::translate( ViewMatrix, MoveSensitivity * static_cast<float>(-delta) * -v_axis );
   updateCamera();
}

void CameraGL::zoomIn()
{
   if (FOV > 0.0f) {
      FOV -= ZoomSensitivity;
      ProjectionMatrix = glm::perspective( glm::radians( FOV ), AspectRatio, NearPlane, FarPlane );
   }
}

void CameraGL::zoomOut()
{
   if (FOV < 90.0f) {
      FOV += ZoomSensitivity;
      ProjectionMatrix = glm::perspective( glm::radians( FOV ), AspectRatio, NearPlane, FarPlane );
   }
}

void CameraGL::resetCamera()
{
   CamPos = InitCamPos; 
   ViewMatrix = glm::lookAt( InitCamPos, InitRefPos, InitUpVec );
   ProjectionMatrix = glm::perspective( glm::radians( InitFOV ), AspectRatio, NearPlane, FarPlane );
}

void CameraGL::update2DCamera(int width, int height)
{
   Width = width;
   Height = height;
   IsPerspective = false;
   ViewMatrix = glm::mat4(1.0f);
   ProjectionMatrix = glm::ortho(
      0.0f, static_cast<float>(width),
      0.0f, static_cast<float>(height),
      -1.0f, 1.0f
   );
}

void CameraGL::updateOrthographicCamera(int width, int height)
{
   Width = width;
   Height = height;
   IsPerspective = false;
   AspectRatio = static_cast<float>(width) / static_cast<float>(height);
   ProjectionMatrix = glm::ortho(
      -static_cast<float>(width) * 0.5f, static_cast<float>(width) * 0.5f,
      -static_cast<float>(height) * 0.5f, static_cast<float>(height) * 0.5f,
      NearPlane, FarPlane
   );
}

void CameraGL::updatePerspectiveCamera(int width, int height)
{
   Width = width;
   Height = height;
   IsPerspective = true;
   AspectRatio = static_cast<float>(width) / static_cast<float>(height);
   ProjectionMatrix = glm::perspective( glm::radians( FOV ), AspectRatio, NearPlane, FarPlane );
}

void CameraGL::updateCameraView(
   const glm::vec3& cam_position,
   const glm::vec3& view_reference_position,
   const glm::vec3& view_up_vector
)
{
   InitCamPos = cam_position;
   InitRefPos = view_reference_position;
   InitUpVec = view_up_vector;
   ViewMatrix = glm::lookAt( InitCamPos, InitRefPos, InitUpVec );
   updateCamera();
}

void CameraGL::updateNearFarPlanes(float near, float far)
{
   NearPlane = near;
   FarPlane = far;
   if (IsPerspective) ProjectionMatrix = glm::perspective( glm::radians( FOV ), AspectRatio, near, far );
   else {
      ProjectionMatrix = glm::ortho(
         -static_cast<float>(Width) * 0.5f, static_cast<float>(Width) * 0.5f,
         -static_cast<float>(Height) * 0.5f, static_cast<float>(Height) * 0.5f,
         near, far
      );
   }
}

float CameraGL::linearizeDepthValue(float depth) const
{
   if (!IsPerspective) return depth;

   const float z_ndc = 2.0f * depth - 1.0f;
   const float z = (2.0f * NearPlane * FarPlane) / (FarPlane + NearPlane - z_ndc * (FarPlane - NearPlane));
   return glm::clamp( (z - NearPlane) / (FarPlane - NearPlane), 0.0f, 1.0f );
}