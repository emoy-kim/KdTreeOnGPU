#include "kdtree_shader.h"

void InitializeShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}

void InitializeReferenceShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}