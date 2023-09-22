#include "kdtree_shader.h"

void InitializeShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}

void InitializeReferenceShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}

void CopyCoordinatesShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
}

void SortByBlockShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
}