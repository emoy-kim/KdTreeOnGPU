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

void SortLastBlockShaderGL::setUniformLocations()
{
   addUniformLocation( "StartOffset" );
   addUniformLocation( "SortedSize" );
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
}

void GenerateSampleRanksShaderGL::setUniformLocations()
{
   addUniformLocation( "SortedSize" );
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
   addUniformLocation( "TotalThreadNum" );
}

void MergeRanksAndIndicesShaderGL::setUniformLocations()
{
   addUniformLocation( "SortedSize" );
   addUniformLocation( "Size" );
   addUniformLocation( "TotalThreadNum" );
}

void MergeReferencesShaderGL::setUniformLocations()
{
   addUniformLocation( "SortedSize" );
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
}

void RemoveDuplicatesShaderGL::setUniformLocations()
{
   addUniformLocation( "SizePerWarp" );
   addUniformLocation( "Size" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
}

void RemoveGapsShaderGL::setUniformLocations()
{
   addUniformLocation( "SizePerWarp" );
   addUniformLocation( "Size" );
}

void PartitionShaderGL::setUniformLocations()
{
   addUniformLocation( "Start" );
   addUniformLocation( "End" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
   addUniformLocation( "Depth" );
}

void RemovePartitionGapsShaderGL::setUniformLocations()
{
   addUniformLocation( "Start" );
   addUniformLocation( "End" );
   addUniformLocation( "Depth" );
}

void SmallPartitionShaderGL::setUniformLocations()
{
   addUniformLocation( "Start" );
   addUniformLocation( "End" );
   addUniformLocation( "Axis" );
   addUniformLocation( "Dim" );
   addUniformLocation( "Depth" );
   addUniformLocation( "MaxControllableDepthForWarp" );
}

void CopyReferenceShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}

void PartitionFinalShaderGL::setUniformLocations()
{
   addUniformLocation( "Start" );
   addUniformLocation( "End" );
   addUniformLocation( "Depth" );
   addUniformLocation( "MidReferenceOffset" );
   addUniformLocation( "LastMidReferenceOffset" );
}

void VerifyShaderGL::setUniformLocations()
{
   addUniformLocation( "Size" );
}

void SearchShaderGL::setUniformLocations()
{
   addUniformLocation( "SearchRadius" );
   addUniformLocation( "NodeIndex" );
   addUniformLocation( "QueryNum" );
   addUniformLocation( "Size" );
   addUniformLocation( "Dim" );
}

void FindNearestNeighborsShaderGL::setUniformLocations()
{
   addUniformLocation( "NodeIndex" );
   addUniformLocation( "QueryNum" );
   addUniformLocation( "NeighborNum" );
   addUniformLocation( "Size" );
   addUniformLocation( "Dim" );
}

void CopyEncodedFoundPointsShaderGL::setUniformLocations()
{
   addUniformLocation( "NeighborNum" );
}