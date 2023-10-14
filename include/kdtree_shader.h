#pragma once

#include "shader.h"

class InitializeShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0 };

   InitializeShaderGL() = default;
   ~InitializeShaderGL() override = default;

   InitializeShaderGL(const InitializeShaderGL&) = delete;
   InitializeShaderGL(const InitializeShaderGL&&) = delete;
   InitializeShaderGL& operator=(const InitializeShaderGL&) = delete;
   InitializeShaderGL& operator=(const InitializeShaderGL&&) = delete;
};

class InitializeReferenceShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0 };

   InitializeReferenceShaderGL() = default;
   ~InitializeReferenceShaderGL() override = default;

   InitializeReferenceShaderGL(const InitializeReferenceShaderGL&) = delete;
   InitializeReferenceShaderGL(const InitializeReferenceShaderGL&&) = delete;
   InitializeReferenceShaderGL& operator=(const InitializeReferenceShaderGL&) = delete;
   InitializeReferenceShaderGL& operator=(const InitializeReferenceShaderGL&&) = delete;
};

class CopyCoordinatesShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0, Axis, Dim };

   CopyCoordinatesShaderGL() = default;
   ~CopyCoordinatesShaderGL() override = default;

   CopyCoordinatesShaderGL(const CopyCoordinatesShaderGL&) = delete;
   CopyCoordinatesShaderGL(const CopyCoordinatesShaderGL&&) = delete;
   CopyCoordinatesShaderGL& operator=(const CopyCoordinatesShaderGL&) = delete;
   CopyCoordinatesShaderGL& operator=(const CopyCoordinatesShaderGL&&) = delete;
};

class SortByBlockShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0, Axis, Dim };

   SortByBlockShaderGL() = default;
   ~SortByBlockShaderGL() override = default;

   SortByBlockShaderGL(const SortByBlockShaderGL&) = delete;
   SortByBlockShaderGL(const SortByBlockShaderGL&&) = delete;
   SortByBlockShaderGL& operator=(const SortByBlockShaderGL&) = delete;
   SortByBlockShaderGL& operator=(const SortByBlockShaderGL&&) = delete;
};

class SortLastBlockShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { StartOffset = 0, SortedSize, Size, Axis, Dim };

   SortLastBlockShaderGL() = default;
   ~SortLastBlockShaderGL() override = default;

   SortLastBlockShaderGL(const SortLastBlockShaderGL&) = delete;
   SortLastBlockShaderGL(const SortLastBlockShaderGL&&) = delete;
   SortLastBlockShaderGL& operator=(const SortLastBlockShaderGL&) = delete;
   SortLastBlockShaderGL& operator=(const SortLastBlockShaderGL&&) = delete;
};

class GenerateSampleRanksShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SortedSize = 0, Size, Axis, Dim, TotalThreadNum };

   GenerateSampleRanksShaderGL() = default;
   ~GenerateSampleRanksShaderGL() override = default;

   GenerateSampleRanksShaderGL(const GenerateSampleRanksShaderGL&) = delete;
   GenerateSampleRanksShaderGL(const GenerateSampleRanksShaderGL&&) = delete;
   GenerateSampleRanksShaderGL& operator=(const GenerateSampleRanksShaderGL&) = delete;
   GenerateSampleRanksShaderGL& operator=(const GenerateSampleRanksShaderGL&&) = delete;
};

class MergeRanksAndIndicesShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SortedSize = 0, Size, TotalThreadNum };

   MergeRanksAndIndicesShaderGL() = default;
   ~MergeRanksAndIndicesShaderGL() override = default;

   MergeRanksAndIndicesShaderGL(const MergeRanksAndIndicesShaderGL&) = delete;
   MergeRanksAndIndicesShaderGL(const MergeRanksAndIndicesShaderGL&&) = delete;
   MergeRanksAndIndicesShaderGL& operator=(const MergeRanksAndIndicesShaderGL&) = delete;
   MergeRanksAndIndicesShaderGL& operator=(const MergeRanksAndIndicesShaderGL&&) = delete;
};

class MergeReferencesShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SortedSize = 0, Size, Axis, Dim };

   MergeReferencesShaderGL() = default;
   ~MergeReferencesShaderGL() override = default;

   MergeReferencesShaderGL(const MergeReferencesShaderGL&) = delete;
   MergeReferencesShaderGL(const MergeReferencesShaderGL&&) = delete;
   MergeReferencesShaderGL& operator=(const MergeReferencesShaderGL&) = delete;
   MergeReferencesShaderGL& operator=(const MergeReferencesShaderGL&&) = delete;
};

class RemoveDuplicatesShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SizePerWarp = 0, Size, Axis, Dim };

   RemoveDuplicatesShaderGL() = default;
   ~RemoveDuplicatesShaderGL() override = default;

   RemoveDuplicatesShaderGL(const RemoveDuplicatesShaderGL&) = delete;
   RemoveDuplicatesShaderGL(const RemoveDuplicatesShaderGL&&) = delete;
   RemoveDuplicatesShaderGL& operator=(const RemoveDuplicatesShaderGL&) = delete;
   RemoveDuplicatesShaderGL& operator=(const RemoveDuplicatesShaderGL&&) = delete;
};

class RemoveGapsShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SizePerWarp = 0, Size };

   RemoveGapsShaderGL() = default;
   ~RemoveGapsShaderGL() override = default;

   RemoveGapsShaderGL(const RemoveGapsShaderGL&) = delete;
   RemoveGapsShaderGL(const RemoveGapsShaderGL&&) = delete;
   RemoveGapsShaderGL& operator=(const RemoveGapsShaderGL&) = delete;
   RemoveGapsShaderGL& operator=(const RemoveGapsShaderGL&&) = delete;
};

class PartitionShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Start = 0, End, Axis, Dim, Depth };

   PartitionShaderGL() = default;
   ~PartitionShaderGL() override = default;

   PartitionShaderGL(const PartitionShaderGL&) = delete;
   PartitionShaderGL(const PartitionShaderGL&&) = delete;
   PartitionShaderGL& operator=(const PartitionShaderGL&) = delete;
   PartitionShaderGL& operator=(const PartitionShaderGL&&) = delete;
};

class RemovePartitionGapsShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Start = 0, End, Depth };

   RemovePartitionGapsShaderGL() = default;
   ~RemovePartitionGapsShaderGL() override = default;

   RemovePartitionGapsShaderGL(const RemovePartitionGapsShaderGL&) = delete;
   RemovePartitionGapsShaderGL(const RemovePartitionGapsShaderGL&&) = delete;
   RemovePartitionGapsShaderGL& operator=(const RemovePartitionGapsShaderGL&) = delete;
   RemovePartitionGapsShaderGL& operator=(const RemovePartitionGapsShaderGL&&) = delete;
};

class SmallPartitionShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Start = 0, End, Axis, Dim, Depth, MaxControllableDepthForWarp };

   SmallPartitionShaderGL() = default;
   ~SmallPartitionShaderGL() override = default;

   SmallPartitionShaderGL(const SmallPartitionShaderGL&) = delete;
   SmallPartitionShaderGL(const SmallPartitionShaderGL&&) = delete;
   SmallPartitionShaderGL& operator=(const SmallPartitionShaderGL&) = delete;
   SmallPartitionShaderGL& operator=(const SmallPartitionShaderGL&&) = delete;
};

class CopyReferenceShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0 };

   CopyReferenceShaderGL() = default;
   ~CopyReferenceShaderGL() override = default;

   CopyReferenceShaderGL(const CopyReferenceShaderGL&) = delete;
   CopyReferenceShaderGL(const CopyReferenceShaderGL&&) = delete;
   CopyReferenceShaderGL& operator=(const CopyReferenceShaderGL&) = delete;
   CopyReferenceShaderGL& operator=(const CopyReferenceShaderGL&&) = delete;
};

class PartitionFinalShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Start = 0, End, Depth, MidReferenceOffset, LastMidReferenceOffset };

   PartitionFinalShaderGL() = default;
   ~PartitionFinalShaderGL() override = default;

   PartitionFinalShaderGL(const PartitionFinalShaderGL&) = delete;
   PartitionFinalShaderGL(const PartitionFinalShaderGL&&) = delete;
   PartitionFinalShaderGL& operator=(const PartitionFinalShaderGL&) = delete;
   PartitionFinalShaderGL& operator=(const PartitionFinalShaderGL&&) = delete;
};

class VerifyShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { Size = 0 };

   VerifyShaderGL() = default;
   ~VerifyShaderGL() override = default;

   VerifyShaderGL(const VerifyShaderGL&) = delete;
   VerifyShaderGL(const VerifyShaderGL&&) = delete;
   VerifyShaderGL& operator=(const VerifyShaderGL&) = delete;
   VerifyShaderGL& operator=(const VerifyShaderGL&&) = delete;
};

class SumNodeNumShaderGL final : public ShaderGL
{
public:
   SumNodeNumShaderGL() = default;
   ~SumNodeNumShaderGL() override = default;

   SumNodeNumShaderGL(const SumNodeNumShaderGL&) = delete;
   SumNodeNumShaderGL(const SumNodeNumShaderGL&&) = delete;
   SumNodeNumShaderGL& operator=(const SumNodeNumShaderGL&) = delete;
   SumNodeNumShaderGL& operator=(const SumNodeNumShaderGL&&) = delete;
};

class SearchShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { SearchRadius = 0, NodeIndex, QueryNum, Size, Dim };

   SearchShaderGL() = default;
   ~SearchShaderGL() override = default;

   SearchShaderGL(const SearchShaderGL&) = delete;
   SearchShaderGL(const SearchShaderGL&&) = delete;
   SearchShaderGL& operator=(const SearchShaderGL&) = delete;
   SearchShaderGL& operator=(const SearchShaderGL&&) = delete;
};

class CopyFoundPointsShaderGL final : public ShaderGL
{
public:
   CopyFoundPointsShaderGL() = default;
   ~CopyFoundPointsShaderGL() override = default;

   CopyFoundPointsShaderGL(const CopyFoundPointsShaderGL&) = delete;
   CopyFoundPointsShaderGL(const CopyFoundPointsShaderGL&&) = delete;
   CopyFoundPointsShaderGL& operator=(const CopyFoundPointsShaderGL&) = delete;
   CopyFoundPointsShaderGL& operator=(const CopyFoundPointsShaderGL&&) = delete;
};

class InitializeKNNShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { QueryNum = 0, NeighborNum };

   InitializeKNNShaderGL() = default;
   ~InitializeKNNShaderGL() override = default;

   InitializeKNNShaderGL(const InitializeKNNShaderGL&) = delete;
   InitializeKNNShaderGL(const InitializeKNNShaderGL&&) = delete;
   InitializeKNNShaderGL& operator=(const InitializeKNNShaderGL&) = delete;
   InitializeKNNShaderGL& operator=(const InitializeKNNShaderGL&&) = delete;
};

class FindNearestNeighborsShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { NodeIndex = 0, QueryNum, NeighborNum, Size, Dim };

   FindNearestNeighborsShaderGL() = default;
   ~FindNearestNeighborsShaderGL() override = default;

   FindNearestNeighborsShaderGL(const FindNearestNeighborsShaderGL&) = delete;
   FindNearestNeighborsShaderGL(const FindNearestNeighborsShaderGL&&) = delete;
   FindNearestNeighborsShaderGL& operator=(const FindNearestNeighborsShaderGL&) = delete;
   FindNearestNeighborsShaderGL& operator=(const FindNearestNeighborsShaderGL&&) = delete;
};

class CopyEncodedFoundPointsShaderGL final : public ShaderGL
{
public:
   enum UNIFORM { NeighborNum = 0 };

   CopyEncodedFoundPointsShaderGL() = default;
   ~CopyEncodedFoundPointsShaderGL() override = default;

   CopyEncodedFoundPointsShaderGL(const CopyEncodedFoundPointsShaderGL&) = delete;
   CopyEncodedFoundPointsShaderGL(const CopyEncodedFoundPointsShaderGL&&) = delete;
   CopyEncodedFoundPointsShaderGL& operator=(const CopyEncodedFoundPointsShaderGL&) = delete;
   CopyEncodedFoundPointsShaderGL& operator=(const CopyEncodedFoundPointsShaderGL&&) = delete;
};