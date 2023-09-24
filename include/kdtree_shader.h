#pragma once

#include "shader.h"

class InitializeShaderGL final : public ShaderGL
{
public:
   InitializeShaderGL() = default;
   ~InitializeShaderGL() override = default;

   InitializeShaderGL(const InitializeShaderGL&) = delete;
   InitializeShaderGL(const InitializeShaderGL&&) = delete;
   InitializeShaderGL& operator=(const InitializeShaderGL&) = delete;
   InitializeShaderGL& operator=(const InitializeShaderGL&&) = delete;

   void setUniformLocations() override;
};

class InitializeReferenceShaderGL final : public ShaderGL
{
public:
   InitializeReferenceShaderGL() = default;
   ~InitializeReferenceShaderGL() override = default;

   InitializeReferenceShaderGL(const InitializeReferenceShaderGL&) = delete;
   InitializeReferenceShaderGL(const InitializeReferenceShaderGL&&) = delete;
   InitializeReferenceShaderGL& operator=(const InitializeReferenceShaderGL&) = delete;
   InitializeReferenceShaderGL& operator=(const InitializeReferenceShaderGL&&) = delete;

   void setUniformLocations() override;
};

class CopyCoordinatesShaderGL final : public ShaderGL
{
public:
   CopyCoordinatesShaderGL() = default;
   ~CopyCoordinatesShaderGL() override = default;

   CopyCoordinatesShaderGL(const CopyCoordinatesShaderGL&) = delete;
   CopyCoordinatesShaderGL(const CopyCoordinatesShaderGL&&) = delete;
   CopyCoordinatesShaderGL& operator=(const CopyCoordinatesShaderGL&) = delete;
   CopyCoordinatesShaderGL& operator=(const CopyCoordinatesShaderGL&&) = delete;

   void setUniformLocations() override;
};

class SortByBlockShaderGL final : public ShaderGL
{
public:
   SortByBlockShaderGL() = default;
   ~SortByBlockShaderGL() override = default;

   SortByBlockShaderGL(const SortByBlockShaderGL&) = delete;
   SortByBlockShaderGL(const SortByBlockShaderGL&&) = delete;
   SortByBlockShaderGL& operator=(const SortByBlockShaderGL&) = delete;
   SortByBlockShaderGL& operator=(const SortByBlockShaderGL&&) = delete;

   void setUniformLocations() override;
};

class SortLastBlockShaderGL final : public ShaderGL
{
public:
   SortLastBlockShaderGL() = default;
   ~SortLastBlockShaderGL() override = default;

   SortLastBlockShaderGL(const SortLastBlockShaderGL&) = delete;
   SortLastBlockShaderGL(const SortLastBlockShaderGL&&) = delete;
   SortLastBlockShaderGL& operator=(const SortLastBlockShaderGL&) = delete;
   SortLastBlockShaderGL& operator=(const SortLastBlockShaderGL&&) = delete;

   void setUniformLocations() override;
};

class GenerateSampleRanksShaderGL final : public ShaderGL
{
public:
   GenerateSampleRanksShaderGL() = default;
   ~GenerateSampleRanksShaderGL() override = default;

   GenerateSampleRanksShaderGL(const GenerateSampleRanksShaderGL&) = delete;
   GenerateSampleRanksShaderGL(const GenerateSampleRanksShaderGL&&) = delete;
   GenerateSampleRanksShaderGL& operator=(const GenerateSampleRanksShaderGL&) = delete;
   GenerateSampleRanksShaderGL& operator=(const GenerateSampleRanksShaderGL&&) = delete;

   void setUniformLocations() override;
};

class MergeRanksAndIndicesShaderGL final : public ShaderGL
{
public:
   MergeRanksAndIndicesShaderGL() = default;
   ~MergeRanksAndIndicesShaderGL() override = default;

   MergeRanksAndIndicesShaderGL(const MergeRanksAndIndicesShaderGL&) = delete;
   MergeRanksAndIndicesShaderGL(const MergeRanksAndIndicesShaderGL&&) = delete;
   MergeRanksAndIndicesShaderGL& operator=(const MergeRanksAndIndicesShaderGL&) = delete;
   MergeRanksAndIndicesShaderGL& operator=(const MergeRanksAndIndicesShaderGL&&) = delete;

   void setUniformLocations() override;
};

class MergeReferencesShaderGL final : public ShaderGL
{
public:
   MergeReferencesShaderGL() = default;
   ~MergeReferencesShaderGL() override = default;

   MergeReferencesShaderGL(const MergeReferencesShaderGL&) = delete;
   MergeReferencesShaderGL(const MergeReferencesShaderGL&&) = delete;
   MergeReferencesShaderGL& operator=(const MergeReferencesShaderGL&) = delete;
   MergeReferencesShaderGL& operator=(const MergeReferencesShaderGL&&) = delete;

   void setUniformLocations() override;
};

class RemoveDuplicatesShaderGL final : public ShaderGL
{
public:
   RemoveDuplicatesShaderGL() = default;
   ~RemoveDuplicatesShaderGL() override = default;

   RemoveDuplicatesShaderGL(const RemoveDuplicatesShaderGL&) = delete;
   RemoveDuplicatesShaderGL(const RemoveDuplicatesShaderGL&&) = delete;
   RemoveDuplicatesShaderGL& operator=(const RemoveDuplicatesShaderGL&) = delete;
   RemoveDuplicatesShaderGL& operator=(const RemoveDuplicatesShaderGL&&) = delete;

   void setUniformLocations() override;
};

class RemoveGapsShaderGL final : public ShaderGL
{
public:
   RemoveGapsShaderGL() = default;
   ~RemoveGapsShaderGL() override = default;

   RemoveGapsShaderGL(const RemoveGapsShaderGL&) = delete;
   RemoveGapsShaderGL(const RemoveGapsShaderGL&&) = delete;
   RemoveGapsShaderGL& operator=(const RemoveGapsShaderGL&) = delete;
   RemoveGapsShaderGL& operator=(const RemoveGapsShaderGL&&) = delete;

   void setUniformLocations() override;
};

class PartitionShaderGL final : public ShaderGL
{
public:
   PartitionShaderGL() = default;
   ~PartitionShaderGL() override = default;

   PartitionShaderGL(const PartitionShaderGL&) = delete;
   PartitionShaderGL(const PartitionShaderGL&&) = delete;
   PartitionShaderGL& operator=(const PartitionShaderGL&) = delete;
   PartitionShaderGL& operator=(const PartitionShaderGL&&) = delete;

   void setUniformLocations() override;
};

class RemovePartitionGapsShaderGL final : public ShaderGL
{
public:
   RemovePartitionGapsShaderGL() = default;
   ~RemovePartitionGapsShaderGL() override = default;

   RemovePartitionGapsShaderGL(const RemovePartitionGapsShaderGL&) = delete;
   RemovePartitionGapsShaderGL(const RemovePartitionGapsShaderGL&&) = delete;
   RemovePartitionGapsShaderGL& operator=(const RemovePartitionGapsShaderGL&) = delete;
   RemovePartitionGapsShaderGL& operator=(const RemovePartitionGapsShaderGL&&) = delete;

   void setUniformLocations() override;
};

class SmallPartitionShaderGL final : public ShaderGL
{
public:
   SmallPartitionShaderGL() = default;
   ~SmallPartitionShaderGL() override = default;

   SmallPartitionShaderGL(const SmallPartitionShaderGL&) = delete;
   SmallPartitionShaderGL(const SmallPartitionShaderGL&&) = delete;
   SmallPartitionShaderGL& operator=(const SmallPartitionShaderGL&) = delete;
   SmallPartitionShaderGL& operator=(const SmallPartitionShaderGL&&) = delete;

   void setUniformLocations() override;
};

class CopyReferenceShaderGL final : public ShaderGL
{
public:
   CopyReferenceShaderGL() = default;
   ~CopyReferenceShaderGL() override = default;

   CopyReferenceShaderGL(const CopyReferenceShaderGL&) = delete;
   CopyReferenceShaderGL(const CopyReferenceShaderGL&&) = delete;
   CopyReferenceShaderGL& operator=(const CopyReferenceShaderGL&) = delete;
   CopyReferenceShaderGL& operator=(const CopyReferenceShaderGL&&) = delete;

   void setUniformLocations() override;
};

class PartitionFinalShaderGL final : public ShaderGL
{
public:
   PartitionFinalShaderGL() = default;
   ~PartitionFinalShaderGL() override = default;

   PartitionFinalShaderGL(const PartitionFinalShaderGL&) = delete;
   PartitionFinalShaderGL(const PartitionFinalShaderGL&&) = delete;
   PartitionFinalShaderGL& operator=(const PartitionFinalShaderGL&) = delete;
   PartitionFinalShaderGL& operator=(const PartitionFinalShaderGL&&) = delete;

   void setUniformLocations() override;
};

class VerifyShaderGL final : public ShaderGL
{
public:
   VerifyShaderGL() = default;
   ~VerifyShaderGL() override = default;

   VerifyShaderGL(const VerifyShaderGL&) = delete;
   VerifyShaderGL(const VerifyShaderGL&&) = delete;
   VerifyShaderGL& operator=(const VerifyShaderGL&) = delete;
   VerifyShaderGL& operator=(const VerifyShaderGL&&) = delete;

   void setUniformLocations() override;
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

   void setUniformLocations() override {}
};

class SearchShaderGL final : public ShaderGL
{
public:
   SearchShaderGL() = default;
   ~SearchShaderGL() override = default;

   SearchShaderGL(const SearchShaderGL&) = delete;
   SearchShaderGL(const SearchShaderGL&&) = delete;
   SearchShaderGL& operator=(const SearchShaderGL&) = delete;
   SearchShaderGL& operator=(const SearchShaderGL&&) = delete;

   void setUniformLocations() override;
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

   void setUniformLocations() override {}
};

class InitializeKNNShaderGL final : public ShaderGL
{
public:
   InitializeKNNShaderGL() = default;
   ~InitializeKNNShaderGL() override = default;

   InitializeKNNShaderGL(const InitializeKNNShaderGL&) = delete;
   InitializeKNNShaderGL(const InitializeKNNShaderGL&&) = delete;
   InitializeKNNShaderGL& operator=(const InitializeKNNShaderGL&) = delete;
   InitializeKNNShaderGL& operator=(const InitializeKNNShaderGL&&) = delete;

   void setUniformLocations() override;
};

class FindNearestNeighborsShaderGL final : public ShaderGL
{
public:
   FindNearestNeighborsShaderGL() = default;
   ~FindNearestNeighborsShaderGL() override = default;

   FindNearestNeighborsShaderGL(const FindNearestNeighborsShaderGL&) = delete;
   FindNearestNeighborsShaderGL(const FindNearestNeighborsShaderGL&&) = delete;
   FindNearestNeighborsShaderGL& operator=(const FindNearestNeighborsShaderGL&) = delete;
   FindNearestNeighborsShaderGL& operator=(const FindNearestNeighborsShaderGL&&) = delete;

   void setUniformLocations() override;
};

class CopyEncodedFoundPointsShaderGL final : public ShaderGL
{
public:
   CopyEncodedFoundPointsShaderGL() = default;
   ~CopyEncodedFoundPointsShaderGL() override = default;

   CopyEncodedFoundPointsShaderGL(const CopyEncodedFoundPointsShaderGL&) = delete;
   CopyEncodedFoundPointsShaderGL(const CopyEncodedFoundPointsShaderGL&&) = delete;
   CopyEncodedFoundPointsShaderGL& operator=(const CopyEncodedFoundPointsShaderGL&) = delete;
   CopyEncodedFoundPointsShaderGL& operator=(const CopyEncodedFoundPointsShaderGL&&) = delete;

   void setUniformLocations() override;
};