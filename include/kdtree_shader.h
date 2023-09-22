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