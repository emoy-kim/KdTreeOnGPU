#pragma once

#include "object.h"

class TextGL final
{
public:
   struct Glyph
   {
      bool IsNewLine;
      glm::vec2 Size;
      glm::vec2 TopRightTextureCoord;
      glm::vec2 Advance;
      glm::vec2 Bearing;
      int TextureIDIndex;

      Glyph() : IsNewLine( false ), Size(), TopRightTextureCoord(), Advance(), Bearing(), TextureIDIndex( -1 ) {}
      Glyph(
         bool is_new_line,
         const glm::vec2& size,
         const glm::vec2& top_right_texture_coord,
         const glm::vec2& advance,
         const glm::vec2& bearing,
         int texture_id
      ) :
         IsNewLine( is_new_line ), Size( size ), TopRightTextureCoord( top_right_texture_coord ), Advance( advance ),
         Bearing( bearing ), TextureIDIndex( texture_id ) {}
   };

   TextGL();
   ~TextGL();

   [[nodiscard]] static int convertFloatTo26Dot6(float value)
   {
      constexpr auto converter = static_cast<float>(1u << 6u);
      return static_cast<int>(std::round( value * converter ));
   }
   [[nodiscard]] static float convert26Dot6ToFloat(int value)
   {
      constexpr auto converter = 1.0f / static_cast<float>(1u << 6u);
      return static_cast<float>(value) * converter;
   }
   [[nodiscard]] float getFontSize() const { return FontSize; }
   [[nodiscard]] const ObjectGL* getGlyphObject() const { return GlyphObject.get(); }
   void initialize(float font_size);
   void getGlyphsFromText(std::vector<Glyph*>& glyphs, const std::string& text);

private:
   struct HorizontalPixels
   {
      int Width;
      int Coverage;
      glm::ivec2 Origin;

      HorizontalPixels() : Width( 0 ), Coverage( 0 ), Origin() {}
      HorizontalPixels(int x, int y, int width, int coverage) :
         Width( width ), Coverage( coverage ), Origin{ x, y } {}
   };

   float FontSize;
   FT_Face FontFace;
   FT_Library FontLibrary;
   std::filesystem::path FontFilePath;
   std::unique_ptr<ObjectGL> GlyphObject;
   std::map<FT_UInt, std::unique_ptr<Glyph>> GlyphFinder;

   static void spanCallback(int y, int count, const FT_Span* spans, void* user)
   {
      auto* span = static_cast<std::vector<HorizontalPixels>*>(user);
      for (int i = 0; i < count; ++i) {
         span->emplace_back( spans[i].x, y, spans[i].len, spans[i].coverage );
      }
   }

   void renderSpans(std::vector<HorizontalPixels>& spans, FT_Outline* outline) const
   {
      FT_Raster_Params parameters;
      std::memset( &parameters, 0, sizeof( parameters ) );
      parameters.flags = FT_RASTER_FLAG_AA | FT_RASTER_FLAG_DIRECT;
      parameters.gray_spans = spanCallback;
      parameters.user = &spans;
      FT_Outline_Render( FontLibrary, outline, &parameters );
   }
};