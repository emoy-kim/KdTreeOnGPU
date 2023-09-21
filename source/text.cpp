#include "text.h"

TextGL::TextGL() :
   FontSize( 50.0f ), FontFace( nullptr ), FontLibrary( nullptr ),
   FontFilePath( std::filesystem::path(CMAKE_SOURCE_DIR) / "3rd_party/freetype2/RobotoMono.ttf" ),
   GlyphObject( std::make_unique<ObjectGL>() )
{
}

TextGL::~TextGL()
{
   if (FontFace != nullptr) FT_Done_Face( FontFace );
   if (FontLibrary != nullptr) FT_Done_FreeType( FontLibrary );
}

void TextGL::initialize(float font_size)
{
   if (FT_Init_FreeType( &FontLibrary )) std::cerr << "Could not initialize FreeType2 library\n";

   FontSize = font_size;
   FT_New_Face( FontLibrary, FontFilePath.c_str(), 0, &FontFace );
   const int width = convertFloatTo26Dot6( font_size );
   const int height = convertFloatTo26Dot6( font_size );
   FT_Set_Char_Size( FontFace, width, height, 72, 72 );

   GlyphObject->setSquareObject( GL_TRIANGLES, true );
}

void TextGL::getGlyphsFromText(std::vector<Glyph*>& glyphs, const std::string& text)
{
   for (const auto& c : text) {
      const bool is_new_line = c == '\n';
      const FT_UInt glyph_index = FT_Get_Char_Index( FontFace, c );
      const auto glyph_it = GlyphFinder.find( glyph_index );
      if (FT_Load_Glyph( FontFace, glyph_index, FT_LOAD_NO_BITMAP )) {
         std::cerr << "Could not load glyph index " << glyph_index << "\n";
      }

      if (glyph_it == GlyphFinder.end()) {
         FT_Glyph glyph_fill = nullptr;
         FT_Get_Glyph( FontFace->glyph, &glyph_fill );

         std::vector<HorizontalPixels> spans;
         renderSpans( spans, &FontFace->glyph->outline );

         int min_x = std::numeric_limits<int>::max();
         int min_y = std::numeric_limits<int>::max();
         int max_x = std::numeric_limits<int>::lowest();
         int max_y = std::numeric_limits<int>::lowest();
         for (const auto& span : spans) {
            min_x = std::min( span.Origin.x, min_x );
            max_x = std::max( span.Origin.x + span.Width, max_x );
            min_y = std::min( span.Origin.y, min_y );
            max_y = std::max( span.Origin.y + 1, max_y );
         }

         const int width = max_x - min_x;
         const int height = max_y - min_y;
         const auto glyph_width = static_cast<int>(std::pow( 2.0, std::ceil( std::log2( width ) ) ));
         const auto glyph_height = static_cast<int>(std::pow( 2.0, std::ceil( std::log2( height ) ) ));
         const size_t glyph_data_size = glyph_width * glyph_height * sizeof( uint8_t );
         auto* glyph_data = (uint8_t*)alloca( glyph_data_size );
         std::memset( glyph_data, 0, glyph_data_size );
         for (const auto& span : spans) {
            uint8_t* ptr = glyph_data + (span.Origin.y - min_y) * glyph_width + span.Origin.x - min_x;
            auto alpha = static_cast<uint8_t>(span.Coverage);
            for (int i = 0; i < span.Width; ++i) *ptr++ = alpha;
         }

         auto glyph = std::make_unique<Glyph>(
            is_new_line,
            glm::vec2{ width, height },
            glm::vec2{
               static_cast<float>(width) / static_cast<float>(glyph_width),
               static_cast<float>(height) / static_cast<float>(glyph_height)
            },
            glm::vec2{
               convert26Dot6ToFloat( static_cast<int>(FontFace->glyph->advance.x) ),
               convert26Dot6ToFloat( static_cast<int>(FontFace->glyph->advance.y) )
            },
            glm::vec2{ FontFace->glyph->bitmap_left, FontFace->glyph->bitmap_top },
            GlyphObject->addTexture( glyph_data, glyph_width, glyph_height, true )
         );
         FT_Done_Glyph( glyph_fill );
         glyphs.emplace_back( glyph.get() );
         GlyphFinder[glyph_index] = std::move( glyph );
      }
      else glyphs.emplace_back( glyph_it->second.get() );
   }
}