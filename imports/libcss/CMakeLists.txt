# $Id: CMakeLists.txt 1273 2011-11-25 03:16:33Z justin $

include_directories(libcss/src libcss/include libparserutils/src libparserutils/include libwapcaplet/src
                    libwapcaplet/include)

add_library(libcss STATIC libcss/src/charset/detect.c libcss/src/lex/lex.c libcss/src/parse/font_face.c
            libcss/src/parse/important.c libcss/src/parse/language.c libcss/src/parse/parse.c
            libcss/src/parse/properties/azimuth.c libcss/src/parse/properties/background.c
            libcss/src/parse/properties/background_position.c libcss/src/parse/properties/border.c
            libcss/src/parse/properties/border_color.c libcss/src/parse/properties/border_spacing.c
            libcss/src/parse/properties/border_style.c libcss/src/parse/properties/border_width.c
            libcss/src/parse/properties/clip.c libcss/src/parse/properties/column_rule.c
            libcss/src/parse/properties/columns.c libcss/src/parse/properties/content.c
            libcss/src/parse/properties/cue.c libcss/src/parse/properties/cursor.c
            libcss/src/parse/properties/elevation.c libcss/src/parse/properties/font.c
            libcss/src/parse/properties/font_family.c libcss/src/parse/properties/font_weight.c
            libcss/src/parse/properties/list_style.c libcss/src/parse/properties/list_style_type.c
            libcss/src/parse/properties/margin.c libcss/src/parse/properties/opacity.c
            libcss/src/parse/properties/outline.c libcss/src/parse/properties/padding.c
            libcss/src/parse/properties/pause.c libcss/src/parse/properties/play_during.c
            libcss/src/parse/properties/properties.c libcss/src/parse/properties/quotes.c
            libcss/src/parse/properties/text_decoration.c libcss/src/parse/properties/utils.c
            libcss/src/parse/properties/voice_family.c libcss/src/parse/propstrings.c
            libcss/src/select/computed.c libcss/src/select/dispatch.c libcss/src/select/font_face.c
            libcss/src/select/hash.c libcss/src/select/properties/azimuth.c
            libcss/src/select/properties/background_attachment.c
            libcss/src/select/properties/background_color.c libcss/src/select/properties/background_image.c
            libcss/src/select/properties/background_position.c libcss/src/select/properties/background_repeat.c
            libcss/src/select/properties/border_bottom_color.c libcss/src/select/properties/border_bottom_style.c
            libcss/src/select/properties/border_bottom_width.c libcss/src/select/properties/border_collapse.c
            libcss/src/select/properties/border_left_color.c libcss/src/select/properties/border_left_style.c
            libcss/src/select/properties/border_left_width.c libcss/src/select/properties/border_right_color.c
            libcss/src/select/properties/border_right_style.c libcss/src/select/properties/border_right_width.c
            libcss/src/select/properties/border_spacing.c libcss/src/select/properties/border_top_color.c
            libcss/src/select/properties/border_top_style.c libcss/src/select/properties/border_top_width.c
            libcss/src/select/properties/bottom.c libcss/src/select/properties/break_after.c
            libcss/src/select/properties/break_before.c libcss/src/select/properties/break_inside.c
            libcss/src/select/properties/caption_side.c libcss/src/select/properties/clear.c
            libcss/src/select/properties/clip.c libcss/src/select/properties/color.c
            libcss/src/select/properties/column_count.c libcss/src/select/properties/column_fill.c
            libcss/src/select/properties/column_gap.c libcss/src/select/properties/column_rule_color.c
            libcss/src/select/properties/column_rule_style.c libcss/src/select/properties/column_rule_width.c
            libcss/src/select/properties/column_span.c libcss/src/select/properties/column_width.c
            libcss/src/select/properties/content.c libcss/src/select/properties/counter_increment.c
            libcss/src/select/properties/counter_reset.c libcss/src/select/properties/cue_after.c
            libcss/src/select/properties/cue_before.c libcss/src/select/properties/cursor.c
            libcss/src/select/properties/direction.c libcss/src/select/properties/display.c
            libcss/src/select/properties/elevation.c libcss/src/select/properties/empty_cells.c
            libcss/src/select/properties/float.c libcss/src/select/properties/font_family.c
            libcss/src/select/properties/font_size.c libcss/src/select/properties/font_style.c
            libcss/src/select/properties/font_variant.c libcss/src/select/properties/font_weight.c
            libcss/src/select/properties/height.c libcss/src/select/properties/helpers.c
            libcss/src/select/properties/left.c libcss/src/select/properties/letter_spacing.c
            libcss/src/select/properties/line_height.c libcss/src/select/properties/list_style_image.c
            libcss/src/select/properties/list_style_position.c libcss/src/select/properties/list_style_type.c
            libcss/src/select/properties/margin_bottom.c libcss/src/select/properties/margin_left.c
            libcss/src/select/properties/margin_right.c libcss/src/select/properties/margin_top.c
            libcss/src/select/properties/max_height.c libcss/src/select/properties/max_width.c
            libcss/src/select/properties/min_height.c libcss/src/select/properties/min_width.c
            libcss/src/select/properties/opacity.c libcss/src/select/properties/orphans.c
            libcss/src/select/properties/outline_color.c libcss/src/select/properties/outline_style.c
            libcss/src/select/properties/outline_width.c libcss/src/select/properties/overflow.c
            libcss/src/select/properties/padding_bottom.c libcss/src/select/properties/padding_left.c
            libcss/src/select/properties/padding_right.c libcss/src/select/properties/padding_top.c
            libcss/src/select/properties/page_break_after.c libcss/src/select/properties/page_break_before.c
            libcss/src/select/properties/page_break_inside.c libcss/src/select/properties/pause_after.c
            libcss/src/select/properties/pause_before.c libcss/src/select/properties/pitch.c
            libcss/src/select/properties/pitch_range.c libcss/src/select/properties/play_during.c
            libcss/src/select/properties/position.c libcss/src/select/properties/quotes.c
            libcss/src/select/properties/richness.c libcss/src/select/properties/right.c
            libcss/src/select/properties/speak.c libcss/src/select/properties/speak_header.c
            libcss/src/select/properties/speak_numeral.c libcss/src/select/properties/speak_punctuation.c
            libcss/src/select/properties/speech_rate.c libcss/src/select/properties/stress.c
            libcss/src/select/properties/table_layout.c libcss/src/select/properties/text_align.c
            libcss/src/select/properties/text_decoration.c libcss/src/select/properties/text_indent.c
            libcss/src/select/properties/text_transform.c libcss/src/select/properties/top.c
            libcss/src/select/properties/unicode_bidi.c libcss/src/select/properties/vertical_align.c
            libcss/src/select/properties/visibility.c libcss/src/select/properties/voice_family.c
            libcss/src/select/properties/volume.c libcss/src/select/properties/white_space.c
            libcss/src/select/properties/widows.c libcss/src/select/properties/width.c
            libcss/src/select/properties/word_spacing.c libcss/src/select/properties/writing_mode.c
            libcss/src/select/properties/z_index.c libcss/src/select/select.c libcss/src/stylesheet.c
            libcss/src/utils/errors.c libcss/src/utils/utils.c libparserutils/src/charset/aliases.c
            libparserutils/src/charset/codec.c libparserutils/src/charset/codecs/codec_8859.c
            libparserutils/src/charset/codecs/codec_ascii.c libparserutils/src/charset/codecs/codec_ext8.c
            libparserutils/src/charset/codecs/codec_utf16.c libparserutils/src/charset/codecs/codec_utf8.c
            libparserutils/src/charset/encodings/utf16.c libparserutils/src/charset/encodings/utf8.c
            libparserutils/src/input/filter.c libparserutils/src/input/inputstream.c
            libparserutils/src/utils/buffer.c libparserutils/src/utils/errors.c
            libparserutils/src/utils/stack.c libparserutils/src/utils/vector.c
            libwapcaplet/src/libwapcaplet.c libcss/src/parse/properties/autogenerated_background_attachment.c
            libcss/src/parse/properties/autogenerated_background_color.c
            libcss/src/parse/properties/autogenerated_background_image.c
            libcss/src/parse/properties/autogenerated_background_repeat.c
            libcss/src/parse/properties/autogenerated_border_bottom.c
            libcss/src/parse/properties/autogenerated_border_bottom_color.c
            libcss/src/parse/properties/autogenerated_border_bottom_style.c
            libcss/src/parse/properties/autogenerated_border_bottom_width.c
            libcss/src/parse/properties/autogenerated_border_collapse.c
            libcss/src/parse/properties/autogenerated_border_left.c
            libcss/src/parse/properties/autogenerated_border_left_color.c
            libcss/src/parse/properties/autogenerated_border_left_style.c
            libcss/src/parse/properties/autogenerated_border_left_width.c
            libcss/src/parse/properties/autogenerated_border_right.c
            libcss/src/parse/properties/autogenerated_border_right_color.c
            libcss/src/parse/properties/autogenerated_border_right_style.c
            libcss/src/parse/properties/autogenerated_border_right_width.c
            libcss/src/parse/properties/autogenerated_border_side_color.c
            libcss/src/parse/properties/autogenerated_border_side_style.c
            libcss/src/parse/properties/autogenerated_border_side_width.c
            libcss/src/parse/properties/autogenerated_border_top.c
            libcss/src/parse/properties/autogenerated_border_top_color.c
            libcss/src/parse/properties/autogenerated_border_top_style.c
            libcss/src/parse/properties/autogenerated_border_top_width.c
            libcss/src/parse/properties/autogenerated_bottom.c
            libcss/src/parse/properties/autogenerated_break_after.c
            libcss/src/parse/properties/autogenerated_break_before.c
            libcss/src/parse/properties/autogenerated_break_inside.c
            libcss/src/parse/properties/autogenerated_caption_side.c
            libcss/src/parse/properties/autogenerated_clear.c
            libcss/src/parse/properties/autogenerated_color.c
            libcss/src/parse/properties/autogenerated_column_count.c
            libcss/src/parse/properties/autogenerated_column_fill.c
            libcss/src/parse/properties/autogenerated_column_gap.c
            libcss/src/parse/properties/autogenerated_column_rule_color.c
            libcss/src/parse/properties/autogenerated_column_rule_style.c
            libcss/src/parse/properties/autogenerated_column_rule_width.c
            libcss/src/parse/properties/autogenerated_column_span.c
            libcss/src/parse/properties/autogenerated_column_width.c
            libcss/src/parse/properties/autogenerated_counter_increment.c
            libcss/src/parse/properties/autogenerated_counter_reset.c
            libcss/src/parse/properties/autogenerated_cue_after.c
            libcss/src/parse/properties/autogenerated_cue_before.c
            libcss/src/parse/properties/autogenerated_direction.c
            libcss/src/parse/properties/autogenerated_display.c
            libcss/src/parse/properties/autogenerated_empty_cells.c
            libcss/src/parse/properties/autogenerated_float.c
            libcss/src/parse/properties/autogenerated_font_size.c
            libcss/src/parse/properties/autogenerated_font_style.c
            libcss/src/parse/properties/autogenerated_font_variant.c
            libcss/src/parse/properties/autogenerated_height.c
            libcss/src/parse/properties/autogenerated_left.c
            libcss/src/parse/properties/autogenerated_letter_spacing.c
            libcss/src/parse/properties/autogenerated_line_height.c
            libcss/src/parse/properties/autogenerated_list_style_image.c
            libcss/src/parse/properties/autogenerated_list_style_position.c
            libcss/src/parse/properties/autogenerated_margin_bottom.c
            libcss/src/parse/properties/autogenerated_margin_left.c
            libcss/src/parse/properties/autogenerated_margin_right.c
            libcss/src/parse/properties/autogenerated_margin_side.c
            libcss/src/parse/properties/autogenerated_margin_top.c
            libcss/src/parse/properties/autogenerated_max_height.c
            libcss/src/parse/properties/autogenerated_max_width.c
            libcss/src/parse/properties/autogenerated_min_height.c
            libcss/src/parse/properties/autogenerated_min_width.c
            libcss/src/parse/properties/autogenerated_orphans.c
            libcss/src/parse/properties/autogenerated_outline_color.c
            libcss/src/parse/properties/autogenerated_outline_style.c
            libcss/src/parse/properties/autogenerated_outline_width.c
            libcss/src/parse/properties/autogenerated_overflow.c
            libcss/src/parse/properties/autogenerated_padding_bottom.c
            libcss/src/parse/properties/autogenerated_padding_left.c
            libcss/src/parse/properties/autogenerated_padding_right.c
            libcss/src/parse/properties/autogenerated_padding_side.c
            libcss/src/parse/properties/autogenerated_padding_top.c
            libcss/src/parse/properties/autogenerated_page_break_after.c
            libcss/src/parse/properties/autogenerated_page_break_before.c
            libcss/src/parse/properties/autogenerated_page_break_inside.c
            libcss/src/parse/properties/autogenerated_pause_after.c
            libcss/src/parse/properties/autogenerated_pause_before.c
            libcss/src/parse/properties/autogenerated_pitch.c
            libcss/src/parse/properties/autogenerated_pitch_range.c
            libcss/src/parse/properties/autogenerated_position.c
            libcss/src/parse/properties/autogenerated_richness.c
            libcss/src/parse/properties/autogenerated_right.c
            libcss/src/parse/properties/autogenerated_side.c
            libcss/src/parse/properties/autogenerated_speak.c
            libcss/src/parse/properties/autogenerated_speak_header.c
            libcss/src/parse/properties/autogenerated_speak_numeral.c
            libcss/src/parse/properties/autogenerated_speak_punctuation.c
            libcss/src/parse/properties/autogenerated_speech_rate.c
            libcss/src/parse/properties/autogenerated_stress.c
            libcss/src/parse/properties/autogenerated_table_layout.c
            libcss/src/parse/properties/autogenerated_text_align.c
            libcss/src/parse/properties/autogenerated_text_indent.c
            libcss/src/parse/properties/autogenerated_text_transform.c
            libcss/src/parse/properties/autogenerated_top.c
            libcss/src/parse/properties/autogenerated_unicode_bidi.c
            libcss/src/parse/properties/autogenerated_vertical_align.c
            libcss/src/parse/properties/autogenerated_visibility.c
            libcss/src/parse/properties/autogenerated_volume.c
            libcss/src/parse/properties/autogenerated_white_space.c
            libcss/src/parse/properties/autogenerated_widows.c
            libcss/src/parse/properties/autogenerated_width.c
            libcss/src/parse/properties/autogenerated_word_spacing.c
            libcss/src/parse/properties/autogenerated_writing_mode.c
            libcss/src/parse/properties/autogenerated_z_index.c)

