/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

DOMString BackgroundAttachment::cssText() const {
  switch (v) {
    case Fixed:  return "fixed";
    case Scroll: return "scroll";
  }; return "";
}

DOMString BackgroundRepeat::cssText() const {
  switch (v) {
    case RepeatX: return "repeat-x";    case Repeat:   return "repeat";
    case RepeatY: return "repeat-y";    case NoRepeat: return "no-repeat";
  }; return "";
}

DOMString BorderCollapse::cssText() const {
  switch (v) {
    case Separate: return "separate";
    case Collapse: return "collapse";
  }; return "";
}

DOMString BorderWidth::cssText() const {
  if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
  switch (attr) {
    case Thin:   return "thin";  
    case Medium: return "medium";
    case Thick:  return "thick"; 
  }; return "";
}

float BorderWidth::getFloatValue(unsigned short U, Flow *inline_context) {
  switch (attr) {
    case Thin:   return ConvertFromPixels(2, U, percentRefersTo, inline_context);
    case Medium: return ConvertFromPixels(4, U, percentRefersTo, inline_context);
    case Thick:  return ConvertFromPixels(6, U, percentRefersTo, inline_context);
  }; return CSSNumericValue::getFloatValue(U, inline_context); 
}

DOMString BorderStyle::cssText() const {
  switch (v) {
    case None:   return "none";      case Double: return "double";
    case Hidden: return "hidden";    case Groove: return "groove";
    case Dotted: return "dotted";    case Ridge:  return "ridge";
    case Dashed: return "dashed";    case Inset:  return "inset";
    case Solid:  return "solid";     case Outset: return "outset";
  }; return "";
}

DOMString CaptionSide::cssText() const {
  switch (v) {
    case Top:    return "top";
    case Bottom: return "bottom";
  }; return "";
};

DOMString Clear::cssText() const {
  switch (v) {
    case None: return "none";    case Right: return "right";
    case Left: return "left";    case Both:  return "both";
  }; return "";
};

DOMString Cursor::cssText() const {
  switch (v) {
    case Auto:      return "auto";         case NResize:   return  "n-resize";
    case Move:      return "move";         case NEResize:  return "ne-resize";
    case Text:      return "text";         case EResize:   return  "e-resize";
    case Wait:      return "wait";         case SEResize:  return "se-resize";
    case Help:      return "help";         case SResize:   return  "s-resize";
    case Default:   return "default";      case SWResize:  return "sw-resize";
    case Pointer:   return "pointer";      case WResize:   return  "w-resize";
    case Crosshair: return "crosshair";    case NWResize:  return "nw-resize";
    case Progress:  return "progress"; 
  }; return "";
};

DOMString Direction::cssText() const {
  switch (v) {
    case LTR: return "ltr";
    case RTL: return "rtl";
  }; return "";
};

DOMString Display::cssText() const {
  switch (v) {
    case None:        return "none";            case RowGroup:    return "table-row-group";
    case Inline:      return "inline";          case ColGroup:    return "table-column-group";
    case Block:       return "block";           case HeaderGroup: return "table-header-group";
    case ListItem:    return "list-item";       case FooterGroup: return "table-footer-group";
    case RunIn:       return "run-in";          case Row:         return "table-row";
    case InlineBlock: return "inline-block";    case Column:      return "table-column";
    case Table:       return "table";           case Cell:        return "table-cell";
    case InlineTable: return "inline-table";    case Caption:     return "table-caption";
  }; return "";
};

DOMString EmptyCells::cssText() const {
  switch (v) {
    case Show: return "show";
    case Hide: return "hide";
  }; return "";
}

DOMString Float::cssText() const {
  switch (v) {
    case Left:  return "left";
    case Right: return "right";
    case None:  return "none";
  }; return "";
}

DOMString FontFamily::cssText() const {
  switch (attr) {
    case Serif:     return "serif";        case SansSerif: return "sans-serif";
    case Cursive:   return "cursive";      case Fantasy:   return "fantasy";
    case Monospace: return "monospace"; 
  }; return "";
}

DOMString FontSize::cssText() const {
  if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
  switch (attr) {
    case XXSmall: return "xx-small";    case XLarge:  return "x-large";
    case XSmall:  return "x-small";     case XXLarge: return "xx-large";
    case Small:   return "small";       case Larger:  return "larger";
    case Medium:  return "medium";      case Smaller: return "smaller";
    case Large:   return "large";
  }; return "";
}

int FontSize::getFontSizeValue(Flow *flow) {
  int base = FLAGS_font_size;
  float scale = 1.2;
  switch (attr) {
    case XXSmall: return base/scale/scale/scale;    case XLarge:  return base*scale*scale;
    case XSmall:  return base/scale/scale;          case XXLarge: return base*scale*scale*scale;
    case Small:   return base/scale;                case Larger:  return flow->cur_attr.font->size*scale;
    case Medium:  return base;                      case Smaller: return flow->cur_attr.font->size/scale;
    case Large:   return base*scale;
  }
  return getPixelValue(flow);
}

DOMString FontStyle::cssText() const {
  switch (v) {
    case Normal:  return "normal";
    case Italic:  return "italic";
    case Oblique: return "oblique";
  }; return "";
}

DOMString FontVariant::cssText() const {
  switch (v) {
    case Normal:    return "normal";
    case SmallCaps: return "small-caps";
  }; return "";
}

DOMString FontWeight::cssText() const {
  switch (v) {
    case Normal:  return "normal";     case _400: return "400";
    case Bold:    return "bold";       case _500: return "500";
    case Bolder:  return "bolder";     case _600: return "600";
    case Lighter: return "lighter";    case _700: return "700";
    case _100:    return "100";        case _800: return "800";
    case _200:    return "200";        case _900: return "900";
    case _300:    return "300";     
  }; return "";
}

DOMString ListStylePosition::cssText() const {
  switch (v) {
    case Inside:  return "inside";
    case Outside: return "outside";
  }; return "";
}

DOMString ListStyleType::cssText() const {
  switch (v) {
    case None:       return "none";           case LowerLatin:         return "lower-latin";
    case Disc:       return "disc";           case UpperLatin:         return "upper-latin";
    case Circle:     return "circle";         case Armenian:           return "armenian";
    case Square:     return "square";         case Georgian:           return "gerogian";
    case Decimal:    return "decimal";        case LowerAlpha:         return "lower-alpha";
    case LowerRoman: return "lower-roman";    case UpperAlpha:         return "upper-alpha";
    case UpperRoman: return "upper-roman";    case DecimalLeadingZero: return "decimal-leading-zero";
    case LowerGreek: return "lower-greek";
  }; return "";
}

DOMString Overflow::cssText() const {
  switch (v) {
    case Visible: return "visible";    case Scroll: return "scroll";
    case Hidden:  return "hidden";     case Auto:   return "auto";  
  }; return "";
}

DOMString PageBreak::cssText() const {
  switch (v) {
    case Auto:   return "auto";      case Page:        return "page";
    case Avoid:  return "avoid";     case Column:      return "column";
    case Always: return "always";    case AvoidPage:   return "avoid-page";
    case Left:   return "left";      case AvoidColumn: return "avoid-column";
    case Right:  return "right"; 
  }; return "";
}

DOMString Position::cssText() const {
  switch (v) {
    case Static:   return "static";      case Absolute: return "absolute";
    case Relative: return "relative";    case Fixed:    return "fixed";
  }; return "";
}

DOMString TableLayout::cssText() const {
  switch (v) {
    case Auto:  return "auto";
    case Fixed: return "fixed";
  }; return "";
}

DOMString TextAlign::cssText() const {
  switch (v) {
    case Left:  return "left";     case Center:  return "center";
    case Right: return "right";    case Justify: return "justify";
  }; return "";
}

DOMString TextDecoration::cssText() const {
  switch (v) {
    case None:        return "none";            case Overline:  return "overline";
    case Blink:       return "blink";           case Underline: return "underline";
    case LineThrough: return "line-through";
  }; return "";
}

DOMString TextTransform::cssText() const {
  switch (v) {
    case Capitalize: return "capitalize";    case Lowercase: return "lowercase";
    case Uppercase:  return "uppercase";     case None:      return "none";
  }; return "";
}

DOMString UnicodeBidi::cssText() const {
  switch (v) {
    case Normal:   return "normal";
    case Embed:    return "embed";
    case Override: return "override";
  }; return "";
}

DOMString VerticalAlign::cssText() const {
  if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
  switch (attr) {
    case Baseline: return "baseline";    case TextTop:    return "text-top";
    case Sub:      return "sub";         case Middle:     return "middle";
    case Super:    return "super";       case Bottom:     return "bottom";
    case Top:      return "top";         case TextBottom: return "text-bottom";
  }; return "";
}

DOMString Visibility::cssText() const {
  switch (v) {
    case Visible:  return "visible";
    case Hidden:   return "hidden";
    case Collapse: return "collapse";
  }; return "";
}

DOMString WhiteSpace::cssText() const {
  switch (v) {
    case Normal: return "normal";    case PreWrap: return "pre-wrap";
    case Pre:    return "pre";       case PreLine: return "pre-line";
    case Nowrap: return "nowrap";
  }; return "";
}

inline int CSSPercentRefersTo(unsigned short prt, Flow *inline_context) {
  switch (prt) {
    case DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth:           return inline_context->cur_attr.font->max_width;
    case DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight:          return inline_context->cur_attr.font->Height();
    case DOM::CSSPrimitiveValue::PercentRefersTo::LineHeight:          return inline_context->layout.line_height;
    case DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight: return inline_context->container->h;
  }; return inline_context->container->w;
}

float CSSPrimitiveValue::ConvertToPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
  switch (u) {
    case CSS_PX:         return          n;
    case CSS_EXS:        return          n  * inline_context->cur_attr.font->max_width;
    case CSS_EMS:        return          n  * inline_context->cur_attr.font->size;
    case CSS_IN:         return          n  * FLAGS_dots_per_inch;
    case CSS_CM:         return CMToInch(n) * FLAGS_dots_per_inch;
    case CSS_MM:         return MMToInch(n) * FLAGS_dots_per_inch;
    case CSS_PC:         return          n  /   6 * FLAGS_dots_per_inch;
    case CSS_PT:         return          n  /  72 * FLAGS_dots_per_inch;
    case CSS_PERCENTAGE: return          n  / 100 * CSSPercentRefersTo(prt, inline_context);
  }; return 0;
}

float CSSPrimitiveValue::ConvertFromPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
  switch (u) {
    case CSS_PX:         return          n;
    case CSS_EXS:        return          n / inline_context->cur_attr.font->max_width;
    case CSS_EMS:        return          n / inline_context->cur_attr.font->size;
    case CSS_IN:         return          n / FLAGS_dots_per_inch;
    case CSS_CM:         return InchToCM(n / FLAGS_dots_per_inch);
    case CSS_MM:         return InchToMM(n / FLAGS_dots_per_inch);
    case CSS_PC:         return          n *   6 / FLAGS_dots_per_inch;
    case CSS_PT:         return          n *  72 / FLAGS_dots_per_inch;
    case CSS_PERCENTAGE: return          n * 100 / CSSPercentRefersTo(prt, inline_context);
  }; return 0;                            
}
