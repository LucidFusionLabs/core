/*
 * $Id: css.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef LFL_CORE_WEB_CSS_H__
#define LFL_CORE_WEB_CSS_H__

extern "C" {
struct css_stylesheet;
typedef struct lwc_string_s lwc_string;
typedef struct css_select_ctx css_select_ctx;
typedef struct css_select_results css_select_results;
typedef struct css_stylesheet_params css_stylesheet_params;
};

namespace LFL {
namespace CSS {
struct PropertyNames : public vector<string> {
  PropertyNames() {
#undef  XX
#define XX(n)
#undef  YY
#define YY(n)
#undef  ZZ
#define ZZ(n) push_back(#n);
#include "core/web/css_properties.h"
  }
};

const char Default[] =
"html { display: block; }\n"
"head { display: none; }\n"
"body { display: block; margin: 8px; line-height: 1.33; }\n"
"div { display: block; }\n"
"h1 { display: block; font-size: 2em; font-weight: bold; margin: .67em 0; }\n"
"h2 { display: block; font-size: 1.5em; font-weight: bold; margin: .69em 0; }\n"
"h3 { display: block; font-size: 1.17em; font-weight: bold; margin: .83em 0; }\n"
"h4 { display: block; font-weight: bold; margin: 1.12em 0; }\n"
"h5 { display: block; font-size: .83em; font-weight: bold; margin: 1.5em 0; }\n"
"h6 { display: block; font-size: .75em; font-weight: bold; margin: 1.67em 0; }\n"
"address { display: block; font-style: italic; }\n"
"em { font-style: italic; }\n"
"strong { font-weight: bold; }\n"
"dfn { font-style: italic; text-decoration: underline; }\n"
"code { font-family: monospace; }\n"
"samp { font-family: monospace; }\n"
"kbd { font-family: monospace; font-weight: bold; }\n"
"var { font-style: italic; }\n"
"cite { font-style: italic; }\n"
"abbr { font-variant: small-caps; }\n"
"acronym { font-variant: small-caps; }\n"
"blockquote { display: block; margin: 1.12em 40px;}\n"
"q { font-style: italic; }\n"
"sub { vertical-align: sub; font-size: .83em; }\n"
"sup { vertical-align: super; font-size: .83em; }\n"
"p { display: block; margin: 1.12em 0; }\n"
"br[clear=left] { clear: left; }\n"
"br[clear=right] { clear: right; }\n"
"br[clear=all] { clear: both; }\n"
"pre { display: block; font-family: monospace; white-space: pre; margin-bottom: 1em; }\n"
"ins { color: green; text-decoration: underline; }\n"
"del { color: red; text-decoration: line-through; }\n"
"ul { display: block; padding-left: 1.5em; margin: 1.12em 0;\n"
"	list-style-type: disc; }\n"
"ol { display: block; padding-left: 1.5em; margin: 1.12em 0;\n"
"	list-style-type: decimal; }\n"
"li { display: list-item; }\n"
"ul ul { list-style-type: circle; }\n"
"ul ul ul { list-style-type: square; }\n"
"ol ul, ul ol, ul ul, ol ol { margin-top: 0; margin-bottom: 0; }\n"
"dl { display: block; padding-left: 1.5em; margin: 1em; }\n"
"dt { display: block; font-weight: bold; }\n"
"dd { display: block; padding-left: 1em; margin-bottom: 0.3em; }\n"
"dir { display: block; padding-left: 1.5em; margin: 1.12em 0; }\n"
"menu { display: block; padding-left: 1.5em; margin: 1.12em 0; }\n"
"table { display: table; border-spacing: 2px; }\n"
"table[border], table[border] td, table[border] tr { border-color: grey; }\n"
"caption { display: table-caption; }\n"
"thead { display: table-header-group; vertical-align: middle; }\n"
"tfoot { display: table-footer-group; vertical-align: middle; }\n"
"tbody { display: table-row-group; vertical-align: middle; }\n"
"colgroup { display: table-column-group; }\n"
"col { display: table-column; }\n"
"table > tr { vertical-align: middle; }\n"
"tr { display: table-row; vertical-align: inherit; }\n"
"td, th { display: table-cell; vertical-align: inherit; padding: 1px; }\n"
"th { font-weight: bold; text-align: center; }\n"
"td[nowrap], th[nowrap] { white-space: nowrap; }\n"
"a:link { color: #00f; text-decoration: underline; }\n"
"img { color: #888; }\n"
"center { display: block; text-align: center; }\n"
"center * { margin-left: auto; margin-right: auto; }\n"
"tt { font-family: monospace; }\n"
"i { font-style: italic; }\n"
"b { font-weight: bold; }\n"
"big { font-size: 1.17em; }\n"
"small { font-size: .83em; }\n"
"strike, s { text-decoration: line-through; }\n"
"u { text-decoration: underline; }\n"
"hr {	display: block; background-color: #000; height: 1px;\n"
"	margin: 4px auto; border: 1px #d9d9d9 inset; }\n"
"hr[noshade] { background-color: #888; height: 2px; border: none; }\n"
"noframes { display: none; }\n"
"iframe { width: 19em; height: 10em; }\n"
"form { display: block; }\n"
"input, button { background-color: #fff; color: #000; \n"
"	font-family: sans-serif; width: auto; height: auto; overflow: hidden;\n"
"	border: 2px solid #333; padding: 1px 2px; line-height: 1.33; }\n"
"input[type=button], input[type=reset], input[type=submit], button {\n"
"	background-color: #d9d9d9; color: #000; text-align: center;\n"
"	border: medium outset #d9d9d9; padding: 1px 0.5em; }\n"
"input[type=image] { background-color: transparent; color: #000;\n"
"	border: none; padding: 0 2px; }\n"
"input[type=checkbox], input[type=radio] { background-color: transparent;\n"
"	border: none; padding: 0 0.1em; }\n"
"input[type=file] { background-color: #d9d9d9; color: #000; font-style: italic;\n"
"	border: medium inset #d9d9d9; padding: 1px 2px; }\n"
"input[align=left] { float: left; }\n"
"input[align=right] { float: right; }\n"
"select { background-color: #d9d9d9; color: #000; \n"
"	font-family: sans-serif; width: auto; height: auto; overflow: hidden;\n"
"	margin: 1px; border: medium inset #d9d9d9; padding: 1px 3px 1px 2px;\n"
"	white-space: nowrap; }\n"
"select:after { content: \"\\25bc\"; border-left: 4px ridge #d9d9d9; }\n"
"textarea { background-color: #fff; color: #000; \n"
"	font-family: monospace; width: auto; height: auto; overflow: scroll;\n"
"	margin: 1px; border: 2px solid #333; padding: 0 2px; }\n"
"fieldset { display: block; border: thin solid #888; margin: 1.12em 0; }\n"
"noembed, script, style, title { display: none; }\n";

}; // namespace CSS
namespace DOM {

struct Rect;
struct Counter;
struct RGBColor;
struct StyleSheet;
struct CSSStyleSheet;

struct CSSValue : public Object {
  enum {
    CSS_INHERIT = 0, CSS_PRIMITIVE_VALUE = 1, CSS_VALUE_LIST = 2, CSS_CUSTOM = 3
  };

  virtual DOMString cssText() const = 0;
  virtual bool      Null()    const = 0;

  unsigned short cssValueType;
  CSSValue(unsigned short type=0) : cssValueType(type) {}
};

struct CSSRule : public Object {
  enum {
    UNKNOWN_RULE = 0, STYLE_RULE = 1, CHARSET_RULE = 2, IMPORT_RULE = 3, MEDIA_RULE = 4, FONT_FACE_RULE = 5, PAGE_RULE = 6
  }; 

  virtual DOMString cssText() const = 0;

  unsigned short  type;
  CSSStyleSheet  *parentStyleSheet;
  CSSRule        *parentRule;
  CSSRule(unsigned short Type=0, CSSStyleSheet *ps=0, CSSRule *pr=0) : type(Type), parentStyleSheet(ps), parentRule(pr)  {}
};

struct CSSValueList : public CSSValue {
  CSSValueList() : CSSValue(CSS_VALUE_LIST) {}
  virtual int length() const = 0;
  virtual       CSSValue *item(int index)       = 0;
  virtual const CSSValue *item(int index) const = 0;
};

struct CSSValueListVector : public CSSValueList {
  vector<CSSValue*> data;
  virtual bool Null() const { return !length(); }
  virtual int length() const { return data.size(); }
  virtual       CSSValue* item(int index)       { return (index >= 0 && index < data.size()) ? data[index] : 0; }
  virtual const CSSValue* item(int index) const { return (index >= 0 && index < data.size()) ? data[index] : 0; }
  virtual DOMString cssText() const { DOMString ret; for (int i=0; i<data.size(); i++) StrAppend(&ret, ret.size()?" ":"", item(i)->cssText()); return ret; }
};

struct CSSRuleList    : public List<CSSRule*>    {};
struct StyleSheetList : public List<StyleSheet*> {};
struct MediaList      : public List<DOMString> {
  DOMString mediaText;
  void deleteMedium(DOMString oldMedium) { FilterByValue(&data, oldMedium); }
  void appendMedium(DOMString newMedium) { deleteMedium(newMedium); data.push_back(newMedium); }
};

struct LinkStyle     : public Object { StyleSheet          *sheet; };
struct DocumentStyle : public Object { StyleSheetList styleSheets; };

struct StyleSheet : public Object {
  DOMString   type;
  bool        disabled;
  Node       *ownerNode;
  StyleSheet *parentStyleSheet;
  DOMString   href, title;
  MediaList   media;
  StyleSheet(Node *owner, StyleSheet *parent) : disabled(0), ownerNode(owner), parentStyleSheet(parent) {}
};

struct CSSStyleSheet : public StyleSheet {
  CSSRule    *ownerRule;
  CSSRuleList cssRules;
  CSSStyleSheet(Node *owner, StyleSheet *parent) : StyleSheet(owner, parent), ownerRule(0) {}

  int insertRule(const DOMString &rule, int index) { return -1; }
  void deleteRule(int index) { CHECK_RANGE(index, 0, cssRules.length()); VectorErase(&cssRules.data, index); }
};

struct CSSStyleDeclaration : public Object {
  CSSRule   *parentRule;
  DOMString  cssText;
  CSSStyleDeclaration(CSSRule *p=0) : parentRule(p) {}

  virtual int        length()        const = 0;
  virtual DOMString  item(int index) const = 0;

  virtual DOMString       getPropertyPriority(const DOMString &propertyName) const = 0;
  virtual DOMString       getPropertyValue(const DOMString &propertyName)    const = 0;
  virtual const CSSValue *getPropertyCSSValue(const DOMString &propertyName) const = 0;
  virtual DOMString       removeProperty(const DOMString &propertyName) = 0;
  virtual void            setProperty(const DOMString &propertyName, const DOMString &value, const DOMString &priority) = 0;
  virtual void            setPropertyValue(const DOMString &n, const DOMString &v) { return setProperty(n, v, ""); }
};

struct CSSStyleDeclarationMap : public CSSStyleDeclaration {
  NamedMap<CSSValue*> propMap;
  CSSStyleDeclarationMap(CSSRule *p=0) : CSSStyleDeclaration(p) {}

  virtual int        length() const { return propMap.length(); };
  virtual DOMString  item(int index) const { auto v = propMap.item(index); return v ? v->cssText() : ""; }

  virtual DOMString       getPropertyValue(const DOMString &n) const { CSSValue *v = propMap.getNamedItem(n);  return v ? v->cssText() : ""; }
  virtual const CSSValue *getPropertyCSSValue(const DOMString &n) const { return propMap.getNamedItem(n); }
  virtual DOMString       getPropertyPriority(const DOMString &n) const { return ""; }
  virtual DOMString       removeProperty(const DOMString &n) { CSSValue *v = propMap.removeNamedItem(n); return v ? v->cssText() : ""; }
  virtual void            setProperty(const DOMString &n, const DOMString &v, const DOMString &priority) { ERROR("not implemented"); }
};

struct CSSPrimitiveValue : public CSSValue {
  struct PercentRefersTo { enum { ContainingBoxWidth=0, ContainingBoxHeight=1, FontWidth=2, FontHeight=3, LineHeight=4 }; };
  enum {
    CSS_UNKNOWN = 0, CSS_NUMBER = 1, CSS_PERCENTAGE = 2, CSS_EMS = 3, CSS_EXS = 4, CSS_PX = 5, CSS_CM = 6, CSS_MM = 7,
    CSS_IN = 8, CSS_PT = 9, CSS_PC = 10, CSS_DEG = 11, CSS_RAD = 12, CSS_GRAD = 13, CSS_MS = 14, CSS_S = 15, CSS_HZ = 16,
    CSS_KHZ = 17, CSS_DIMENSION = 18, CSS_STRING  = 19, CSS_URI = 20, CSS_IDENT = 21, CSS_ATTR = 22, CSS_COUNTER = 23,
    CSS_RECT = 24, CSS_RGBCOLOR = 25
  };
  unsigned short primitiveType, percentRefersTo;
  CSSPrimitiveValue(unsigned short type=0, unsigned short prt=0) : CSSValue(CSS_PRIMITIVE_VALUE), primitiveType(type), percentRefersTo(prt) {}
  bool IsPercent() const { return primitiveType == CSS_PERCENTAGE; }

  virtual void  setStringValue(unsigned short U, const DOMString &V) {}
  virtual void  setFloatValue (unsigned short U, float            V) {}
  virtual float getFloatValue (unsigned short U, Flow *inline_context) { return -1; }

  int getPixelValue(Flow *inline_context) { return RoundF(getFloatValue(CSS_PX, inline_context)); }
  float getFloatValue(unsigned short U, float V, Flow *inline_context) {
    if (U == CSS_PX) return                      ConvertToPixels(V, primitiveType, percentRefersTo, inline_context);
    else             return ConvertFromPixels(U, ConvertToPixels(V, primitiveType, percentRefersTo, inline_context), percentRefersTo, inline_context);
  }

  DOMString  getStringValue()   { return cssText(); }
  Counter   *getCounterValue()  { return primitiveType == CSS_COUNTER  ? reinterpret_cast<Counter*> (this) : nullptr; }
  Rect      *getRectValue()     { return primitiveType == CSS_RECT     ? reinterpret_cast<Rect*>    (this) : nullptr; }
  RGBColor  *getRGBColorValue() { return primitiveType == CSS_RGBCOLOR ? reinterpret_cast<RGBColor*>(this) : nullptr; }

  static float ConvertToPixels  (float n, unsigned short u, unsigned short prt, Flow *inline_context);
  static float ConvertFromPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context);

  const char *GetUnitString() const { return GetUnitString(primitiveType); }
  static const char *GetUnitString(unsigned short u) {
    switch (u) {
      case CSS_PX:  return "px";    case CSS_MM:         return "mm";
      case CSS_EXS: return "ex";    case CSS_PT:         return "pt";
      case CSS_EMS: return "em";    case CSS_PC:         return "pc";
      case CSS_IN:  return "in";    case CSS_PERCENTAGE: return "%" ;
      case CSS_CM:  return "cm";
    }; return "";
  }
};

struct CSSStringValue : public CSSPrimitiveValue {
  DOMString v;
  CSSStringValue()                   : CSSPrimitiveValue(CSS_STRING)       {}
  CSSStringValue(const DOMString &V) : CSSPrimitiveValue(CSS_STRING), v(V) {}
  virtual bool      Null()    const { return v.empty(); }
  virtual DOMString cssText() const { return v; }
  virtual void  setStringValue(unsigned short U, const DOMString &V) { v = StrCat(V, " ", GetUnitString(U)); }
  virtual void  setFloatValue (unsigned short U, float            V) { v = StringPrintf("%g", V); }
  virtual float getFloatValue (unsigned short U, Flow *inline_context) { return CSSPrimitiveValue::getFloatValue(U, atof(String::ToAscii(v).c_str()), inline_context); }
};

struct CSSNumericValue : public CSSPrimitiveValue {
  float v;
  CSSNumericValue() : v(0) {}
  CSSNumericValue(unsigned short PRT)                              : CSSPrimitiveValue(0, PRT), v(0) {}
  CSSNumericValue(float V, unsigned short U, unsigned short PRT=0) : CSSPrimitiveValue(U, PRT), v(V) {}
  void SetValue(float V, unsigned short U) { v=V; primitiveType=U; }
  virtual bool      Null()    const { return !v && !primitiveType; }
  virtual DOMString cssText() const { return Null() ? "" : StrCat(v, " ", GetUnitString()); }
  virtual void  setStringValue(unsigned short U, const DOMString &V) { primitiveType = U; v = atof(String::ToAscii(V).c_str()); }
  virtual void  setFloatValue (unsigned short U, float            V) { primitiveType = U; v = V; }
  virtual float getFloatValue (unsigned short U, Flow *inline_context) { return CSSPrimitiveValue::getFloatValue(U, v, inline_context); }
};

struct CSSAutoNumericValue : public CSSNumericValue {
  bool _auto;
  CSSAutoNumericValue(bool A) : _auto(A) {}
  CSSAutoNumericValue(float V, unsigned short U) : CSSNumericValue(V, U), _auto(0) {}
  virtual bool      Null()    const { return !_auto && !v && !primitiveType; }
  virtual DOMString cssText() const { return _auto ? "auto" : CSSNumericValue::cssText(); }
};

struct CSSNoneNumericValue : public CSSNumericValue {
  bool _none;
  CSSNoneNumericValue(bool N) : _none(N) {}
  CSSNoneNumericValue(float V, unsigned short U) : CSSNumericValue(V, U), _none(0) {}
  virtual bool      Null()    const { return !_none && !v && !primitiveType; }
  virtual DOMString cssText() const { return _none ? "none" : CSSNumericValue::cssText(); }
};

struct CSSNormNumericValue : public CSSNumericValue {
  bool _norm;
  CSSNormNumericValue(bool N,                    unsigned PRT) : CSSNumericValue(0, 0, PRT), _norm(N) {}
  CSSNormNumericValue(float V, unsigned short U, unsigned PRT) : CSSNumericValue(V, U, PRT), _norm(0) {}
  virtual bool      Null()    const { return !_norm && !v && !primitiveType; }
  virtual DOMString cssText() const { return _norm ? "normal" : CSSNumericValue::cssText(); }
};

struct CSSNumericValuePair : public CSSValueList {
  CSSNumericValue first, second;
  CSSNumericValuePair() {}
  CSSNumericValuePair(int v1, unsigned short u1, unsigned short prt1, int v2, unsigned short u2, unsigned short prt2) : first(v1, u1, prt1), second(v2, u2, prt2) {}
  virtual       CSSValue* item(int index)       { if (index == 1) return &first; else if (index == 2) return &second; return 0; }
  virtual const CSSValue* item(int index) const { if (index == 1) return &first; else if (index == 2) return &second; return 0; }
  virtual int       length()  const { return Null() ? 0 : 2; }
  virtual bool      Null()    const { return first.Null() && second.Null(); }
  virtual DOMString cssText() const {
    DOMString fs = first.cssText(), ss = second.cssText();
    return StrCat(fs, fs.size() ? " " : (ss.size() ? "0 " : ""), ss);
  }
};

struct Rect : public CSSPrimitiveValue {
  bool top_auto, right_auto, bottom_auto, left_auto;
  CSSNumericValue top, right, bottom, left; 
  Rect()                                                                                                           : CSSPrimitiveValue(CSS_RECT), top_auto( 1), right_auto( 1), bottom_auto( 1), left_auto( 1), top(LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight), bottom(LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight) {}
  Rect(         float v1, int u1,          float v2, int u2,          float v3, int u3,          float v4, int u4) : CSSPrimitiveValue(CSS_RECT), top_auto( 0), right_auto( 0), bottom_auto( 0), left_auto( 0), top(v1, u1, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight), right(v2, u2), bottom(v3, u3, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight), left(v4, u4) {}
  Rect(bool a1, float v1, int u1, bool a2, float v2, int u2, bool a3, float v3, int u3, bool a4, float v4, int u4) : CSSPrimitiveValue(CSS_RECT), top_auto(a1), right_auto(a2), bottom_auto(a3), left_auto(a4), top(v1, u1, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight), right(v2, u2), bottom(v3, u3, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight), left(v4, u4) {}
  virtual bool      Null()    const { return top_auto && right_auto && bottom_auto && left_auto; }
  virtual DOMString cssText() const {
    return StrCat(ReplaceEmpty(   top_auto ? "auto" : top   .cssText(), "0"), " ",
                  ReplaceEmpty( right_auto ? "auto" : right .cssText(), "0"), " ",
                  ReplaceEmpty(bottom_auto ? "auto" : bottom.cssText(), "0"), " ",
                  ReplaceEmpty(  left_auto ? "auto" : left  .cssText(), "0"), " ");
  }
};

struct BackgroundAttachment : public CSSValue {
  static const int Fixed, Scroll;
  int v;
  BackgroundAttachment(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct BackgroundRepeat : public CSSValue {
  static const int RepeatX, RepeatY, Repeat, NoRepeat;
  int v;
  BackgroundRepeat(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct BorderCollapse : public CSSValue {
  static const int Separate, Collapse;
  int v;
  BorderCollapse(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct BorderWidth : public CSSNumericValue {
  static const int Thin, Medium, Thick;
  int attr;
  BorderWidth(int A=0) : attr(A) {}
  BorderWidth(int V, unsigned short U) : CSSNumericValue(V, U), attr(0) {}
  virtual bool Null() const { return !attr && CSSNumericValue::Null(); }
  virtual DOMString cssText() const;
  virtual float getFloatValue(unsigned short U, Flow *inline_context);
};

struct BorderStyle : public CSSValue {
  static const int None, Hidden, Dotted, Dashed, Solid, Double, Groove, Ridge, Inset, Outset;
  int v;
  BorderStyle(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !(v && v != None && v != Hidden); }
  virtual DOMString cssText() const;
};

struct CaptionSide : public CSSValue {
  static const int Top, Bottom;
  int v;
  CaptionSide(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Clear : public CSSValue {
  static const int None, Left, Right, Both;
  int v;
  Clear(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Cursor : public CSSValue {
  static const int Auto, Default, NEResize, WResize, Move, Pointer, EResize, NWResize, Text;
  static const int Crosshair, SEResize, Wait, Progress, SResize, Help, NResize, SWResize;
  int v;
  vector<DOMString> url; Cursor(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Direction : public CSSValue {
  static const int LTR, RTL;
  int v;
  Direction(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Display : public CSSValue {
  static const int None, Row, HeaderGroup, Inline, Cell, FooterGroup, Block, Column, InlineBlock;
  static const int RunIn, Caption, InlineTable, ListItem, RowGroup, Table, ColGroup;
  int v;
  Display(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool TableElement() const {
    return v == Row || v == Cell || v == Column || v == Caption || v == RowGroup || v == ColGroup || v == HeaderGroup || v == FooterGroup;
  }
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct EmptyCells : public CSSValue {
  static const int Show, Hide;
  int v;
  EmptyCells(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Float : public CSSValue { 
  static const int Left, Right, None; 
  int v;
  Float(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v || v == None; }
  virtual DOMString cssText() const;
};

struct FontFamily : public CSSValue {
  static const int Serif, SansSerif, Cursive, Fantasy, Monospace;
  int attr;
  vector<string> name;
  FontFamily(int A=0)         : CSSValue(CSS_CUSTOM), attr(A) {}
  FontFamily(const string &n) : CSSValue(CSS_CUSTOM), attr(0) { name.push_back(n); } 
  virtual bool Null() const { return name.empty() && !attr; }
  virtual DOMString cssText() const;
};

struct FontSize : public CSSNumericValue {
  static const int XXSmall, XLarge, XSmall, XXLarge, Small, Larger, Medium, Smaller, Large;
  int attr;
  FontSize(int A=0)                 :                        attr(A) {}
  FontSize(int V, unsigned short U) : CSSNumericValue(V, U), attr(0) {}
  virtual int getFontSizeValue(Flow *flow);
  virtual bool Null() const { return CSSNumericValue::Null() && !attr; }
  virtual DOMString cssText() const;
};

struct FontStyle : public CSSValue {
  static const int Normal, Italic, Oblique;
  int v;
  FontStyle(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct FontVariant : public CSSValue {
  static const int Normal, SmallCaps;
  int v;
  FontVariant(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct FontWeight : public CSSValue {
  static const int Normal, Bold, Bolder, Lighter, _100, _200, _300, _400, _500, _600, _700, _800, _900;
  int v;
  FontWeight(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct FontFace {
  FontStyle style;
  FontFamily family;
  FontWeight weight;
  vector<string> source;
};

struct ListStylePosition : public CSSValue {
  static const int Inside, Outside;
  int v;
  ListStylePosition(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct ListStyleType : public CSSValue {
  static const int Disc, Circle, Square, Decimal, DecimalLeadingZero, LowerRoman, UpperRoman;
  static const int LowerGreek, LowerLatin, UpperLatin, Armenian, Georgian, LowerAlpha, UpperAlpha, None;
  int v;
  ListStyleType(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Overflow : public CSSValue {
  static const int Visible, Hidden, Scroll, Auto;
  int v;
  Overflow(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct PageBreak : public CSSValue {
  static const int Auto, Avoid, Always, Left, Right, Page, Column, AvoidPage, AvoidColumn;
  int v;
  PageBreak(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Position : public CSSValue {
  static const int Static, Relative, Absolute, Fixed;
  int v;
  Position(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct RGBColor : public CSSValue {
  Color v;
  RGBColor(int hexval) : CSSValue(CSS_CUSTOM), v(hexval) {}
  RGBColor()           : CSSValue(CSS_INHERIT) {}
  int red  () const { return v.R(); }
  int green() const { return v.G(); }
  int blue () const { return v.B(); }
  virtual bool Null() const { return false; }
  virtual DOMString cssText() const { return v.HexString(); }
};

struct TableLayout : public CSSValue {
  static const int Auto, Fixed;
  int v;
  TableLayout(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct TextAlign : public CSSValue {
  static const int Left, Right, Center, Justify;
  int v;
  TextAlign(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v || v == Left; }
  virtual DOMString cssText() const;
};

struct TextDecoration : public CSSValue {
  static const int None, Blink, LineThrough, Overline, Underline;
  int v;
  TextDecoration(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct TextTransform : public CSSValue {
  static const int Capitalize, Uppercase, Lowercase, None;
  int v;
  TextTransform(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct UnicodeBidi : public CSSValue {
  static const int Normal, Embed, Override;
  int v;
  UnicodeBidi(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct VerticalAlign : public CSSNumericValue {
  static const int Baseline, Sub, Super, Top, TextTop, Middle, Bottom, TextBottom;
  int attr;
  VerticalAlign(int A=0) : attr(A) {}
  VerticalAlign(int V, unsigned short U) : CSSNumericValue(V, U, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::LineHeight), attr(0) {}
  virtual bool Null() const { return !attr && CSSNumericValue::Null(); }
  virtual DOMString cssText() const;
};

struct Visibility : public CSSValue {
  static const int Visible, Hidden, Collapse;
  int v;
  Visibility(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct WhiteSpace : public CSSValue {
  static const int Normal, Pre, Nowrap, PreWrap, PreLine;
  int v;
  WhiteSpace(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
  virtual bool Null() const { return !v; }
  virtual DOMString cssText() const;
};

struct Counter               : public Object { DOMString identifier, listStyle, separator; };
struct ElementCSSInlineStyle : public Object { CSSStyleDeclarationMap style; };

struct CSSFontFaceRule : public CSSRule {
  CSSStyleDeclarationMap style; 
  CSSFontFaceRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(FONT_FACE_RULE, ps, pr) {}
  virtual DOMString cssText() const { return ""; }
};

struct CSSCharsetRule : public CSSRule {
  DOMString encoding;
  CSSCharsetRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(CHARSET_RULE, ps, pr) {}
  virtual DOMString cssText() const { return ""; }
};

struct CSSUnknownRule : public CSSRule {
  CSSUnknownRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(UNKNOWN_RULE, ps, pr) {}
  virtual DOMString cssText() const { return ""; }
};

struct CSSStyleRule : public CSSRule {
  CSSStyleDeclarationMap style;
  DOMString              selectorText;
  CSSStyleRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(STYLE_RULE, ps, pr) {}
  virtual DOMString cssText() const { return ""; }
};

struct CSSMediaRule : public CSSRule {
  MediaList   media;
  CSSRuleList cssRules;
  CSSMediaRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(MEDIA_RULE, ps, pr) {}

  virtual DOMString cssText() const { return ""; }
  int insertRule(const DOMString &rule, int index) { return -1; }
  void deleteRule(int index) { CHECK_RANGE(index, 0, cssRules.length()); VectorErase(&cssRules.data, index); }
};

struct CSSPageRule : public CSSRule {
  CSSStyleDeclarationMap style;
  DOMString              selectorText;
  CSSPageRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(PAGE_RULE, ps, pr) {}
  virtual DOMString cssText() const { return ""; }
};

struct CSSImportRule : public CSSRule {
  DOMString      href;
  MediaList      media;
  CSSStyleSheet *styleSheet;
  CSSImportRule(CSSStyleSheet *ps=0, CSSRule *pr=0) : CSSRule(IMPORT_RULE, ps, pr), styleSheet(0) {}
  virtual DOMString cssText() const { return ""; }
};
}; // namespace DOM

struct StyleSheet : public LFL::DOM::Object {
  LFL::DOM::Document *ownerDocument;
  css_stylesheet *sheet=0;
  css_stylesheet_params *params=0;
  string content_url, character_set;

  ~StyleSheet();
  StyleSheet(LFL::DOM::Document *D, const char *U=0, const char *CS=0, bool in_line=0, bool quirks=0, const char *Content=0);

  void Parse(File *f) { NextRecordReader nr(f); for (const char *l = nr.NextLine(); l; l = nr.NextLine()) Parse(l, nr.record_len); }
  void Parse(const string &content) { return Parse(content.c_str(), content.size()); }
  void Parse(const char *content, int content_len);
  void Done();

  static StyleSheet *Default();
};

struct ComputedStyle : public LFL::DOM::CSSStyleDeclaration {
  LFL::DOM::Node *node;
  string override_style;
  css_select_results *style=0;
  bool is_root=0, font_not_inherited=0, color_not_inherited=0, bgcolor_not_inherited=0, class_cached=0;
  vector<lwc_string*> class_cache;
  ComputedStyle(LFL::DOM::Node *N) : node(N) {}
  virtual ~ComputedStyle() { Reset(); }

  bool Computed() const;
  bool IsLink() const;
  void Reset();

  LFL::DOM::Display              Display             () const;
  LFL::DOM::Clear                Clear               () const;
  LFL::DOM::Float                Float               () const;
  LFL::DOM::Position             Position            () const;
  LFL::DOM::Overflow             Overflow            () const;
  LFL::DOM::TextAlign            TextAlign           () const;
  LFL::DOM::Direction            Direction           () const;
  LFL::DOM::FontStyle            FontStyle           () const;
  LFL::DOM::EmptyCells           EmptyCells          () const;
  LFL::DOM::FontWeight           FontWeight          () const;
  LFL::DOM::Visibility           Visibility          () const;
  LFL::DOM::WhiteSpace           WhiteSpace          () const;
  LFL::DOM::CaptionSide          CaptionSide         () const;
  LFL::DOM::FontVariant          FontVariant         () const;
  LFL::DOM::TableLayout          TableLayout         () const;
  LFL::DOM::UnicodeBidi          UnicodeBidi         () const;
  LFL::DOM::BorderStyle          OutlineStyle        () const;
  LFL::DOM::TextTransform        TextTransform       () const;
  LFL::DOM::TextDecoration       TextDecoration      () const;
  LFL::DOM::BorderCollapse       BorderCollapse      () const;
  LFL::DOM::ListStyleType        ListStyleType       () const;
  LFL::DOM::ListStylePosition    ListStylePosition   () const;
  LFL::DOM::PageBreak            PageBreakAfter      () const;
  LFL::DOM::PageBreak            PageBreakBefore     () const;
  LFL::DOM::PageBreak            PageBreakInside     () const;
  LFL::DOM::BorderStyle          BorderTopStyle      () const;
  LFL::DOM::BorderStyle          BorderLeftStyle     () const;
  LFL::DOM::BorderStyle          BorderRightStyle    () const;
  LFL::DOM::BorderStyle          BorderBottomStyle   () const;
  LFL::DOM::BackgroundRepeat     BackgroundRepeat    () const;
  LFL::DOM::BackgroundAttachment BackgroundAttachment() const;
  LFL::DOM::RGBColor             BackgroundColor     () const;
  LFL::DOM::RGBColor             BorderBottomColor   () const;
  LFL::DOM::RGBColor             BorderLeftColor     () const;
  LFL::DOM::RGBColor             BorderRightColor    () const;
  LFL::DOM::RGBColor             BorderTopColor      () const;
  LFL::DOM::RGBColor             OutlineColor        () const;
  LFL::DOM::RGBColor             Color               () const;
  LFL::DOM::Cursor               Cursor              () const;
  LFL::DOM::FontFamily           FontFamily          () const;
  LFL::DOM::CSSStringValue       BackgroundImage     () const;
  LFL::DOM::CSSNumericValuePair  BackgroundPosition  () const;
  LFL::DOM::Rect                 Clip                () const;
  LFL::DOM::CSSNumericValue      Opacity             () const;
  LFL::DOM::CSSNumericValue      Orphans             () const;
  LFL::DOM::CSSNumericValue      Widows              () const;
  LFL::DOM::CSSNumericValue      MinHeight           () const;
  LFL::DOM::CSSNumericValue      MinWidth            () const;
  LFL::DOM::CSSNumericValue      PaddingTop          () const;
  LFL::DOM::CSSNumericValue      PaddingRight        () const;
  LFL::DOM::CSSNumericValue      PaddingBottom       () const;
  LFL::DOM::CSSNumericValue      PaddingLeft         () const;
  LFL::DOM::CSSNumericValue      TextIndent          () const;
  LFL::DOM::CSSAutoNumericValue  Top                 () const;
  LFL::DOM::CSSAutoNumericValue  Bottom              () const;
  LFL::DOM::CSSAutoNumericValue  Left                () const;
  LFL::DOM::CSSAutoNumericValue  Right               () const;
  LFL::DOM::CSSAutoNumericValue  Width               () const;
  LFL::DOM::CSSAutoNumericValue  Height              () const;
  LFL::DOM::FontSize             FontSize            () const;
  LFL::DOM::VerticalAlign        VerticalAlign       () const;
  LFL::DOM::BorderWidth          BorderTopWidth      () const;
  LFL::DOM::BorderWidth          BorderRightWidth    () const;
  LFL::DOM::BorderWidth          BorderBottomWidth   () const;
  LFL::DOM::BorderWidth          BorderLeftWidth     () const;
  LFL::DOM::BorderWidth          OutlineWidth        () const;
  LFL::DOM::CSSAutoNumericValue  MarginBottom        () const;
  LFL::DOM::CSSAutoNumericValue  MarginLeft          () const;
  LFL::DOM::CSSAutoNumericValue  MarginRight         () const;
  LFL::DOM::CSSAutoNumericValue  MarginTop           () const;
  LFL::DOM::CSSNoneNumericValue  MaxHeight           () const;
  LFL::DOM::CSSNoneNumericValue  MaxWidth            () const;
  LFL::DOM::CSSAutoNumericValue  ZIndex              () const;
  LFL::DOM::CSSNormNumericValue  WordSpacing         () const;
  LFL::DOM::CSSNormNumericValue  LetterSpacing       () const;
  LFL::DOM::CSSNormNumericValue  LineHeight          () const;
  
  virtual const LFL::DOM::CSSValue *getPropertyCSSValue(const DOMString &n) const;
  virtual DOMString getPropertyPriority(const DOMString &n) const;
  virtual DOMString removeProperty(const DOMString &n);
  virtual int length() const;
  virtual DOMString item(int index) const;
  virtual DOMString getPropertyValue(const DOMString &k) const;
  virtual void setProperty(const DOMString &k, const DOMString &v, const DOMString &priority);
};

struct StyleContext : public LFL::DOM::Object {
  LFL::DOM::HTMLDocument *doc;
  css_select_ctx *select_ctx=0;
  StyleContext(LFL::DOM::HTMLDocument *D);
  virtual ~StyleContext();

  void AppendSheet(StyleSheet *ss);
  void Match(ComputedStyle *out, LFL::DOM::Node *node, const ComputedStyle *parent_sheet=0, StyleSheet *inline_style=0);
  int FontFaces(const string &face_name, vector<LFL::DOM::FontFace> *out);
};

}; // namespace LFL
#endif // LFL_CORE_WEB_CSS_H__
