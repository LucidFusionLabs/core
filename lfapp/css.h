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

#ifndef __LFL_LFAPP_CSS_H__
#define __LFL_LFAPP_CSS_H__

#ifdef LFL_LIBCSS
extern "C" {
#include "libcss/libcss.h"
};
#endif

namespace LFL {
namespace CSS {

struct PropertyNames : public vector<string> {
    PropertyNames() {
#undef  XX
#define XX(n) push_back(#n);
#undef  YY
#define YY(n)
#include "crawler/css_properties.h"
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
    virtual CSSValue *item(int index) = 0;
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
    void deleteMedium(DOMString oldMedium) { FilterValues(&data, oldMedium); }
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

    virtual int        length()        = 0;
    virtual DOMString  item(int index) = 0;

    virtual DOMString  getPropertyValue(const DOMString &propertyName)    = 0;
    virtual CSSValue  *getPropertyCSSValue(const DOMString &propertyName) = 0;
    virtual DOMString  removeProperty(const DOMString &propertyName)      = 0;
    virtual DOMString  getPropertyPriority(const DOMString &propertyName) = 0;
    virtual void       setProperty(const DOMString &propertyName, const DOMString &value, const DOMString &priority) = 0;
};

struct CSSStyleDeclarationMap : public CSSStyleDeclaration {
    NamedMap<CSSValue*> propMap;
    CSSStyleDeclarationMap(CSSRule *p=0) : CSSStyleDeclaration(p) {}

    virtual int        length() { return propMap.length(); };
    virtual DOMString  item(int index) { CSSValue *v = propMap.item(index); return v ? v->cssText() : ""; }

    virtual DOMString  getPropertyValue(const DOMString &n) { CSSValue *v = propMap.getNamedItem(n);  return v ? v->cssText() : ""; }
    virtual DOMString  removeProperty(const DOMString &n) { CSSValue *v = propMap.removeNamedItem(n); return v ? v->cssText() : ""; }
    virtual CSSValue  *getPropertyCSSValue(const DOMString &n) { return propMap.getNamedItem(n); }
    virtual DOMString  getPropertyPriority(const DOMString &n) { return ""; }
    virtual void       setProperty(const DOMString &n, const DOMString &v, const DOMString &priority) { ERROR("not implemented"); }
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
    Counter   *getCounterValue()  { return primitiveType == CSS_COUNTER  ? (Counter*) this : 0; }
    Rect      *getRectValue()     { return primitiveType == CSS_RECT     ? (Rect*)    this : 0; }
    RGBColor  *getRGBColorValue() { return primitiveType == CSS_RGBCOLOR ? (RGBColor*)this : 0; }

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
    virtual CSSValue* item(int index) { if (index == 1) return &first; else if (index == 2) return &second; return 0; }
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
    int v; BackgroundAttachment(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Fixed = CSS_BACKGROUND_ATTACHMENT_FIXED, Scroll = CSS_BACKGROUND_ATTACHMENT_SCROLL };
#else
    enum { Fixed, Scroll };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Fixed:  return "fixed";
            case Scroll: return "scroll";
        }; return "";
    }
};

struct BackgroundRepeat : public CSSValue {
    int v; BackgroundRepeat(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        RepeatX = CSS_BACKGROUND_REPEAT_REPEAT_X, RepeatY  = CSS_BACKGROUND_REPEAT_REPEAT_Y,
        Repeat  = CSS_BACKGROUND_REPEAT_REPEAT,   NoRepeat = CSS_BACKGROUND_REPEAT_NO_REPEAT
#else
        RepeatX, RepeatY, Repeat, NoRepeat
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case RepeatX: return "repeat-x";    case Repeat:   return "repeat";
            case RepeatY: return "repeat-y";    case NoRepeat: return "no-repeat";
        }; return "";
    }
};

struct BorderCollapse : public CSSValue {
    int v; BorderCollapse(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Separate = CSS_BORDER_COLLAPSE_SEPARATE, Collapse = CSS_BORDER_COLLAPSE_COLLAPSE };
#else
    enum { Separate, Collapse };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Separate: return "separate";
            case Collapse: return "collapse";
        }; return "";
    }
};

struct BorderWidth : public CSSNumericValue {
    int attr;
    BorderWidth(int A=0) : attr(A) {}
    BorderWidth(int V, unsigned short U) : CSSNumericValue(V, U), attr(0) {}
#ifdef LFL_LIBCSS
    enum { Thin = CSS_BORDER_WIDTH_THIN, Medium = CSS_BORDER_WIDTH_MEDIUM, Thick = CSS_BORDER_WIDTH_THICK };
#else
    enum { Thin, Medium, Thick };
#endif
    virtual bool Null() const { return !attr && CSSNumericValue::Null(); }
    virtual DOMString cssText() const {
        if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
        switch (attr) {
            case Thin:   return "thin";  
            case Medium: return "medium";
            case Thick:  return "thick"; 
        }; return "";
    }
    virtual float getFloatValue(unsigned short U, Flow *inline_context) {
        switch (attr) {
            case Thin:   return ConvertFromPixels(2, U, percentRefersTo, inline_context);
            case Medium: return ConvertFromPixels(4, U, percentRefersTo, inline_context);
            case Thick:  return ConvertFromPixels(6, U, percentRefersTo, inline_context);
        }; return CSSNumericValue::getFloatValue(U, inline_context); 
    }
};

struct BorderStyle : public CSSValue {
    int v; BorderStyle(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        None   = CSS_BORDER_STYLE_NONE,   Hidden = CSS_BORDER_STYLE_HIDDEN, Dotted = CSS_BORDER_STYLE_DOTTED,
        Dashed = CSS_BORDER_STYLE_DASHED, Solid  = CSS_BORDER_STYLE_SOLID,  Double = CSS_BORDER_STYLE_DOUBLE,
        Groove = CSS_BORDER_STYLE_GROOVE, Ridge  = CSS_BORDER_STYLE_RIDGE,  Inset  = CSS_BORDER_STYLE_INSET,
        Outset = CSS_BORDER_STYLE_OUTSET
#else
        None, Hidden, Dotted, Dashed, Solid, Double, Groove, Ridge, Inset, Outset
#endif
    };
    virtual bool Null() const { return !(v && v != None && v != Hidden); }
    virtual DOMString cssText() const {
        switch (v) {
            case None:   return "none";      case Double: return "double";
            case Hidden: return "hidden";    case Groove: return "groove";
            case Dotted: return "dotted";    case Ridge:  return "ridge";
            case Dashed: return "dashed";    case Inset:  return "inset";
            case Solid:  return "solid";     case Outset: return "outset";
        }; return "";
    }
};

struct CaptionSide : public CSSValue {
    int v; CaptionSide(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Top = CSS_CAPTION_SIDE_TOP, Bottom = CSS_CAPTION_SIDE_BOTTOM };
#else
    enum { Top, Bottom };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Top:    return "top";
            case Bottom: return "bottom";
        }; return "";
    };
};

struct Clear : public CSSValue {
    int v; Clear(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { None = CSS_CLEAR_NONE, Left = CSS_CLEAR_LEFT, Right = CSS_CLEAR_RIGHT, Both = CSS_CLEAR_BOTH };
#else
    enum { None, Left, Right, Both };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case None: return "none";    case Right: return "right";
            case Left: return "left";    case Both:  return "both";
        }; return "";
    };
};

struct Cursor : public CSSValue {
    int v; vector<DOMString> url; Cursor(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Auto = CSS_CURSOR_AUTO, Default   = CSS_CURSOR_DEFAULT,   NEResize  = CSS_CURSOR_NE_RESIZE, WResize  = CSS_CURSOR_W_RESIZE,
        Move = CSS_CURSOR_MOVE, Pointer   = CSS_CURSOR_POINTER,   EResize   = CSS_CURSOR_E_RESIZE,  NWResize = CSS_CURSOR_NW_RESIZE,
        Text = CSS_CURSOR_TEXT, Crosshair = CSS_CURSOR_CROSSHAIR, SEResize  = CSS_CURSOR_SE_RESIZE,
        Wait = CSS_CURSOR_WAIT, Progress  = CSS_CURSOR_PROGRESS,  SResize   = CSS_CURSOR_S_RESIZE,
        Help = CSS_CURSOR_HELP, NResize   = CSS_CURSOR_N_RESIZE,  SWResize  = CSS_CURSOR_SW_RESIZE
#else
        Auto, Default, NEResize, WResize, Move, Pointer, EResize, NWResize, Text, Crosshair, SEResize, Wait, Progress,
        SResize, Help, NResize, SWResize 
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
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
};

struct Direction : public CSSValue {
    int v; Direction(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { LTR = CSS_DIRECTION_LTR, RTL = CSS_DIRECTION_RTL };
#else
    enum { LTR, RTL };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case LTR: return "ltr";
            case RTL: return "rtl";
        }; return "";
    };
};

struct Display : public CSSValue {
    int v; Display(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        None     = CSS_DISPLAY_NONE,      Row      = CSS_DISPLAY_TABLE_ROW,      HeaderGroup = CSS_DISPLAY_TABLE_HEADER_GROUP,
        Inline   = CSS_DISPLAY_INLINE,    Cell     = CSS_DISPLAY_TABLE_CELL,     FooterGroup = CSS_DISPLAY_TABLE_FOOTER_GROUP,
        Block    = CSS_DISPLAY_BLOCK,     Column   = CSS_DISPLAY_TABLE_COLUMN,   InlineBlock = CSS_DISPLAY_INLINE_BLOCK,
        RunIn    = CSS_DISPLAY_RUN_IN,    Caption  = CSS_DISPLAY_TABLE_CAPTION,  InlineTable = CSS_DISPLAY_INLINE_TABLE,
        ListItem = CSS_DISPLAY_LIST_ITEM, RowGroup = CSS_DISPLAY_TABLE_ROW_GROUP,                                            
        Table    = CSS_DISPLAY_TABLE,     ColGroup = CSS_DISPLAY_TABLE_COLUMN_GROUP, 
#else
        None, Row, HeaderGroup, Inline, Cell, FooterGroup, Block, Column, InlineBlock, RunIn, Caption, InlineTable,
        ListItem, RowGroup, Table, ColGroup
#endif
    };
    virtual bool TableElement() const {
        return v == Row || v == Cell || v == Column || v == Caption || v == RowGroup || v == ColGroup || v == HeaderGroup || v == FooterGroup;
    }
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
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
};

struct EmptyCells : public CSSValue {
    int v; EmptyCells(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Show = CSS_EMPTY_CELLS_SHOW, Hide = CSS_EMPTY_CELLS_HIDE };
#else
    enum { Show, Hide };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Show: return "show";
            case Hide: return "hide";
        }; return "";
    };
};

struct Float : public CSSValue { 
    int v; Float(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Left = CSS_FLOAT_LEFT, Right = CSS_FLOAT_RIGHT, None = CSS_FLOAT_NONE }; 
#else
    enum { Left, Right, None }; 
#endif
    virtual bool Null() const { return !v || v == None; }
    virtual DOMString cssText() const {
        switch (v) {
            case Left:  return "left";
            case Right: return "right";
            case None:  return "none";
        }; return "";
    }
};

struct FontFamily : public CSSValue {
    vector<string> name; int attr;
    FontFamily(int A=0)         : CSSValue(CSS_CUSTOM), attr(A) {}
    FontFamily(const string &n) : CSSValue(CSS_CUSTOM), attr(0) { name.push_back(n); } 
    enum {
#ifdef LFL_LIBCSS
        Serif     = CSS_FONT_FAMILY_SERIF,    SansSerif = CSS_FONT_FAMILY_SANS_SERIF,
        Cursive   = CSS_FONT_FAMILY_CURSIVE,  Fantasy   = CSS_FONT_FAMILY_FANTASY,
        Monospace = CSS_FONT_FAMILY_MONOSPACE
#else
        Serif, SansSerif, Cursive, Fantasy, Monospace
#endif
    }; 
    virtual bool Null() const { return name.empty() && !attr; }
    virtual DOMString cssText() const {
        switch (attr) {
            case Serif:     return "serif";        case SansSerif: return "sans-serif";
            case Cursive:   return "cursive";      case Fantasy:   return "fantasy";
            case Monospace: return "monospace"; 
        }; return "";
    }
};

struct FontSize : public CSSNumericValue {
    int attr;
    FontSize(int A=0)                 :                        attr(A) {}
    FontSize(int V, unsigned short U) : CSSNumericValue(V, U), attr(0) {}
    enum {
#ifdef LFL_LIBCSS
        XXSmall = CSS_FONT_SIZE_XX_SMALL,    XLarge  = CSS_FONT_SIZE_X_LARGE,
        XSmall  = CSS_FONT_SIZE_X_SMALL,     XXLarge = CSS_FONT_SIZE_XX_LARGE,
        Small   = CSS_FONT_SIZE_SMALL,       Larger  = CSS_FONT_SIZE_LARGER,
        Medium  = CSS_FONT_SIZE_MEDIUM,      Smaller = CSS_FONT_SIZE_SMALLER,
        Large   = CSS_FONT_SIZE_LARGE,
#else
        XXSmall, XLarge, XSmall, XXLarge, Small, Larger, Medium, Smaller, Large   
#endif
    }; 
    virtual int getFontSizeValue(Flow *flow);
    virtual bool Null() const { return CSSNumericValue::Null() && !attr; }
    virtual DOMString cssText() const {
        if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
        switch (attr) {
            case XXSmall: return "xx-small";    case XLarge:  return "x-large";
            case XSmall:  return "x-small";     case XXLarge: return "xx-large";
            case Small:   return "small";       case Larger:  return "larger";
            case Medium:  return "medium";      case Smaller: return "smaller";
            case Large:   return "large";
        }; return "";
    }
};

struct FontStyle : public CSSValue {
    int v; FontStyle(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Normal = CSS_FONT_STYLE_NORMAL, Italic = CSS_FONT_STYLE_ITALIC, Oblique = CSS_FONT_STYLE_OBLIQUE };
#else
    enum { Normal, Italic, Oblique };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Normal:  return "normal";
            case Italic:  return "italic";
            case Oblique: return "oblique";
        }; return "";
    }
};

struct FontVariant : public CSSValue {
    int v; FontVariant(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Normal = CSS_FONT_VARIANT_NORMAL, SmallCaps = CSS_FONT_VARIANT_SMALL_CAPS };
#else
    enum { Normal, SmallCaps };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Normal:    return "normal";
            case SmallCaps: return "small-caps";
        }; return "";
    }
};

struct FontWeight : public CSSValue {
    int v; FontWeight(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum { 
#ifdef LFL_LIBCSS
        Normal  = CSS_FONT_WEIGHT_NORMAL,  _400 = CSS_FONT_WEIGHT_400,
        Bold    = CSS_FONT_WEIGHT_BOLD,    _500 = CSS_FONT_WEIGHT_500,
        Bolder  = CSS_FONT_WEIGHT_BOLDER,  _600 = CSS_FONT_WEIGHT_600,
        Lighter = CSS_FONT_WEIGHT_LIGHTER, _700 = CSS_FONT_WEIGHT_700,
        _100    = CSS_FONT_WEIGHT_100,     _800 = CSS_FONT_WEIGHT_800,
        _200    = CSS_FONT_WEIGHT_200,     _900 = CSS_FONT_WEIGHT_900,
        _300    = CSS_FONT_WEIGHT_300
#else
        Normal, Bold, Bolder, Lighter, _100, _200, _300, _400, _500, _600, _700, _800, _900    
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
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
};

struct FontFace {
    FontStyle style;
    FontFamily family;
    FontWeight weight;
    vector<string> source;
};

struct ListStylePosition : public CSSValue {
    int v; ListStylePosition(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Inside = CSS_LIST_STYLE_POSITION_INSIDE,    Outside = CSS_LIST_STYLE_POSITION_OUTSIDE
#else
        Inside, Outside
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Inside:  return "inside";
            case Outside: return "outside";
        }; return "";
    }
};

struct ListStyleType : public CSSValue {
    int v; ListStyleType(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        None       = CSS_LIST_STYLE_TYPE_NONE,           LowerLatin         = CSS_LIST_STYLE_TYPE_LOWER_LATIN,
        Disc       = CSS_LIST_STYLE_TYPE_DISC,           UpperLatin         = CSS_LIST_STYLE_TYPE_UPPER_LATIN,
        Circle     = CSS_LIST_STYLE_TYPE_CIRCLE,         Armenian           = CSS_LIST_STYLE_TYPE_ARMENIAN,
        Square     = CSS_LIST_STYLE_TYPE_SQUARE,         Georgian           = CSS_LIST_STYLE_TYPE_GEORGIAN,
        Decimal    = CSS_LIST_STYLE_TYPE_DECIMAL,        LowerAlpha         = CSS_LIST_STYLE_TYPE_LOWER_ALPHA,
        LowerRoman = CSS_LIST_STYLE_TYPE_LOWER_ROMAN,    UpperAlpha         = CSS_LIST_STYLE_TYPE_UPPER_ALPHA,
        UpperRoman = CSS_LIST_STYLE_TYPE_UPPER_ROMAN,    DecimalLeadingZero = CSS_LIST_STYLE_TYPE_DECIMAL_LEADING_ZERO,
        LowerGreek = CSS_LIST_STYLE_TYPE_LOWER_GREEK
#else
        Disc, Circle, Square, Decimal, DecimalLeadingZero, LowerRoman, UpperRoman, LowerGreek, LowerLatin, UpperLatin,
        Armenian, Georgian, LowerAlpha, UpperAlpha, None
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
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
};

struct Overflow : public CSSValue {
    int v; Overflow(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Visible = CSS_OVERFLOW_VISIBLE,    Scroll = CSS_OVERFLOW_SCROLL,
        Hidden  = CSS_OVERFLOW_HIDDEN,     Auto   = CSS_OVERFLOW_AUTO
#else
        Visible, Hidden, Scroll, Auto   
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Visible: return "visible";    case Scroll: return "scroll";
            case Hidden:  return "hidden";     case Auto:   return "auto";  
        }; return "";
    }
};

struct PageBreak : public CSSValue {
    int v; PageBreak(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Auto   = CSS_BREAK_AFTER_AUTO,      Page        = CSS_BREAK_AFTER_PAGE,
        Avoid  = CSS_BREAK_AFTER_AVOID,     Column      = CSS_BREAK_AFTER_COLUMN,
        Always = CSS_BREAK_AFTER_ALWAYS,    AvoidPage   = CSS_BREAK_AFTER_AVOID_PAGE,
        Left   = CSS_BREAK_AFTER_LEFT,      AvoidColumn = CSS_BREAK_AFTER_AVOID_COLUMN,
        Right  = CSS_BREAK_AFTER_RIGHT
#else
        Auto, Avoid, Always, Left, Right, Page, Column, AvoidPage, AvoidColumn 
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Auto:   return "auto";      case Page:        return "page";
            case Avoid:  return "avoid";     case Column:      return "column";
            case Always: return "always";    case AvoidPage:   return "avoid-page";
            case Left:   return "left";      case AvoidColumn: return "avoid-column";
            case Right:  return "right"; 
        }; return "";
    }
};

struct Position : public CSSValue {
    int v; Position(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Static = CSS_POSITION_STATIC, Relative = CSS_POSITION_RELATIVE, Absolute = CSS_POSITION_ABSOLUTE, 
        Fixed  = CSS_POSITION_FIXED
#else
        Static, Relative, Absolute, Fixed 
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Static:   return "static";      case Absolute: return "absolute";
            case Relative: return "relative";    case Fixed:    return "fixed";
        }; return "";
    }
};

struct RGBColor : public CSSValue {
    Color v; RGBColor(int hexval) : CSSValue(CSS_CUSTOM), v(hexval) {}
    int red  () const { return v.R(); }
    int green() const { return v.G(); }
    int blue () const { return v.B(); }
    virtual bool Null() const { return false; }
    virtual DOMString cssText() const { return v.HexString(); }
};

struct TableLayout : public CSSValue {
    int v; TableLayout(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Auto = CSS_TABLE_LAYOUT_AUTO, Fixed = CSS_TABLE_LAYOUT_FIXED };
#else
    enum { Auto, Fixed };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Auto:  return "auto";
            case Fixed: return "fixed";
        }; return "";
    }
};

struct TextAlign : public CSSValue {
    int v; TextAlign(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Left    = CSS_TEXT_ALIGN_LEFT,     Center  = CSS_TEXT_ALIGN_CENTER,
        Right   = CSS_TEXT_ALIGN_RIGHT,    Justify = CSS_TEXT_ALIGN_JUSTIFY,
#else
        Left, Right, Center, Justify
#endif
    };
    virtual bool Null() const { return !v || v == Left; }
    virtual DOMString cssText() const {
        switch (v) {
            case Left:  return "left";     case Center:  return "center";
            case Right: return "right";    case Justify: return "justify";
        }; return "";
    }
};

struct TextDecoration : public CSSValue {
    int v; TextDecoration(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        None        = CSS_TEXT_DECORATION_NONE,            Overline  = CSS_TEXT_DECORATION_OVERLINE,
        Blink       = CSS_TEXT_DECORATION_BLINK,           Underline = CSS_TEXT_DECORATION_UNDERLINE,
        LineThrough = CSS_TEXT_DECORATION_LINE_THROUGH,   
#else
        None=1, Blink=(1<<1), LineThrough=(1<<2), Overline=(1<<3), Underline=(1<<4)
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case None:        return "none";            case Overline:  return "overline";
            case Blink:       return "blink";           case Underline: return "underline";
            case LineThrough: return "line-through";
        }; return "";
    }
};

struct TextTransform : public CSSValue {
    int v; TextTransform(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum {
#ifdef LFL_LIBCSS
        Capitalize = CSS_TEXT_TRANSFORM_CAPITALIZE,    Lowercase = CSS_TEXT_TRANSFORM_LOWERCASE,
        Uppercase  = CSS_TEXT_TRANSFORM_UPPERCASE,     None      = CSS_TEXT_TRANSFORM_NONE,
#else
        Capitalize, Uppercase, Lowercase, None
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Capitalize: return "capitalize";    case Lowercase: return "lowercase";
            case Uppercase:  return "uppercase";     case None:      return "none";
        }; return "";
    }
};

struct UnicodeBidi : public CSSValue {
    int v; UnicodeBidi(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Normal = CSS_UNICODE_BIDI_NORMAL, Embed = CSS_UNICODE_BIDI_EMBED, Override = CSS_UNICODE_BIDI_BIDI_OVERRIDE };
#else
    enum { Normal, Embed, Override };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Normal:   return "normal";
            case Embed:    return "embed";
            case Override: return "override";
        }; return "";
    }
};

struct VerticalAlign : public CSSNumericValue {
    int attr;
    VerticalAlign(int A=0) : attr(A) {}
    VerticalAlign(int V, unsigned short U) : CSSNumericValue(V, U, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::LineHeight), attr(0) {}
    enum {
#ifdef LFL_LIBCSS
        Baseline = CSS_VERTICAL_ALIGN_BASELINE,    TextTop    = CSS_VERTICAL_ALIGN_TEXT_TOP,
        Sub      = CSS_VERTICAL_ALIGN_SUB,         Middle     = CSS_VERTICAL_ALIGN_MIDDLE,
        Super    = CSS_VERTICAL_ALIGN_SUPER,       Bottom     = CSS_VERTICAL_ALIGN_BOTTOM,
        Top      = CSS_VERTICAL_ALIGN_TOP,         TextBottom = CSS_VERTICAL_ALIGN_TEXT_BOTTOM
#else
        Baseline, Sub, Super, Top, TextTop, Middle, Bottom, TextBottom 
#endif
    };
    virtual bool Null() const { return !attr && CSSNumericValue::Null(); }
    virtual DOMString cssText() const {
        if (!CSSNumericValue::Null()) return CSSNumericValue::cssText();
        switch (attr) {
            case Baseline: return "baseline";    case TextTop:    return "text-top";
            case Sub:      return "sub";         case Middle:     return "middle";
            case Super:    return "super";       case Bottom:     return "bottom";
            case Top:      return "top";         case TextBottom: return "text-bottom";
        }; return "";
    }
};

struct Visibility : public CSSValue {
    int v; Visibility(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
#ifdef LFL_LIBCSS
    enum { Visible = CSS_VISIBILITY_VISIBLE, Hidden = CSS_VISIBILITY_HIDDEN, Collapse = CSS_VISIBILITY_COLLAPSE };
#else
    enum { Visible, Hidden, Collapse };
#endif
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Visible:  return "visible";
            case Hidden:   return "hidden";
            case Collapse: return "collapse";
        }; return "";
    }
};

struct WhiteSpace : public CSSValue {
    int v; WhiteSpace(int V=0) : CSSValue(CSS_CUSTOM), v(V) {}
    enum { 
#ifdef LFL_LIBCSS
        Normal  = CSS_WHITE_SPACE_NORMAL,    PreWrap = CSS_WHITE_SPACE_PRE_WRAP,
        Pre     = CSS_WHITE_SPACE_PRE,       PreLine = CSS_WHITE_SPACE_PRE_LINE,
        Nowrap  = CSS_WHITE_SPACE_NOWRAP
#else
        Normal, Pre, Nowrap, PreWrap, PreLine
#endif
    };
    virtual bool Null() const { return !v; }
    virtual DOMString cssText() const {
        switch (v) {
            case Normal: return "normal";    case PreWrap: return "pre-wrap";
            case Pre:    return "pre";       case PreLine: return "pre-line";
            case Nowrap: return "nowrap";
        }; return "";
    }
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

#ifdef LFL_LIBCSS
struct LibCSS_String {
    static lwc_string *Intern(const string   &s) {                             lwc_string *v=0; CHECK_EQ(lwc_intern_string(s.data(), s.size(), &v), lwc_error_ok); return v; }
    static lwc_string *Intern(const String16 &s) { string S=String::ToUTF8(s); lwc_string *v=0; CHECK_EQ(lwc_intern_string(S.data(), S.size(), &v), lwc_error_ok); return v; }
    static LFL::DOM::DOMString ToString(lwc_string *s) { return LFL::DOM::DOMString(lwc_string_data(s), lwc_string_length(s)); }
    static string ToUTF8String(lwc_string *s) { return String::ToUTF8(ToString(s)); }
};

struct LibCSS_Unit {
    static int ToPrimitive(int n) {
        switch (n) {
            case CSS_UNIT_PX:  return LFL::DOM::CSSPrimitiveValue::CSS_PX;         break;
            case CSS_UNIT_EX:  return LFL::DOM::CSSPrimitiveValue::CSS_EXS;        break;
            case CSS_UNIT_EM:  return LFL::DOM::CSSPrimitiveValue::CSS_EMS;        break;
            case CSS_UNIT_IN:  return LFL::DOM::CSSPrimitiveValue::CSS_IN;         break;
            case CSS_UNIT_CM:  return LFL::DOM::CSSPrimitiveValue::CSS_CM;         break;
            case CSS_UNIT_MM:  return LFL::DOM::CSSPrimitiveValue::CSS_MM;         break;
            case CSS_UNIT_PT:  return LFL::DOM::CSSPrimitiveValue::CSS_PT;         break;
            case CSS_UNIT_PC:  return LFL::DOM::CSSPrimitiveValue::CSS_PC;         break;
            case CSS_UNIT_PCT: return LFL::DOM::CSSPrimitiveValue::CSS_PERCENTAGE; break;
        } return 0;
    }
};

struct StyleSheet : public LFL::DOM::Object {
    LFL::DOM::Document *ownerDocument;
    css_stylesheet *sheet;
    css_stylesheet_params params;
    string content_url, character_set;
    ~StyleSheet() { CHECK_EQ(css_stylesheet_destroy(sheet), CSS_OK); }
    StyleSheet(LFL::DOM::Document *D, const char *U=0, const char *CS=0, bool in_line=0, bool quirks=0, const char *Content=0) :
        ownerDocument(D), sheet(0), content_url(BlankNull(U)), character_set(BlankNull(CS)) {
        params.params_version = CSS_STYLESHEET_PARAMS_VERSION_1;
        params.level = CSS_LEVEL_DEFAULT;
        params.title = NULL;
        params.url = content_url.c_str();
        params.charset = !character_set.empty() ? character_set.c_str() : 0;
        params.inline_style = in_line;
        params.allow_quirks = quirks;
        params.resolve = Resolve;
        params.import = Import;
        params.color = Color;
        params.font = Font;
        params.resolve_pw = params.import_pw = params.color_pw = params.font_pw = this;
        CHECK_EQ(css_stylesheet_create(&params, &sheet), CSS_OK);
        if (Content) { Parse(Content); Done(); }
    }
    void Parse(File *f) { for (const char *line = f->NextLine(); line; line = f->NextLine()) Parse(line, f->nr.record_len); }
    void Parse(const string &content) { return Parse(content.c_str(), content.size()); }
    void Parse(const char *content, int content_len) {
        css_error code = css_stylesheet_append_data(sheet, (const unsigned char *)content, content_len);
        CHECK(code == CSS_OK || code == CSS_NEEDDATA);
    }
    void Done() { CHECK_EQ(css_stylesheet_data_done(sheet), CSS_OK); }
    static StyleSheet *Default() { static StyleSheet ret(0, "", "UTF-8", 0, 0, LFL::CSS::Default); return &ret; }
    static css_error Resolve(void *pw, const char *base, lwc_string *rel, lwc_string **abs) { *abs = lwc_string_ref(rel); return CSS_OK; }
    static css_error Import(void *pw, css_stylesheet *parent, lwc_string *url, uint64_t media);
    static css_error Color(void *pw, lwc_string *name, css_color *color) {
        // INFO("libcss Color ", LibCSS_String::ToString(name)); // none, medium, ridge, solid, thin, threedface, no-repeat, center
        return CSS_INVALID; // CSS_OK;
    }
    static css_error Font(void *pw, lwc_string *name, css_system_font *system_font) {
        INFO("libcss Font ", LibCSS_String::ToString(name));
        return CSS_INVALID; // CSS_OK;
    }
};

struct ComputedStyle : public LFL::DOM::CSSStyleDeclaration {
    LFL::DOM::Node *node;
    css_select_results *style;
    bool is_root, font_not_inherited, bgcolor_not_inherited;
    virtual ~ComputedStyle() { Reset(); }
    ComputedStyle(LFL::DOM::Node *N) : node(N), style(0), is_root(0), font_not_inherited(0), bgcolor_not_inherited(0) {}

    bool Computed() { return style; }
    void Reset() { if (style) CHECK_EQ(css_select_results_destroy(style), CSS_OK); style = 0; }
    css_computed_style *Style() { return style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0; }

    LFL::DOM::Display              Display             () { return LFL::DOM::Display(css_computed_display(Style(), is_root)); }
    LFL::DOM::Clear                Clear               () { return LFL::DOM::Clear(css_computed_clear(Style())); }
    LFL::DOM::Float                Float               () { return LFL::DOM::Float(css_computed_float(Style())); }
    LFL::DOM::Position             Position            () { return LFL::DOM::Position(css_computed_position(Style())); }
    LFL::DOM::Overflow             Overflow            () { return LFL::DOM::Overflow(css_computed_overflow(Style())); }
    LFL::DOM::TextAlign            TextAlign           () { return LFL::DOM::TextAlign(css_computed_text_align(Style())); }
    LFL::DOM::Direction            Direction           () { return LFL::DOM::Direction(css_computed_direction(Style())); }
    LFL::DOM::FontStyle            FontStyle           () { return LFL::DOM::FontStyle(css_computed_font_style(Style())); }
    LFL::DOM::EmptyCells           EmptyCells          () { return LFL::DOM::EmptyCells(css_computed_empty_cells(Style())); }
    LFL::DOM::FontWeight           FontWeight          () { return LFL::DOM::FontWeight(css_computed_font_weight(Style())); }
    LFL::DOM::Visibility           Visibility          () { return LFL::DOM::Visibility(css_computed_visibility(Style())); }
    LFL::DOM::WhiteSpace           WhiteSpace          () { return LFL::DOM::WhiteSpace(css_computed_white_space(Style())); }
    LFL::DOM::CaptionSide          CaptionSide         () { return LFL::DOM::CaptionSide(css_computed_caption_side(Style())); }
    LFL::DOM::FontVariant          FontVariant         () { return LFL::DOM::FontVariant(css_computed_font_variant(Style())); }
    LFL::DOM::TableLayout          TableLayout         () { return LFL::DOM::TableLayout(css_computed_table_layout(Style())); }
    LFL::DOM::UnicodeBidi          UnicodeBidi         () { return LFL::DOM::UnicodeBidi(css_computed_unicode_bidi(Style())); }
    LFL::DOM::BorderStyle          OutlineStyle        () { return LFL::DOM::BorderStyle(css_computed_outline_style(Style())); }
    LFL::DOM::TextTransform        TextTransform       () { return LFL::DOM::TextTransform(css_computed_text_transform(Style())); }
    LFL::DOM::TextDecoration       TextDecoration      () { return LFL::DOM::TextDecoration(css_computed_text_decoration(Style())); }
    LFL::DOM::BorderCollapse       BorderCollapse      () { return LFL::DOM::BorderCollapse(css_computed_border_collapse(Style())); }
    LFL::DOM::ListStyleType        ListStyleType       () { return LFL::DOM::ListStyleType(css_computed_list_style_type(Style())); }
    LFL::DOM::ListStylePosition    ListStylePosition   () { return LFL::DOM::ListStylePosition(css_computed_list_style_position(Style())); }
    LFL::DOM::PageBreak            PageBreakAfter      () { return LFL::DOM::PageBreak(css_computed_page_break_after(Style())); }
    LFL::DOM::PageBreak            PageBreakBefore     () { return LFL::DOM::PageBreak(css_computed_page_break_before(Style())); }
    LFL::DOM::PageBreak            PageBreakInside     () { return LFL::DOM::PageBreak(css_computed_page_break_inside(Style())); }
    LFL::DOM::BorderStyle          BorderTopStyle      () { return LFL::DOM::BorderStyle(css_computed_border_top_style(Style())); }
    LFL::DOM::BorderStyle          BorderLeftStyle     () { return LFL::DOM::BorderStyle(css_computed_border_left_style(Style())); }
    LFL::DOM::BorderStyle          BorderRightStyle    () { return LFL::DOM::BorderStyle(css_computed_border_right_style(Style())); }
    LFL::DOM::BorderStyle          BorderBottomStyle   () { return LFL::DOM::BorderStyle(css_computed_border_bottom_style(Style())); }
    LFL::DOM::BackgroundRepeat     BackgroundRepeat    () { return LFL::DOM::BackgroundRepeat(css_computed_background_repeat(Style())); }
    LFL::DOM::BackgroundAttachment BackgroundAttachment() { return LFL::DOM::BackgroundAttachment(css_computed_background_attachment(Style())); }
    LFL::DOM::RGBColor             BackgroundColor     () { css_color c; uint8_t t = css_computed_background_color   (Style(), &c); CHECK_EQ(t, CSS_BACKGROUND_COLOR_COLOR); return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             BorderBottomColor   () { css_color c; uint8_t t = css_computed_border_bottom_color(Style(), &c); CHECK_EQ(t, CSS_BORDER_COLOR_COLOR);     return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             BorderLeftColor     () { css_color c; uint8_t t = css_computed_border_left_color  (Style(), &c); CHECK_EQ(t, CSS_BORDER_COLOR_COLOR);     return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             BorderRightColor    () { css_color c; uint8_t t = css_computed_border_right_color (Style(), &c); CHECK_EQ(t, CSS_BORDER_COLOR_COLOR);     return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             BorderTopColor      () { css_color c; uint8_t t = css_computed_border_top_color   (Style(), &c); CHECK_EQ(t, CSS_BORDER_COLOR_COLOR);     return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             OutlineColor        () { css_color c; uint8_t t = css_computed_outline_color      (Style(), &c); /*CHECK_EQ(t, CSS_OUTLINE_COLOR_COLOR);*/ return LFL::DOM::RGBColor(c); }
    LFL::DOM::RGBColor             Color               () { css_color c; uint8_t t = css_computed_color              (Style(), &c); CHECK_EQ(t, CSS_COLOR_COLOR);            return LFL::DOM::RGBColor(c); }
    LFL::DOM::Cursor               Cursor              () { lwc_string **n=0; LFL::DOM::Cursor     ret(css_computed_cursor     (Style(), &n)); for (; n && *n; n++) ret.url .push_back(LibCSS_String::ToUTF8String(*n)); return ret; }
    LFL::DOM::FontFamily           FontFamily          () { lwc_string **n=0; LFL::DOM::FontFamily ret(css_computed_font_family(Style(), &n)); for (; n && *n; n++) ret.name.push_back(LibCSS_String::ToUTF8String(*n)); return ret; }
    LFL::DOM::CSSStringValue       BackgroundImage     () { lwc_string *v=0; css_computed_background_image (Style(), &v); return v ? LFL::DOM::CSSStringValue(LibCSS_String::ToString(v)) : LFL::DOM::CSSStringValue(); }
    LFL::DOM::CSSNumericValuePair  BackgroundPosition  () { css_fixed x,y; css_unit xu,yu; return css_computed_background_position(Style(), &x, &xu, &y, &yu) == CSS_BACKGROUND_POSITION_SET ? LFL::DOM::CSSNumericValuePair(FIXTOFLT(x), LibCSS_Unit::ToPrimitive(xu), 0, FIXTOFLT(y), LibCSS_Unit::ToPrimitive(yu), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight) : LFL::DOM::CSSNumericValuePair(); }
    LFL::DOM::Rect                 Clip                () { css_computed_clip_rect r;      return css_computed_clip             (Style(), &r)      == CSS_CLIP_RECT       ? LFL::DOM::Rect(r.top_auto, FIXTOFLT(r.top), LibCSS_Unit::ToPrimitive(r.tunit), r.right_auto, FIXTOFLT(r.right), LibCSS_Unit::ToPrimitive(r.runit), r.bottom_auto, FIXTOFLT(r.bottom), LibCSS_Unit::ToPrimitive(r.bunit), r.left_auto, FIXTOFLT(r.left), LibCSS_Unit::ToPrimitive(r.lunit)) : LFL::DOM::Rect(); }
    LFL::DOM::CSSNumericValue      Opacity             () { css_fixed s;                   return css_computed_opacity          (Style(), &s)      == CSS_OPACITY_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      Orphans             () { css_fixed s;                   return css_computed_orphans          (Style(), &s)      == CSS_ORPHANS_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      Widows              () { css_fixed s;                   return css_computed_widows           (Style(), &s)      == CSS_WIDOWS_SET      ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      MinHeight           () { css_fixed s; css_unit su;      return css_computed_min_height       (Style(), &s, &su) == CSS_MIN_HEIGHT_SET  ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      MinWidth            () { css_fixed s; css_unit su;      return css_computed_min_width        (Style(), &s, &su) == CSS_MIN_WIDTH_SET   ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      PaddingTop          () { css_fixed s; css_unit su;      return css_computed_padding_top      (Style(), &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      PaddingRight        () { css_fixed s; css_unit su;      return css_computed_padding_right    (Style(), &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      PaddingBottom       () { css_fixed s; css_unit su;      return css_computed_padding_bottom   (Style(), &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      PaddingLeft         () { css_fixed s; css_unit su;      return css_computed_padding_left     (Style(), &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      TextIndent          () { css_fixed s; css_unit su;      return css_computed_text_indent      (Style(), &s, &su) == CSS_TEXT_INDENT_SET ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSAutoNumericValue  Top                 () { css_fixed s; css_unit su; int ret = css_computed_top                (Style(), &s, &su); return ret == CSS_TOP_SET             ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_TOP_AUTO); }
    LFL::DOM::CSSAutoNumericValue  Bottom              () { css_fixed s; css_unit su; int ret = css_computed_bottom             (Style(), &s, &su); return ret == CSS_BOTTOM_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_BOTTOM_AUTO); }
    LFL::DOM::CSSAutoNumericValue  Left                () { css_fixed s; css_unit su; int ret = css_computed_left               (Style(), &s, &su); return ret == CSS_LEFT_SET            ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_LEFT_AUTO); }
    LFL::DOM::CSSAutoNumericValue  Right               () { css_fixed s; css_unit su; int ret = css_computed_right              (Style(), &s, &su); return ret == CSS_RIGHT_SET           ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_RIGHT_AUTO); }
    LFL::DOM::CSSAutoNumericValue  Width               () { css_fixed s; css_unit su; int ret = css_computed_width              (Style(), &s, &su); return ret == CSS_WIDTH_SET           ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_WIDTH_AUTO); }
    LFL::DOM::CSSAutoNumericValue  Height              () { css_fixed s; css_unit su; int ret = css_computed_height             (Style(), &s, &su); return ret == CSS_HEIGHT_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_HEIGHT_AUTO); }
    LFL::DOM::FontSize             FontSize            () { css_fixed s; css_unit su; int ret = css_computed_font_size          (Style(), &s, &su); return ret == CSS_FONT_SIZE_DIMENSION ? LFL::DOM::FontSize           (FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::FontSize(ret); }
    LFL::DOM::VerticalAlign        VerticalAlign       () { css_fixed s; css_unit su; int ret = css_computed_vertical_align     (Style(), &s, &su); return ret == CSS_VERTICAL_ALIGN_SET  ? LFL::DOM::VerticalAlign      (FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::VerticalAlign(ret); }
    LFL::DOM::BorderWidth          BorderTopWidth      () { css_fixed w; css_unit wu; int ret = css_computed_border_top_width   (Style(), &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
    LFL::DOM::BorderWidth          BorderRightWidth    () { css_fixed w; css_unit wu; int ret = css_computed_border_right_width (Style(), &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
    LFL::DOM::BorderWidth          BorderBottomWidth   () { css_fixed w; css_unit wu; int ret = css_computed_border_bottom_width(Style(), &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
    LFL::DOM::BorderWidth          BorderLeftWidth     () { css_fixed w; css_unit wu; int ret = css_computed_border_left_width  (Style(), &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
    LFL::DOM::BorderWidth          OutlineWidth        () { css_fixed w; css_unit wu; int ret = css_computed_outline_width      (Style(), &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
    LFL::DOM::CSSAutoNumericValue  MarginBottom        () { css_fixed s; css_unit su; int ret = css_computed_margin_bottom      (Style(), &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
    LFL::DOM::CSSAutoNumericValue  MarginLeft          () { css_fixed s; css_unit su; int ret = css_computed_margin_left        (Style(), &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
    LFL::DOM::CSSAutoNumericValue  MarginRight         () { css_fixed s; css_unit su; int ret = css_computed_margin_right       (Style(), &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
    LFL::DOM::CSSAutoNumericValue  MarginTop           () { css_fixed s; css_unit su; int ret = css_computed_margin_top         (Style(), &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
    LFL::DOM::CSSNoneNumericValue  MaxHeight           () { css_fixed s; css_unit su; int ret = css_computed_max_height         (Style(), &s, &su); return ret == CSS_MAX_HEIGHT_SET      ? LFL::DOM::CSSNoneNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNoneNumericValue(ret == CSS_MAX_HEIGHT_NONE); }
    LFL::DOM::CSSNoneNumericValue  MaxWidth            () { css_fixed s; css_unit su; int ret = css_computed_max_width          (Style(), &s, &su); return ret == CSS_MAX_WIDTH_SET       ? LFL::DOM::CSSNoneNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNoneNumericValue(ret == CSS_MAX_WIDTH_NONE); }
    LFL::DOM::CSSAutoNumericValue  ZIndex              () { css_fixed s;              int ret = css_computed_z_index            (Style(), &s);      return ret == CSS_Z_INDEX_SET         ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSAutoNumericValue(ret == CSS_Z_INDEX_AUTO); }
    LFL::DOM::CSSNormNumericValue  WordSpacing         () { css_fixed s; css_unit su; int ret = css_computed_word_spacing       (Style(), &s, &su); return ret == CSS_WORD_SPACING_SET    ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth) : LFL::DOM::CSSNormNumericValue(ret == CSS_WORD_SPACING_NORMAL,   LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
    LFL::DOM::CSSNormNumericValue  LetterSpacing       () { css_fixed s; css_unit su; int ret = css_computed_letter_spacing     (Style(), &s, &su); return ret == CSS_LETTER_SPACING_SET  ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth) : LFL::DOM::CSSNormNumericValue(ret == CSS_LETTER_SPACING_NORMAL, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
    LFL::DOM::CSSNormNumericValue  LineHeight          () { css_fixed s; css_unit su; int ret = css_computed_line_height        (Style(), &s, &su); bool dim = ret == CSS_LINE_HEIGHT_DIMENSION; return (ret == CSS_LINE_HEIGHT_NUMBER || dim) ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), dim ? LibCSS_Unit::ToPrimitive(su) : LFL::DOM::CSSPrimitiveValue::CSS_NUMBER, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight) : LFL::DOM::CSSNormNumericValue(ret == CSS_LINE_HEIGHT_NORMAL, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight); }

    virtual LFL::DOM::CSSValue *getPropertyCSSValue(const LFL::DOM::DOMString &n) { return 0; };
    virtual LFL::DOM::DOMString getPropertyPriority(const LFL::DOM::DOMString &n) { return ""; }
    virtual LFL::DOM::DOMString removeProperty(const LFL::DOM::DOMString &n)      { ERROR("not implemented"); return ""; }
    virtual void                setProperty(const LFL::DOM::DOMString &n, const LFL::DOM::DOMString &v, const LFL::DOM::DOMString &priority) { ERROR("not impelemented"); }
    virtual int                 length() { return Singleton<LFL::CSS::PropertyNames>::Get()->size(); };
    virtual LFL::DOM::DOMString item(int index) { CHECK_RANGE(index, 0, length()); return (*Singleton<LFL::CSS::PropertyNames>::Get())[index]; }
    virtual LFL::DOM::DOMString getPropertyValue(const LFL::DOM::DOMString &n) {
        string name = String::ToUTF8(n);
        if (0) {}
#undef  XX
#define XX(x) else if (name == #x)
#undef  YY
#define YY(y)     return y().cssText();
#include "crawler/css_properties.h"
        return "";
    }
};

struct StyleContext : public LFL::DOM::Object {
    LFL::DOM::HTMLDocument *doc;
    css_select_ctx *select_ctx;
    virtual ~StyleContext() { CHECK_EQ(css_select_ctx_destroy(select_ctx), CSS_OK); }
    StyleContext(LFL::DOM::HTMLDocument *D) : doc(D), select_ctx(0) { CHECK_EQ(css_select_ctx_create(&select_ctx), CSS_OK); }
    void AppendSheet(StyleSheet *ss) {
        CHECK_EQ(css_select_ctx_append_sheet(select_ctx, ss->sheet, CSS_ORIGIN_AUTHOR, CSS_MEDIA_ALL), CSS_OK);
    }
    void Match(ComputedStyle *out, LFL::DOM::Node *node, ComputedStyle *parent_sheet=0, const char *inline_style=0) {
        out->Reset();

        string html4_style = node->HTML4Style();
        if (!html4_style.empty() && inline_style) html4_style.append(inline_style);
        const char *concat_style = html4_style.empty() ? inline_style : html4_style.c_str();
        if (concat_style) {
            StyleSheet inline_sheet(doc, "", "UTF-8", true, false, concat_style);
            CHECK_EQ(css_select_style(select_ctx, node, CSS_MEDIA_SCREEN, inline_sheet.sheet, SelectHandler(), this, &out->style), CSS_OK);
        } else {
            CHECK_EQ(css_select_style(select_ctx, node, CSS_MEDIA_SCREEN, 0,                  SelectHandler(), this, &out->style), CSS_OK);
        }

        lwc_string **n; css_fixed s; css_unit su; css_color c;
        out->font_not_inherited =
            (css_computed_font_family (out->Style(), &n)      != CSS_FONT_FAMILY_INHERIT) ||
            (css_computed_font_size   (out->Style(), &s, &su) != CSS_FONT_SIZE_INHERIT)   ||
            (css_computed_font_style  (out->Style())          != CSS_FONT_STYLE_INHERIT)  ||
            (css_computed_font_weight (out->Style())          != CSS_FONT_WEIGHT_INHERIT) ||
            (css_computed_font_variant(out->Style())          != CSS_FONT_VARIANT_INHERIT);

        out->bgcolor_not_inherited = css_computed_background_color(out->Style(), &c) == CSS_BACKGROUND_COLOR_COLOR;

        if (parent_sheet) {
            CHECK_EQ(css_computed_style_compose(parent_sheet->Style(), out->Style(), compute_font_size, NULL, out->Style()), CSS_OK);
        }
    }
    int FontFaces(const string &face_name, vector<LFL::DOM::FontFace> *out) {
        out->clear();
        lwc_string *n=0; css_select_font_faces_results *v=0; uint32_t l;
        int rc = css_select_font_faces(select_ctx, CSS_MEDIA_SCREEN, LibCSS_String::Intern(face_name), &v);
        for (int i=0; rc == CSS_OK && v && i<v->n_font_faces; i++) {
            LFL::DOM::FontFace ff;
            ff.style  = LFL::DOM::FontStyle (css_font_face_font_style( v->font_faces[i]));
            ff.weight = LFL::DOM::FontWeight(css_font_face_font_weight(v->font_faces[i]));
            n=0; css_font_face_get_font_family(v->font_faces[i], &n); if (n) ff.family.name.push_back(LibCSS_String::ToUTF8String(n));
            l=0; css_font_face_count_srcs(v->font_faces[i], &l);
            for (int j=0; j<l; j++) {
                const css_font_face_src *src=0; css_font_face_get_src(v->font_faces[i], j, &src);
                if (css_font_face_src_location_type(src) == CSS_FONT_FACE_LOCATION_TYPE_UNSPECIFIED) continue;
                n=0; if (src) css_font_face_src_get_location(src, &n);
                if (n) ff.source.push_back(LibCSS_String::ToUTF8String(n));
            }
            out->push_back(ff);
        }
        if (v) css_select_font_faces_results_destroy(v);
        return out->size();
    }
    static css_select_handler *SelectHandler() {
        static css_select_handler select_handler = {
            CSS_SELECT_HANDLER_VERSION_1, node_name, node_classes, node_id, named_ancestor_node, named_parent_node, named_sibling_node,
            named_generic_sibling_node, parent_node, sibling_node, node_has_name, node_has_class, node_has_id, node_has_attribute,
            node_has_attribute_equal, node_has_attribute_dashmatch, node_has_attribute_includes, node_has_attribute_prefix,
            node_has_attribute_suffix, node_has_attribute_substring, node_is_root, node_count_siblings, node_is_empty, node_is_link,
            node_is_visited, node_is_hover, node_is_active, node_is_focus, node_is_enabled, node_is_disabled, node_is_checked,
            node_is_target, node_is_lang, node_presentational_hint, ua_default_for_property, compute_font_size, set_libcss_node_data,
            get_libcss_node_data
        };
        return &select_handler;
    }
    static css_error node_name(void *pw, void *n, css_qname *qname) {
        qname->name = LibCSS_String::Intern(((LFL::DOM::Node*)n)->nodeName());
        return CSS_OK;
    }
    static css_error node_classes(void *pw, void *n, lwc_string ***classes_out, uint32_t *n_classes) {
        *classes_out = NULL; *n_classes = 0;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if (!(attr = node->getAttributeNode("class"))) return CSS_OK;

        vector<LFL::DOM::DOMString> classes;
        Split(attr->nodeValue(), isspace, &classes);
        if (!classes.size()) return CSS_OK;

        *n_classes = classes.size();
        *classes_out = (lwc_string**)malloc(classes.size() * sizeof(lwc_string*));
        for (int i=0; i<classes.size(); i++) (*classes_out)[i] = LibCSS_String::Intern(classes[i]);
        return CSS_OK;
    }
    static css_error node_id(void *pw, void *n, lwc_string **id) {
        *id = NULL;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode("id"))) *id = LibCSS_String::Intern(attr->nodeValue());
        return CSS_OK;
    }
    static css_error named_ancestor_node(void *pw, void *n, const css_qname *qname, void **ancestor) {
        *ancestor = NULL;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        LFL::DOM::DOMString query = LibCSS_String::ToString(qname->name);
        for (node = node->parentNode; node; node = node->parentNode) {
            if (node->nodeName() == query && node->AsElement()) { *ancestor = node; break; }
        }
        return CSS_OK;
    }
    static css_error named_parent_node(void *pw, void *n, const css_qname *qname, void **parent) {
        *parent = NULL;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        for (node = node->parentNode; node && !node->AsElement(); node = node->parentNode) {}
        if (node && node->nodeName() == LibCSS_String::ToString(qname->name)) *parent = node;
        return CSS_OK;
    }
    static css_error named_sibling_node(void *pw, void *n, const css_qname *qname, void **sibling) {
        *sibling = NULL;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        for (node = node->previousSibling(); node && !node->AsElement(); node = node->previousSibling()) {}
        if (node && node->nodeName() == LibCSS_String::ToString(qname->name)) *sibling = node;
        return CSS_OK;
    }
    static css_error named_generic_sibling_node(void *pw, void *n, const css_qname *qname, void **sibling) {
        *sibling = NULL;
        LFL::DOM::DOMString query = LibCSS_String::ToString(qname->name);
        for (LFL::DOM::Node *i = ((LFL::DOM::Node*)n)->previousSibling(); i; i = i->previousSibling()) {
            if (i->nodeName() == query && i->AsElement()) { *sibling = i; break; }
        }
        return CSS_OK;
    }
    static css_error parent_node(void *pw, void *n, void **parent) {
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        for (node = node->parentNode; node && !node->AsElement(); node = node->parentNode) {}
        *parent = node;
        return CSS_OK;
    }
    static css_error sibling_node(void *pw, void *n, void **sibling) {
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        for (node = node->previousSibling(); node && !node->AsElement(); node = node->previousSibling()) {}
        *sibling = node;
        return CSS_OK;
    }
    static css_error node_has_name(void *pw, void *n, const css_qname *qname, bool *match) {
        *match = LibCSS_String::ToString(qname->name) == ((LFL::DOM::Node*)n)->nodeName();
        return CSS_OK;
    }
    static css_error node_has_class(void *pw, void *n, lwc_string *name, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if (!(attr = node->getAttributeNode("class"))) return CSS_OK;

        vector<LFL::DOM::DOMString> classes;
        Split(attr->nodeValue(), isspace, &classes);
        if (!classes.size()) return CSS_OK;

        LFL::DOM::DOMString query = LibCSS_String::ToString(name);
        for (int i=0; i<classes.size(); i++) if (classes[i] == query) { *match = true; break; }
        return CSS_OK;
    }
    static css_error node_has_id(void *pw, void *n, lwc_string *name, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode("id"))) *match = attr->nodeValue() == LibCSS_String::ToString(name);
        return CSS_OK;
    }
    static css_error node_has_attribute(void *pw, void *n, const css_qname *qname, bool *match) {
        *match = ((LFL::DOM::Node*)n)->getAttributeNode(LibCSS_String::ToString(qname->name));
        return CSS_OK;
    }
    static css_error node_has_attribute_equal(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
            *match = attr->nodeValue() == LibCSS_String::ToString(value);
        return CSS_OK;
    }
    static css_error node_has_attribute_dashmatch(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name)))) {
            LFL::DOM::DOMString attr_val = attr->nodeValue(), seek_val = LibCSS_String::ToString(value);
            *match = (seek_val == attr_val) ||
                     (seek_val == attr_val.substr(0, seek_val.size()) && attr_val[seek_val.size()-1] == '-');
        } 
        return CSS_OK;
    }
    static css_error node_has_attribute_includes(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name)))) {
            LFL::DOM::DOMString attr_val = attr->nodeValue(), seek_val = LibCSS_String::ToString(value);
            *match = seek_val == attr_val.substr(0, attr_val.find(' '));
        } 
        return CSS_OK;
    }
    static css_error node_has_attribute_prefix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
            *match = PrefixMatch(attr->nodeValue(), LibCSS_String::ToString(value));
        return CSS_OK;
    }
    static css_error node_has_attribute_suffix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
            *match = SuffixMatch(attr->nodeValue(), LibCSS_String::ToString(value));
        return CSS_OK;
    }
    static css_error node_has_attribute_substring(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
        *match = false;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n, *attr = 0;
        if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
            *match = attr->nodeValue().find(LibCSS_String::ToString(value)) != string::npos;
        return CSS_OK;
    }
    static css_error node_is_root(void *pw, void *n, bool *match) {
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        *match = !node->parentNode || node->parentNode->AsDocument();
        return CSS_OK;
    }
    static css_error node_count_siblings(void *pw, void *n, bool same_name, bool after, int32_t *count) {
        *count = 0;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        LFL::DOM::DOMString query = same_name ? node->nodeName() : "";
        while ((node = after ? node->nextSibling() : node->previousSibling())) {
            if (!node->AsElement() || (same_name && node->nodeName() != query)) continue;
            (*count)++;
        }
        return CSS_OK;
    }
    static css_error node_is_empty(void *pw, void *n, bool *match) {
        *match = true;
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        for (int i = 0; i < node->childNodes.length(); i++)
            if (node->childNodes.item(i)->AsElement() || node->childNodes.item(i)->AsText())
            { *match = false; break; }
        return CSS_OK;
    }
    static css_error node_is_link(void *pw, void *n, bool *match) {
        LFL::DOM::Node *node = (LFL::DOM::Node*)n;
        *match = node->htmlElementType == LFL::DOM::HTML_ANCHOR_ELEMENT && node->getAttributeNode("href");
        return CSS_OK;
    }
    static css_error node_is_visited(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_hover(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_active(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_focus(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_enabled(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_disabled(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_checked(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_target(void *pw, void *n, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_is_lang(void *pw, void *n, lwc_string *lang, bool *match) {
        *match = false; return CSS_OK;
    }
    static css_error node_presentational_hint(void *pw, void *node, uint32_t property, css_hint *hint) {
        return CSS_PROPERTY_NOT_SET;
    }
    static css_error ua_default_for_property(void *pw, uint32_t property, css_hint *hint) {
        if      (property == CSS_PROP_COLOR)        { hint->data.color = 0x00000000; hint->status = CSS_COLOR_COLOR; }
        else if (property == CSS_PROP_FONT_FAMILY)  { hint->data.strings = NULL;     hint->status = CSS_FONT_FAMILY_SANS_SERIF; }
        else if (property == CSS_PROP_QUOTES)       { hint->data.strings = NULL;     hint->status = CSS_QUOTES_NONE; }
        else if (property == CSS_PROP_VOICE_FAMILY) { hint->data.strings = NULL;     hint->status = 0; }
        else return CSS_INVALID;
        return CSS_OK;
    }
    static css_error compute_font_size(void *pw, const css_hint *parent, css_hint *size) {
        static css_hint_length sizes[] = {
            { FLTTOFIX(6.75), CSS_UNIT_PT },
            { FLTTOFIX(7.50), CSS_UNIT_PT },
            { FLTTOFIX(9.75), CSS_UNIT_PT },
            { FLTTOFIX(12.0), CSS_UNIT_PT },
            { FLTTOFIX(13.5), CSS_UNIT_PT },
            { FLTTOFIX(18.0), CSS_UNIT_PT },
            { FLTTOFIX(24.0), CSS_UNIT_PT }
        };
        const css_hint_length *parent_size;
        /* Grab parent size, defaulting to medium if none */
        if (parent == NULL) {
            parent_size = &sizes[CSS_FONT_SIZE_MEDIUM - 1];
        } else {
            CHECK_EQ(parent->status, CSS_FONT_SIZE_DIMENSION);
            CHECK_NE(parent->data.length.unit, CSS_UNIT_EM);
            CHECK_NE(parent->data.length.unit, CSS_UNIT_EX);
            parent_size = &parent->data.length;
        }
        CHECK_NE(size->status, CSS_FONT_SIZE_INHERIT);
        if (size->status < CSS_FONT_SIZE_LARGER) {
            /* Keyword -- simple */
            size->data.length = sizes[size->status - 1];
        } else if (size->status == CSS_FONT_SIZE_LARGER) {
            /** \todo Step within table, if appropriate */
            size->data.length.value = FMUL(parent_size->value, FLTTOFIX(1.2));
            size->data.length.unit = parent_size->unit;
        } else if (size->status == CSS_FONT_SIZE_SMALLER) {
            /** \todo Step within table, if appropriate */
            size->data.length.value = FMUL(parent_size->value, FLTTOFIX(1.2));
            size->data.length.unit = parent_size->unit;
        } else if (size->data.length.unit == CSS_UNIT_EM || size->data.length.unit == CSS_UNIT_EX) {
            size->data.length.value = FMUL(size->data.length.value, parent_size->value);
            if (size->data.length.unit == CSS_UNIT_EX) size->data.length.value = FMUL(size->data.length.value, FLTTOFIX(0.6));
            size->data.length.unit = parent_size->unit;
        } else if (size->data.length.unit == CSS_UNIT_PCT) {
            size->data.length.value = FDIV(FMUL(size->data.length.value, parent_size->value), FLTTOFIX(100));
            size->data.length.unit  = parent_size->unit;
        }
        size->status = CSS_FONT_SIZE_DIMENSION;
        return CSS_OK;
    }
    static css_error set_libcss_node_data(void *pw, void *n, void *libcss_node_data) {
        /* Since we're not storing it, ensure node data gets deleted */
        css_libcss_node_data_handler(SelectHandler(), CSS_NODE_DELETED, pw, n, NULL, libcss_node_data);
        return CSS_OK;
    }
    static css_error get_libcss_node_data(void *pw, void *n, void **libcss_node_data) {
        *libcss_node_data = NULL;
        return CSS_OK;
    }
};
#else /* LFL_LIBCSS */
struct StyleSheet : public LFL::DOM::Object {
    LFL::DOM::Document *ownerDocument;
    static StyleSheet *Default() { static StyleSheet ret(0); return &ret; }
    StyleSheet(LFL::DOM::Document *D, const char *U=0, const char *T=0, bool in_line=0, bool quirks=0, const char *Content=0) : ownerDocument(D) {}
    void Parse(File *f) { for (const char *line = f->NextLine(); line; line = f->NextLine()) Parse(line, f->nr.record_len); }
    void Parse(const string &content) { return Parse(content.c_str(), content.size()); }
    void Parse(const char *content, int content_len) {}
    void Done()                                      {}
};
struct ComputedStyle : public LFL::DOM::CSSStyleDeclaration {
    LFL::DOM::Node *node;
    bool is_root, font_not_inherited, bgcolor_not_inherited;
    ComputedStyle(LFL::DOM::Node *N) : node(N), is_root(0), font_not_inherited(0), bgcolor_not_inherited(0) {}
    bool IsLink() const {
        for (LFL::DOM::Node *n = node ? node->parentNode : 0; n; n = n->parentNode)
            if (n->htmlElementType == LFL::DOM::HTML_ANCHOR_ELEMENT) return true;
        return false;
    }

    virtual int                  length()        { return  0; }
    virtual LFL::DOM::DOMString  item(int index) { return ""; }
    virtual LFL::DOM::DOMString  getPropertyValue   (const LFL::DOM::DOMString &n) { return ""; }
    virtual LFL::DOM::CSSValue  *getPropertyCSSValue(const LFL::DOM::DOMString &n) { return  0; }
    virtual LFL::DOM::DOMString  getPropertyPriority(const LFL::DOM::DOMString &n) { return ""; }
    virtual LFL::DOM::DOMString  removeProperty     (const LFL::DOM::DOMString &n) { ERROR("not implemented"); return ""; }
    virtual void                 setProperty(const LFL::DOM::DOMString &n, const LFL::DOM::DOMString &v, const LFL::DOM::DOMString &priority) { ERROR("not implemented"); }

    LFL::DOM::Display Display() {
        switch (node->htmlElementType) {
            case LFL::DOM::HTML_BLOCK_QUOTE_ELEMENT:    case LFL::DOM::HTML_HR_ELEMENT:
            case LFL::DOM::HTML_CANVAS_ELEMENT:         case LFL::DOM::HTML_MENU_ELEMENT:
            case LFL::DOM::HTML_DIV_ELEMENT:            case LFL::DOM::HTML_O_LIST_ELEMENT:
            case LFL::DOM::HTML_D_LIST_ELEMENT:         case LFL::DOM::HTML_PARAGRAPH_ELEMENT:
            case LFL::DOM::HTML_FIELD_SET_ELEMENT:      case LFL::DOM::HTML_PRE_ELEMENT:
            case LFL::DOM::HTML_FORM_ELEMENT:           case LFL::DOM::HTML_TABLE_ELEMENT:
            case LFL::DOM::HTML_HEADING_ELEMENT:        case LFL::DOM::HTML_U_LIST_ELEMENT:
                return LFL::DOM::Display(LFL::DOM::Display::Block);
        } return LFL::DOM::Display(LFL::DOM::Display::Inline);
    }
    LFL::DOM::Clear                Clear               () { return LFL::DOM::Clear(LFL::DOM::Clear::None); }
    LFL::DOM::Float                Float               () { return LFL::DOM::Float(LFL::DOM::Float::None); }
    LFL::DOM::Position             Position            () { return LFL::DOM::Position(LFL::DOM::Position::Static); }
    LFL::DOM::Overflow             Overflow            () { return LFL::DOM::Overflow(LFL::DOM::Overflow::Visible); }
    LFL::DOM::TextAlign            TextAlign           () { return LFL::DOM::TextAlign(LFL::DOM::TextAlign::Left); }
    LFL::DOM::Direction            Direction           () { return LFL::DOM::Direction(LFL::DOM::Direction::LTR); }
    LFL::DOM::FontStyle            FontStyle           () { return LFL::DOM::FontStyle(LFL::DOM::FontStyle::Normal); }
    LFL::DOM::EmptyCells           EmptyCells          () { return LFL::DOM::EmptyCells(LFL::DOM::EmptyCells::Show); }
    LFL::DOM::FontWeight           FontWeight          () { return LFL::DOM::FontWeight(LFL::DOM::FontWeight::Normal); }
    LFL::DOM::Visibility           Visibility          () { return LFL::DOM::Visibility(LFL::DOM::Visibility::Visible); }
    LFL::DOM::WhiteSpace           WhiteSpace          () { return LFL::DOM::WhiteSpace(LFL::DOM::WhiteSpace::Normal); }
    LFL::DOM::CaptionSide          CaptionSide         () { return LFL::DOM::CaptionSide(LFL::DOM::CaptionSide::Top); }
    LFL::DOM::FontVariant          FontVariant         () { return LFL::DOM::FontVariant(LFL::DOM::FontVariant::Normal); }
    LFL::DOM::TableLayout          TableLayout         () { return LFL::DOM::TableLayout(LFL::DOM::TableLayout::Auto); }
    LFL::DOM::UnicodeBidi          UnicodeBidi         () { return LFL::DOM::UnicodeBidi(LFL::DOM::UnicodeBidi::Normal); }
    LFL::DOM::BorderStyle          OutlineStyle        () { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
    LFL::DOM::TextTransform        TextTransform       () { return LFL::DOM::TextTransform(LFL::DOM::TextTransform::None); }
    LFL::DOM::TextDecoration       TextDecoration      () { return LFL::DOM::TextDecoration(IsLink() ? LFL::DOM::TextDecoration::Underline : LFL::DOM::TextDecoration::None); }
    LFL::DOM::BorderCollapse       BorderCollapse      () { return LFL::DOM::BorderCollapse(LFL::DOM::BorderCollapse::Separate); }
    LFL::DOM::ListStyleType        ListStyleType       () { return LFL::DOM::ListStyleType(LFL::DOM::ListStyleType::Disc); }
    LFL::DOM::ListStylePosition    ListStylePosition   () { return LFL::DOM::ListStylePosition(LFL::DOM::ListStylePosition::Outside); }
    LFL::DOM::PageBreak            PageBreakAfter      () { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
    LFL::DOM::PageBreak            PageBreakBefore     () { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
    LFL::DOM::PageBreak            PageBreakInside     () { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
    LFL::DOM::BorderWidth          BorderTopWidth      () { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
    LFL::DOM::BorderWidth          BorderLeftWidth     () { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
    LFL::DOM::BorderWidth          BorderRightWidth    () { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
    LFL::DOM::BorderWidth          BorderBottomWidth   () { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
    LFL::DOM::BackgroundRepeat     BackgroundRepeat    () { return LFL::DOM::BackgroundRepeat(LFL::DOM::BackgroundRepeat::Repeat); }
    LFL::DOM::BackgroundAttachment BackgroundAttachment() { return LFL::DOM::BackgroundAttachment(LFL::DOM::BackgroundAttachment::Scroll); }
    LFL::DOM::RGBColor             BackgroundColor     () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             BorderBottomColor   () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             BorderLeftColor     () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             BorderRightColor    () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             BorderTopColor      () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             OutlineColor        () { return LFL::DOM::RGBColor(0x00000000); }
    LFL::DOM::RGBColor             Color               () { return LFL::DOM::RGBColor(0xffffffff); }
    LFL::DOM::Cursor               Cursor              () { return LFL::DOM::Cursor(LFL::DOM::Cursor::Auto); }
    LFL::DOM::FontFamily           FontFamily          () { return LFL::DOM::FontFamily(FLAGS_default_font); }
    LFL::DOM::CSSStringValue       BackgroundImage     () { return LFL::DOM::CSSStringValue(); }
    LFL::DOM::CSSNumericValuePair  BackgroundPosition  () { return LFL::DOM::CSSNumericValuePair(); }
    LFL::DOM::Rect                 Clip                () { return LFL::DOM::Rect(); }
    LFL::DOM::CSSNumericValue      Opacity             () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      Orphans             () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      Widows              () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      MinHeight           () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      MinWidth            () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSNumericValue      PaddingTop          () { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSNumericValue      PaddingRight        () { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSNumericValue      PaddingBottom       () { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSNumericValue      PaddingLeft         () { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSNumericValue      TextIndent          () { return LFL::DOM::CSSNumericValue(); }
    LFL::DOM::CSSAutoNumericValue  Top                 () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  Bottom              () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  Left                () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  Right               () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  Width               () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  Height              () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::FontSize             FontSize            () { return LFL::DOM::FontSize(LFL::DOM::FontSize::Medium); }
    LFL::DOM::VerticalAlign        VerticalAlign       () { return LFL::DOM::VerticalAlign(LFL::DOM::VerticalAlign::Baseline); }
    LFL::DOM::BorderStyle          BorderTopStyle      () { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
    LFL::DOM::BorderStyle          BorderRightStyle    () { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
    LFL::DOM::BorderStyle          BorderBottomStyle   () { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
    LFL::DOM::BorderStyle          BorderLeftStyle     () { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
    LFL::DOM::BorderWidth          OutlineWidth        () { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
    LFL::DOM::CSSAutoNumericValue  MarginBottom        () { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSAutoNumericValue  MarginLeft          () { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSAutoNumericValue  MarginRight         () { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSAutoNumericValue  MarginTop           () { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
    LFL::DOM::CSSNoneNumericValue  MaxHeight           () { return LFL::DOM::CSSNoneNumericValue(true); }
    LFL::DOM::CSSNoneNumericValue  MaxWidth            () { return LFL::DOM::CSSNoneNumericValue(true); }
    LFL::DOM::CSSAutoNumericValue  ZIndex              () { return LFL::DOM::CSSAutoNumericValue(true); }
    LFL::DOM::CSSNormNumericValue  WordSpacing         () { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
    LFL::DOM::CSSNormNumericValue  LetterSpacing       () { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
    LFL::DOM::CSSNormNumericValue  LineHeight          () { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight); }
};
struct StyleContext : public LFL::DOM::Object {
    StyleContext(LFL::DOM::HTMLDocument *D) {}
    void AppendSheet(StyleSheet *ss) {}
    void Match(ComputedStyle *out, LFL::DOM::Node *node, ComputedStyle *parent_sheet=0, const char *inline_style=0) {}
    int FontFaces(const string &face_name, vector<LFL::DOM::FontFace> *out) { out->clear(); return 0; }
};
#endif // LFL_LIBCSS

}; // namespace LFL
#endif // __LFL_LFAPP_CSS_H__
