/*
 * $Id: html.h 1316 2014-10-20 04:34:40Z justin $
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

#ifndef LFL_WEB_HTML_H__
#define LFL_WEB_HTML_H__
namespace LFL {

struct HTMLParser {
  typedef LFL::DOM::Char      Char;
  typedef LFL::DOM::DOMString String;

  struct KV : public map<String, String> {
    KV() {}
    KV(const KV &in) : map<String, String>(in) {}
    String str() const { String v; for (const_iterator i=begin(); i!=end(); i++) StringAppendf(&v, "%s=\"%s\" ", (*i).first.c_str(), (*i).second.c_str()); return v; }
    String field(const String &k) const { const_iterator i = find(k); return i != end() ? (*i).second : ""; }
    String id() const { return field("id"); }
  };

  struct Tag {
    int id; String name; KV attr; bool closed;
    Tag()                                     : id(0),                    closed(0) {}
    Tag(int ID, const String &N, const KV &A) : id(ID), name(N), attr(A), closed(0) {}

    enum {
#undef   XX
#define  XX(x) x,
#include "web/html_tags.h"
      bang, doctype, _small
    };
  };

  struct TagStack : public vector<Tag> {
    String str() const { String v; for (const_iterator i=begin(); i!=end(); i++) StringAppendf(&v, ".<%s>", (*i).name.c_str()); return v; }
    String idstr(bool filter=0) const { String v; for (const_iterator i=begin(); i!=end(); i++) if (!filter || (filter && (*i).attr.id().size())) StringAppendf(&v, ".<%s id=%s>", (*i).name.c_str(), (*i).attr.id().c_str()); return v; }
    String attrstr() const { String v; for (const_iterator i=begin(); i!=end(); i++) StringAppendf(&v, ".<%s %s>", (*i).name.c_str(), (*i).attr.str().c_str()); return v; }
  } stack;

  bool lower_attrs, tag_soup, intag, incomment, inscript, instyle, inpre, done_html, done_head, done_body;
  String text; string charset, deferred;

  HTMLParser() : lower_attrs(1), tag_soup(1) { text.reserve(4096); Reset(); }
  void Reset() {
    intag=incomment=inscript=instyle=inpre=done_html=done_head=done_body=0;
    stack.clear(); text.clear(); charset.clear();
  }

  virtual void OpenTag(const String &tag, const KV &attr, const TagStack &stack) {}
  virtual void CloseTag(const String &tag, const KV &attr, const TagStack &stack) {}
  virtual void Text(const String &text, const TagStack &stack) {}
  virtual void Script(const String &text, const TagStack &stack) {}
  virtual void Style(const String &text, const TagStack &stack) {}
  virtual void Comment(const String &text, const TagStack &stack) {}

  virtual void ParseChunkCB(const Char *content, int content_len) {}
  virtual void WGetContentBegin(Connection *c, const char *h, const string &content_type) {}
  virtual void WGetContentEnd(Connection *c) {}

  void ParseDeferred() { Parse(deferred.data(), deferred.size()); deferred.clear(); }
  void Parse(File *f) { NextRecordReader nr(f); for (const char *b = nr.NextChunk(); b; b = nr.NextChunk()) Parse(b, nr.record_len); }
  void Parse(const string &content) { return Parse(content.c_str(), content.size()); }
  void Parse(const char *input_content, int input_content_len) {
    static const char *target_charset = LFL::DOM::Encoding<Char>();
    String charset_normalized_content;
    if (charset.empty()) charset = "UTF-8";
    bool convert_charset = charset != target_charset;
    if (convert_charset)
      ::LFL::String::Convert(StringPiece(input_content, input_content_len),
                             &charset_normalized_content, charset.c_str(), target_charset);
    StringPiece content(convert_charset ? charset_normalized_content.data() : input_content,
                        convert_charset ? charset_normalized_content.size() : input_content_len);

    static String comment_start="!--", comment_close="--", style_close="</style", script_close="</script";
    for (const Char *p=content.buf, *e=content.buf+content.len; app->run && p<e; p++) {
      if (intag && text.size() == comment_start.size() && text == comment_start) { intag=0; incomment=1; text.clear(); }
      bool inspecial = inscript || instyle || incomment;
      if      (*p == '>' && instyle   && CloseSpecial(style_close,   &HTMLParser::Style,   1)) continue;
      else if (*p == '>' && inscript  && CloseSpecial(script_close,  &HTMLParser::Script,  1)) continue;
      else if (*p == '>' && incomment && CloseSpecial(comment_close, &HTMLParser::Comment, 0)) { incomment=0; continue; }
      else if (*p == '<' && !intag && !inspecial) { intag = 1; if (text.size()) {    Text(text); text.clear(); } continue; }
      else if (*p == '>' &&  intag && !inspecial) { intag = 0; if (text.size()) { TagText(text); text.clear(); } continue; }
      text.append(1, *p);
    }
    if (text.size() && !intag && !inscript && !instyle && !incomment) { Text(text); text.clear(); }
    ParseChunkCB(content.buf, content.len);
  }

  void Text(String &text) {
    if (!inpre && !FindChar(text.c_str(), notspace, text.size())) return;
    Text(text, stack);
  }

  void TagText(const String &tag_in) {
    KV attr;
    String tag = tag_in, tagname;
    Char *base = reinterpret_cast<Char*>(&tag[0]), *space = FindChar(base, isspace);
    bool selfclose = tag.size() && tag[tag.size()-1] == '/';
    if (!space) tagname = tag.substr(0, tag.size()-selfclose);
    else {
      int len = space-base, outlen;
      tagname = tag.substr(0, len);
      len = tag.size()-len-1;
      Char *key, *val; bool done=0, dquote=0;
      for (Char *b = space+1, *e = b+len, *p = b; p && p < e; /**/) {
        if (!(key = FindChar(p, notspace, len-(p-b)))) break;
        if (!(p = FindChar(key, notalnum, len-(key-b)))) break;
        if (isspace(*p)) *p++ = 0;
        if (!(p = FindChar(p, notspace, len-(p-b)))) break;
        if (*p != '=') break;
        else *p++ = 0;
        if (!(val = FindChar(p, notspace, len-(p-b)))) break;
        if ((dquote = *val == '"') || *val == '\'') {
          val++;
          if (!(p = FindChar(val, dquote ? isdquote : issquote, len-(val-b)))) break;
          *p++ = 0;
          if (*p && !isspace(*p)) done=1;
        }
        else {
          if ((p = FindChar(val, isspace, len-(val-b)))) *p++ = 0;
        }
        attr[lower_attrs ? tolower(key) : key] = val;
        if (done) break;
      }
    }
    if (!tagname.size()) return;

    tagname = tolower(tagname);
    bool close = tagname[0] == '/';
    int tagid = Tags::Lookup(tagname.substr(close));
    bool ptag = tagid == Tag::p, litag = tagid == Tag::li, optiontag = tagid == Tag::option;
    bool roottag = tagid == Tag::html, bodytag = tagid == Tag::body, headtag = tagid == Tag::head;
    if (close) tagname = tagname.substr(1);
    if (!close) {
      bool root_child       = headtag             || bodytag;
      bool header_selfclose = tagid == Tag::meta  || tagid == Tag::link   || tagid == Tag::base;
      bool header_normclose = tagid == Tag::title || tagid == Tag::script || tagid == Tag::style;
      bool specialtag       = tagid == Tag::bang  || tagid == Tag::doctype;
      bool imagetag         = tagid == Tag::img   || tagid == Tag::image;
      bool coltag           = tagid == Tag::col;
      bool headertag        = header_selfclose   || header_normclose;

      if (tagid == Tag::input || tagid == Tag::br || tagid == Tag::hr || imagetag || specialtag
          || header_selfclose || coltag) selfclose = true;

      if (!specialtag) {
        if (stack.size() == 0 && !roottag)  PushTag("html", KV(), &done_html);
        if (stack.size() == 1 && !root_child) {
          if      (!done_head && headertag) PushTag("head", KV(), &done_head);
          else if (!done_body)              PushTag("body", KV(), &done_head, &done_body);
        }
        if (!done_body && (bodytag || (stack.size() >= 2 && stack[1].id == Tag::head && !headertag))) {
          PopTag(stack.size()-1);
          if (!bodytag) PushTag("body", KV(), &done_head, &done_body);
        }
        if      (roottag) { if (done_html) return; done_html = true; }
        else if (headtag) { if (done_head) return; done_head = true; }
        else if (bodytag) { if (done_body) return; done_head = done_body = true; }
      }

      bool table_row = tagid == Tag::tr,    table_col = tagid == Tag::th || tagid == Tag::td;
      bool table     = tagid == Tag::table, table_dep = TableDependent(tagid) && !coltag;
      bool block_el  = BlockElement(tagid), addrow=0, addtable=0, addcolgroup=0;

      if (block_el)  { PopTag(FindStackOffset(stack, "p")); if (Tag::p == stack.back().id) PopTag(1); }
      if (litag)     { PopTag(FindStackOffset(stack, "li", "ul", "ol")); }
      if (optiontag) { PopTag(FindStackOffset(stack, "option")); }
      if (table_col) { PopTag(FindStackOffset(stack, "tr", "table")); if (Tag::tr != stack.back().id) { addrow=1; table_row=1; } }
      if (coltag) { PopTag(FindStackOffset(stack, "colgroup")); if (Tag::colgroup != stack.back().id) { addcolgroup=1; table_dep=1; } }
      if (table_row) { PopTag(FindStackOffsetRowCont(stack)); if (!RowContainer(stack.back().id)) addtable=1; }
      if (table_dep) { PopTag(FindStackOffset(stack, "table")); if (Tag::table != stack.back().id) addtable=1; }

      if (addtable)    PushTag("table",    KV());
      if (addcolgroup) PushTag("colgroup", KV());
      if (addrow)      PushTag("tr",       KV());

      if (!selfclose) stack.push_back(Tag(tagid, tagname, attr));
      OpenTag(tagname, attr, stack);
      if (selfclose) CloseTag(tagname, attr, stack);
    }
    else {
      if (!tag_soup) {
        if (tagid == Tag::table) PopTag(FindStackOffset(stack, "table"));
        else if (stack.size() && stack.back().name == tagname) PopTag(1);
      } else {
        bool cant_close = 0;
        int match_offset = stack.size()-1;
        for (/**/; match_offset >= 0 && stack[match_offset].name != tagname; match_offset--)
          if (!cant_close && TagSoupCantClose(stack[match_offset].id)) cant_close = 1;

        if      (match_offset == stack.size()-1) { PopTag(1); PopTag(FindStackOffsetClosed(stack)); }
        else if (match_offset >= 0 && cant_close) stack[match_offset].closed = 1;
        else if (match_offset >= 0) {
          TagStack restart;
          for (int i=match_offset+1; i<stack.size(); i++)
            if (TagSoupRestartable(stack[i].id)) restart.push_back(stack[i]);
          PopTag(stack.size() - match_offset);
          PopTag(FindStackOffsetClosed(stack));
          for (int i=0; i<restart.size(); i++)
            PushTag(restart[i].id, restart[i].name, restart[i].attr);
        }
      }
    }
    if      (tagid == Tag::script) inscript = !close;
    else if (tagid == Tag::style)  instyle  = !close;
    else if (tagid == Tag::pre)    inpre    = !close;
  }

  void PushTag(           const String &tag, const KV &attr, bool *done1=0, bool *done2=0) { return PushTag(Tags::Lookup(tag), tag, attr, done1, done2); }
  void PushTag(int tagid, const String &tag, const KV &attr, bool *done1=0, bool *done2=0) {
    stack.push_back(Tag(tagid, tag, attr));
    OpenTag(tag, attr, stack);
    if (done1) *done1 = true;
    if (done2) *done2 = true;
  }

  void PopTag(int n) {
    for (/**/; n > 0 && stack.size(); n--) {
      const Tag &tag = stack.back();
      if (tag.id == Tag::html || tag.id == Tag::body) return;
      CloseTag(tag.name, stack.back().attr, stack);
      stack.pop_back();
    }
  }

  bool CloseSpecial(const String &close_text, void (HTMLParser::*close_cb)(const String&, const TagStack&), bool close_tag) {
    if (text.size() >= close_text.size() &&
        StringEquals(text.substr(text.size()-close_text.size()), close_text)) {
      (this->*close_cb)(text.substr(0, text.size()-close_text.size()), stack);
      if (close_tag) TagText(close_text.substr(1));
      text.clear(); 
      return true;
    } return false;
  }

  void WGetCB(Connection *c, const char *h, const string &ct, const char *b, int l) {
    if (h) {
      int charset_ind = ct.find("charset=");
      if (charset_ind != string::npos) charset = ct.substr(charset_ind + 8);
      WGetContentBegin(c, h, ct);
    } else if (charset.empty()) {
#if 0
      else if (tag == "meta") {
        if (StringEquals(addNode->getAttribute("http-equiv"), "Content-Type")) {
          browser->doc.content_type = addNode->getAttribute("content");
        } else if (LFL::DOM::Node *cs = addNode->getAttributeNode("charset")) {
          browser->doc.char_set = cs->nodeValue();
        }
      }
#endif
      charset = "UTF-8";
      ParseDeferred();
    }

    if      (h) {}
    else if (l) {
      if (charset.empty()) deferred.append(b, l);
      else                 Parse          (b, l);
    } else {
      if (charset.empty()) {
        charset = "UTF-8";
        ParseDeferred();
      }
      WGetContentEnd(c);
    }
  }

  static bool RowContainer(int n) { return n == Tag::table || TableSection(n); }
  static bool TableSection(int n) { return n == Tag::thead || n == Tag::tbody || n == Tag::tfoot; }
  static bool TableDependent(const string &n) { return n == "caption" || n == "col" || n == "colgroup" || n == "thead" || n == "tbody" || n == "tfoot"; }
  static bool TableDependent(int n) { return n == Tag::caption || n == Tag::col || n == Tag::colgroup || TableSection(n); }
  static bool BlockElement(int n) {
    switch (n) {
      case Tag::blockquote:    case Tag::canvas:     case Tag::div:     case Tag::dl:
      case Tag::fieldset:      case Tag::form:       case Tag::h1:      case Tag::h2:
      case Tag::h3:            case Tag::h4:         case Tag::h5:      case Tag::h6:
      case Tag::hr:            case Tag::menu:       case Tag::ol:      case Tag::p:
      case Tag::pre:           case Tag::table:      case Tag::ul:      return 1;
      default:                                                          return 0;
    }
  }
  static bool InlineElement(int n) {
    switch (n) {
      case Tag::abbr:      case Tag::acronym:    case Tag::b:       case Tag::bdo:
      case Tag::big:       case Tag::blink:      case Tag::cite:    case Tag::code:
      case Tag::del:       case Tag::dfn:        case Tag::em:      case Tag::i:
      case Tag::ins:       case Tag::kbd:        case Tag::q:       case Tag::rb:
      case Tag::rbc:       case Tag::rp:         case Tag::rt:      case Tag::rtc:
      case Tag::ruby:      case Tag::s:          case Tag::samp:    case Tag::_small:
      case Tag::strike:    case Tag::strong:     case Tag::sub:     case Tag::sup:
      case Tag::tt:        case Tag::u:          return 1;
      default:                                   return 0;
    }
  }
  static bool TagSoupCantClose(int n) { return n == Tag::body || n == Tag::form || n == Tag::table; }
  static bool TagSoupRestartable(int n) { return InlineElement(n); }

  static int FindStackOffset(const TagStack &stack, const String &tag) {
    for (int i=stack.size()-1; i>=0; i--) if (stack[i].name == tag) return stack.size()-1-i;
    return -1;
  }
  static int FindStackOffset(const TagStack &stack, const String &tag, const String &stop) {
    for (int i=stack.size()-1; i>=0; i--) {
      if (stack[i].name == stop) return Xge0_or_Y(stack.size()-i-2, 0);
      if (stack[i].name == tag)  return           stack.size()-i-1;
    }
    return -1;
  }
  static int FindStackOffset(const TagStack &stack, const String &tag, const String &stop1, const String &stop2) {
    for (int i=stack.size()-1; i>=0; i--) {
      if (stack[i].name == stop1 || stack[i].name == stop2) return Xge0_or_Y(stack.size()-i-2, 0);
      if (stack[i].name == tag)                             return           stack.size()-i-1;
    }
    return -1;
  }
  static int FindStackOffsetClosed(const TagStack &stack) {
    for (int i=stack.size()-1; i>=0; i--) if (!stack[i].closed) return stack.size()-1-i;
    return -1;
  }
  static int FindStackOffsetRowCont(const TagStack &stack) {
    for (int i=stack.size()-1; i>=0; i--) if (RowContainer(stack[i].id)) return stack.size()-1-i;
    return -1;
  }

  static bool MatchStack(const TagStack &stack, int numtagkey, ...) {
    va_list ap; va_start(ap, numtagkey);
    for (int i=stack.size()-1; i>=0; i--) {
      if (tolower(stack[i].name) != va_arg(ap, Char*)) break;
      if (--numtagkey == 0) return true;
    }
    va_end(ap);
    return false;
  }
  static bool MatchStackCase(const TagStack &stack, int numtagkey, ...) {
    va_list ap; va_start(ap, numtagkey);
    for (int i=stack.size()-1; i>=0; i--) {
      if (stack[i].name != va_arg(ap, Char*)) break;
      if (--numtagkey == 0) return true;
    }
    va_end(ap);
    return false;
  }
  static bool MatchStackOffset(const TagStack &stack, int offset, int numtagkey, ...) {
    va_list ap; va_start(ap, numtagkey);
    for (int i=stack.size()-1-offset; i>=0; i--) {
      if (stack[i].name != va_arg(ap, Char*)) break;
      if (--numtagkey == 0) return true;
    }
    va_end(ap);
    return false;
  }
  static bool MatchStackAttr(const TagStack &stack, const Char *tagkey, const Char *attrkey, int numattrval, ...) {
    va_list ap; va_start(ap, numattrval);
    for (int i=stack.size()-1; i>=0; i--) {
      if (stack[i].name != tagkey) continue;
      const KV &attrs = stack[i].attr;
      KV::const_iterator match;
      if ((match = attrs.find(attrkey)) == attrs.end()) continue;
      if ((*match).second != va_arg(ap, Char*)) break;
      if (--numattrval == 0) return true;
    }
    va_end(ap);
    return false;
  }

  template <class X> static basic_string<X> ReplaceEntitySymbols(const basic_string<X> &word) {
    bool utf16 = sizeof(X) != 1;
    basic_string<X> ret; ret.reserve(word.size());
    const X *p = word.c_str(), *chunk_start = p, *end = p + word.size();
    while (p < end) {
      if (*p != '&') { p++; continue; }
      const X *s = p + 1, *max_entity = s + 8; bool num = *s == '#';
      for (p = s + num; p < max_entity && (num ? isnumber(*p) : isalnum(*p)); p++) {}
      if (*p != ';') continue;

      ret.append(chunk_start, s-1-chunk_start);
      int symbol = num ? atoi(s+1) : Entity::Lookup(basic_string<X>(s, p-s));
      ret.append(UTF<X>::WriteGlyph(symbol));
      chunk_start = p++ + 1;
    }
    ret.append(chunk_start, end);
    return ret;
  }
  string   ReplaceEntitySymbols(const string   &word) { return ReplaceEntitySymbols<char>    (word); }
  String16 ReplaceEntitySymbols(const String16 &word) { return ReplaceEntitySymbols<char16_t>(word); }

  struct Entity : public unordered_map<string, char16_t> {
    enum {
#undef   XX
#define  XX(x, y) x = y,
#include "web/html_entities.h"
    };
    Entity() {
#undef   XX
#define  XX(x, y) (*this)[#x] = y;
#include "web/html_entities.h"
      // C++ reserved word entities
      (*this)["not"] = 172;
      (*this)["and"] = 8743;
      (*this)["or"]  = 8744;
      (*this)["int"] = 8747;
    }
    static char16_t Lookup(const string   &n) { return FindOrDefault(*Singleton<Entity>::Get(), n,                         ' '); }
    static char16_t Lookup(const String16 &n) { return FindOrDefault(*Singleton<Entity>::Get(), ::LFL::String::ToAscii(n), ' '); }
  };

  struct Tags : public unordered_map<string, char16_t> {
    Tags() {
#undef   XX
#define  XX(x) (*this)[#x] = Tag::x;
#include "web/html_tags.h"
      (*this)["!"]        = Tag::bang;
      (*this)["!doctype"] = Tag::doctype;
      (*this)["small"]    = Tag::_small;
    }
    static char16_t Lookup(const string   &n) { return FindOrDefault(*Singleton<Tags>::Get(), n,                         0); }
    static char16_t Lookup(const String16 &n) { return FindOrDefault(*Singleton<Tags>::Get(), ::LFL::String::ToAscii(n), 0); }
  };
};

struct DebugHTMLParser : public HTMLParser {
  //  virtual void OpenTag (const String &tag, const KV &attr, const TagStack &stack) { INFO("<", tag, " ", attr.str(), "> [", stack.idstr(), "]"); }
  //  virtual void CloseTag(const String &tag, const KV &attr, const TagStack &stack) { INFO("<", tag, " ", attr.str(), "> [", stack.idstr(), "]"); }
  virtual void Text(const String &text, const TagStack &stack) { INFO(text, " [", stack.idstr(true), "]"); }
};

}; // namespace LFL
#endif // LFL_WEB_HTML_H__
