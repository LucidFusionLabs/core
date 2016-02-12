/*
 * $Id: dom.h 1334 2014-11-28 09:14:21Z justin $
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

#ifndef LFL_LFAPP_DOM_H__
#define LFL_LFAPP_DOM_H__
namespace LFL {
namespace DOM {
struct Node;
struct Attr;
struct Element;
struct Frame;
struct Text;
struct CDATASection;
struct Comment;
struct Notation;
struct Entity;
struct EntityReference;
struct ProcessingInstruction;
struct Document;
struct DocumentType;
struct DocumentFragment;
struct HTMLDocument;
struct HTMLTableElement;
struct Renderer;

#undef  XX
#define XX(n) struct HTML ## n;
#include "web/html_elements.h"

typedef char Char;
typedef StringWordIterT<Char> StringWordIter;
typedef StringPieceT<Char> StringPiece;
#define DOMStringNull DOMString()

template <class X> static const char *Encoding() { 
  switch (sizeof(X)) {
    case 1: return "UTF-8";
    case 2: return "UTF-16LE";
  } return 0;
}

#if 1
typedef basic_string<Char> DOMString;
#else
struct DOMString : public basic_string<Char> {
  mutable int hash;
  int Hash() const { hash = fnv32(data(), size()); return hash; }
  DOMString()                                                :                                     hash(0) {}
  DOMString(size_t n, Char c)                                : basic_string<Char>(n,c),            hash(0) {}
  DOMString(const DOMString     &s, size_t o, size_t l=npos) : basic_string<Char>(s,o,l),          hash(0) {}
};
}; }; // namespace LFL::DOM
namespace LFL_STL_HASH_NAMESPACE {
template<> struct hash<LFL::DOM::DOMString> { size_t operator()(const LFL::DOM::DOMString &x) const { return x.Hash(); } };
}; // namespace LFL_STL_HASH_NAMESPAACE
namespace LFL {
namespace DOM {
#endif

enum {
  ELEMENT_NODE = 1, ATTRIBUTE_NODE = 2, TEXT_NODE = 3, CDATA_SECTION_NODE = 4,
  ENTITY_REFERENCE_NODE = 5, ENTITY_NODE = 6, PROCESSING_INSTRUCTION_NODE = 7,
  COMMENT_NODE = 8, DOCUMENT_NODE = 9, DOCUMENT_TYPE_NODE = 10, DOCUMENT_FRAGMENT_NODE = 11,
  NOTATION_NODE = 12
};

enum {
  HTML_FORM_ELEMENT = 1, HTML_SELECT_ELEMENT = 2, HTML_OPTION_ELEMENT = 3, HTML_INPUT_ELEMENT = 4,
  HTML_TEXT_AREA_ELEMENT = 5, HTML_BUTTON_ELEMENT = 6, HTML_ANCHOR_ELEMENT = 7, HTML_IMAGE_ELEMENT = 8,
  HTML_OBJECT_ELEMENT = 9, HTML_TABLE_CAPTION_ELEMENT = 10, HTML_TABLE_ELEMENT = 11,
  HTML_TABLE_SECTION_ELEMENT = 12, HTML_TABLE_ROW_ELEMENT = 13, HTML_TABLE_CELL_ELEMENT = 14,
  HTML_HTML_ELEMENT = 15, HTML_HEAD_ELEMENT = 16, HTML_OPT_GROUP_ELEMENT = 17, HTML_STYLE_ELEMENT = 18,
  HTML_LINK_ELEMENT = 19, HTML_TITLE_ELEMENT = 20, HTML_META_ELEMENT = 21, HTML_BASE_ELEMENT = 22,
  HTML_IS_INDEX_ELEMENT = 23, HTML_BODY_ELEMENT = 24, HTML_LABEL_ELEMENT = 25, HTML_FIELD_SET_ELEMENT = 26,
  HTML_LEGEND_ELEMENT = 27, HTML_U_LIST_ELEMENT = 28, HTML_O_LIST_ELEMENT = 29, HTML_D_LIST_ELEMENT = 30,
  HTML_DIRECTORY_ELEMENT = 31, HTML_MENU_ELEMENT = 32, HTML_LI_ELEMENT = 33, HTML_DIV_ELEMENT = 34,
  HTML_PARAGRAPH_ELEMENT = 35, HTML_HEADING_ELEMENT = 36, HTML_QUOTE_ELEMENT = 37, HTML_PRE_ELEMENT = 38,
  HTML_BR_ELEMENT = 39, HTML_BASE_FONT_ELEMENT = 40, HTML_FONT_ELEMENT = 41, HTML_HR_ELEMENT = 42,
  HTML_MOD_ELEMENT = 43, HTML_PARAM_ELEMENT = 44, HTML_APPLET_ELEMENT = 45, HTML_MAP_ELEMENT = 46,
  HTML_AREA_ELEMENT = 47, HTML_SCRIPT_ELEMENT = 48, HTML_TABLE_COL_ELEMENT = 49,
  HTML_FRAME_SET_ELEMENT = 50, HTML_FRAME_ELEMENT = 51, HTML_I_FRAME_ELEMENT = 52,
  HTML_BLOCK_QUOTE_ELEMENT = 53, HTML_CANVAS_ELEMENT = 54, HTML_SPAN_ELEMENT = 55,
  HTML_EMBED_ELEMENT = 56, HTML_MARQUEE_ELEMENT = 57
};

struct Object {
  virtual ~Object() {}
};

typedef int ExceptionCode;
struct DOMException : public Object {
  unsigned short code;
  enum {
    INDEX_SIZE_ERR = 1, DOMSTRING_SIZE_ERR = 2, HIERARCHY_REQUEST_ERR = 3, WRONG_DOCUMENT_ERR = 4,
    INVALID_CHARACTER_ERR = 5, NO_DATA_ALLOWED_ERR = 6, NO_MODIFICATION_ALLOWED_ERR = 7,
    NOT_FOUND_ERR = 8, NOT_SUPPORTED_ERR = 9, INUSE_ATTRIBUTE_ERR = 10
  };
};

struct DOMImplementation : public Object {
  virtual bool hasFeature(const DOMString &feature, const DOMString &version) { return false; }
};

template <class X> struct List : public Object {
  vector<X> data;
  void EraseAt(int i) { CHECK_RANGE(i, 0, data.size()); data.erase(data.begin() + i); }
  X InsertAt(int i, X v) { CHECK_RANGE(i, 0, data.size()+1); data.insert(data.begin() + i, v); return v; }
  X item(int index) const { return (index >= 0 && index < data.size()) ? data[index] : 0; }
  int length() const { return data.size(); }
  void clear() { data.clear(); }
};
typedef List<Node*> NodeList;

struct Node : public Object {
  virtual DOMString nodeName()  const = 0;
  virtual DOMString nodeValue() const = 0;

  virtual Attr* getAttributeNode(const DOMString &name) { return 0; }
  virtual string DebugString() const { return StrCat("<", nodeName(), "=\"", nodeValue(), "\">"); }
  virtual string HTML4Style();

  virtual Element*               AsElement()               { return 0; }
  virtual Attr*                  AsAttr()                  { return 0; }
  virtual Text*                  AsText()                  { return 0; }
  virtual CDATASection*          AsCDATASection()          { return 0; }
  virtual Comment*               AsComment()               { return 0; } 
  virtual Notation*              AsNotation()              { return 0; }
  virtual Entity*                AsEntity()                { return 0; }
  virtual EntityReference*       AsEntityReference()       { return 0; }
  virtual ProcessingInstruction* AsProcessingInstruction() { return 0; }
  virtual Document*              AsDocument()              { return 0; }
  virtual DocumentType*          AsDocumentType()          { return 0; }
  virtual DocumentFragment*      AsDocumentFragment()      { return 0; }
  virtual HTMLDocument*          AsHTMLDocument()          { return 0; }

# undef  XX
# define XX(n) virtual HTML ## n *AsHTML ## n () { return 0; }
# include "web/html_elements.h"

  unsigned short nodeType, htmlElementType;
  NodeList       childNodes;
  Node          *parentNode;
  Document      *ownerDocument;
  int            parentChildNodeIndex;

  LFL::DOM::Renderer *render;
  LFL::DOM::Renderer *AttachRender();
  void AssignParent(Node *n, int ind) { n->parentNode=this; n->parentChildNodeIndex=ind; }
  void ClearParent() { parentNode=0; parentChildNodeIndex=-1; }
  void ClearComputedInlineStyle();
  void SetLayoutDirty();
  void SetStyleDirty();

  virtual ~Node() {}
  Node(unsigned short t, Document *doc) : nodeType(t), htmlElementType(0), parentNode(0), ownerDocument(doc), parentChildNodeIndex(-1), render(0) {}
  Document *document() { return ownerDocument; }
  Node *firstChild() { return childNodes.item(0); }
  Node *lastChild() { int l=childNodes.length(); return l ? childNodes.item(l-1) : 0; }
  bool hasChildNodes() const { return childNodes.length(); }
  Node *cloneNode(bool deep);

  Node *insertBefore(Node *newChild, const Node *refChild) {
    Node *ret = 0;
    for (vector<Node*>::iterator i = childNodes.data.begin(); i != childNodes.data.end(); ++i) {
      if (ret) (*i)->parentChildNodeIndex++;
      else if (*i == refChild) {
        ret = newChild;
        AssignParent(newChild, i-childNodes.data.begin());
        i = childNodes.data.insert(i, ret);
      }
    }
    return 0;
  }

  Node *replaceChild(Node *newChild, const Node *oldChild) {
    for (vector<Node*>::iterator i = childNodes.data.begin(); i != childNodes.data.end(); ++i) {
      if (*i != oldChild) continue;
      (*i)->ClearParent();
      AssignParent(newChild, i-childNodes.data.begin());
      *i = newChild;
      return newChild;
    }
    return 0;
  }

  Node *removeChild(Node *oldChild) {
    Node *ret = 0;
    for (vector<Node*>::iterator i = childNodes.data.begin(); i != childNodes.data.end(); ++i) {
      if (*i == oldChild) {
        ret = oldChild;
        ret->ClearParent();
        i = childNodes.data.erase(i);
      }
      if (i == childNodes.data.end()) break;
      if (ret) (*i)->parentChildNodeIndex--;
    }
    return 0;
  }

  Node *appendChild(Node *newChild, ExceptionCode ec=0) {
    removeChild(newChild);
    AssignParent(newChild, childNodes.data.size());
    childNodes.data.push_back(newChild);
    return newChild;
  }

  Node *previousSibling() { return parentNode ? parentNode->childNodes.item(parentChildNodeIndex-1) : 0; }   
  Node     *nextSibling() { return parentNode ? parentNode->childNodes.item(parentChildNodeIndex+1) : 0; }
};

template <class X> struct NamedMap : public Object {
  vector<X> data_val;
  unordered_map<DOMString, int> data_index;

  X getNamedItem(const DOMString &name) const {
    int ind = FindOrDefault(data_index, name, -1);
    return ind >= 0 ? data_val[ind] : 0;
  }

  X createNamedItem(const DOMString &name, X arg) {
    data_index[name] = data_val.size();
    data_val.push_back(arg);
    return arg;
  }

  X setNamedItem(const DOMString &name, X arg) {
    bool inserted = false;
    unordered_map<DOMString, int>::iterator it = FindOrInsert(data_index, name, (int)data_val.size(), &inserted);
    if (inserted) data_val.push_back(arg);
    else data_val[it->second] = arg;
    return arg;
  }

  X removeNamedItem(const DOMString &name) {
    unordered_map<DOMString, int>::iterator it = data_index.find(name);
    if (it == data_index.end()) return 0;
    int ind = it->second;
    X ret = data_val[ind];
    data_index.erase(it);
    data_val.erase(data_val.begin() + ind);
    for (it = data_index.begin(); it != data_index.end(); it++) if (it->second > ind) it->second--;
    return ret;
  }

  X item(int index) const { return (index >= 0 && index < data_val.size()) ? data_val[index] : 0; }
  int length() const { return data_val.size(); }
};
typedef NamedMap<Node*> NamedNodeMap;

#define DERIVE_NODE_IMPL(name, type) \
  static const int Type = type; \
  name* As ## name() { return this; }
#define DERIVE_NODE_FROM(name, from, type) DERIVE_NODE_IMPL(name, type); name(Document *doc) : from(type, doc)
#define DERIVE_NODE(name, type) DERIVE_NODE_FROM(name, Node, type)

struct Attr : public Node { DERIVE_NODE(Attr, ATTRIBUTE_NODE) {}
  DOMString name, value; bool specified;
  virtual DOMString nodeName()  const { return name; }
  virtual DOMString nodeValue() const { return value; }
  virtual void SetValue(const DOMString &v) { value=v; specified=true; }
  virtual void SetNameValue(const DOMString &n, const DOMString &v) { name=n; SetValue(v); }
};

struct Entity : public Node { DERIVE_NODE(Entity, ENTITY_NODE) {}
  DOMString nodeName_, publicId, systemId, notationName;
  virtual DOMString nodeName()  const { return nodeName_; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct Notation : public Node { DERIVE_NODE(Notation, NOTATION_NODE) {}
  DOMString nodeName_, publicId, systemId;
  virtual DOMString nodeName()  const { return nodeName_; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct ProcessingInstruction : public Node { DERIVE_NODE(ProcessingInstruction, PROCESSING_INSTRUCTION_NODE) {}
  DOMString target, data;
  virtual DOMString nodeName()  const { return target; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct EntityReference : public Node { DERIVE_NODE(EntityReference, ENTITY_NODE) {}
  virtual DOMString nodeName()  const { return DOMStringNull; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct DocumentFragment : public Node { DERIVE_NODE(DocumentFragment, DOCUMENT_FRAGMENT_NODE) {}
  virtual DOMString nodeName()  const { static DOMString v("#document-fragment"); return v; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct DocumentType : public Node { DERIVE_NODE(DocumentType, DOCUMENT_TYPE_NODE) {}
  DOMString name; NamedNodeMap entities, notations;
  virtual DOMString nodeName()  const { return name; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
};

struct Element : public Node { DERIVE_NODE(Element, ELEMENT_NODE) {}
  DOMString    tagName;
  NamedNodeMap attributes;
  virtual DOMString nodeName()  const { return tagName; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
  virtual void      ParseAttributes(const map<DOMString, DOMString> &attr) {}
  virtual void      normalize() {}
  virtual void      setAttribute(const DOMString &name, const DOMString &value);
  virtual DOMString getAttribute(const DOMString &name) { Node *n = getAttributeNode(name); return n ? n->nodeValue() : DOMString(); }
  virtual void      removeAttribute(const DOMString &name) { attributes.removeNamedItem(name); }
  virtual Attr     *getAttributeNode(const DOMString &name) { return (Attr*)attributes.getNamedItem(name); }
  virtual Attr     *setAttributeNode(Attr *newAttr) { attributes.setNamedItem(newAttr->nodeName(), newAttr); return newAttr; }
  virtual Attr     *removeAttributeNode(const Attr *oldAttr) { attributes.removeNamedItem(oldAttr->nodeName()); return NULL; }
  virtual NodeList  getElementsByTagName(const DOMString &name) { return getElementsByTagName(this, name); }
  virtual string DebugString() const {
    string ret = "<" + String::ToUTF8(nodeName());
    for (int i=0; i<attributes.data_val.size(); i++) {
      Node *n = attributes.data_val[i];
      StrAppend(&ret, " ", n->nodeName(), "=\"", n->nodeValue(), "\"");
    }
    return ret + ">";
  }

  static NodeList getElementsByTagName(Node *root, const DOMString &name) {
    NodeList ret;
    for (int i=0, l=root->childNodes.length(); i<l; i++) {
      Node *n = root->childNodes.item(i);
      Element *e = n->AsElement();
      if (e && e->tagName == name) ret.data.push_back(e);
    }
    return ret;
  }
};

struct CharacterData : public Node {
  DOMString data;
  CharacterData(unsigned short t, Document *doc) : Node(t, doc) {}
  int length() const { return data.size(); }
  virtual DOMString nodeName()  const { return DOMStringNull; }
  virtual DOMString nodeValue() const { return data; } 
  virtual DOMString substringData(unsigned long offset, unsigned long count) { return data.substr(offset, count); }
  virtual void appendData(const DOMString &arg, ExceptionCode ec=0) { data.append(arg); }
  virtual void insertData(unsigned long offset, const DOMString &arg) { data.insert(offset, arg); } 
  virtual void deleteData(unsigned long offset, unsigned long count) { data.erase(offset, count); }
  virtual void replaceData(unsigned long offset, unsigned long count, const DOMString &arg) { data.assign(arg, offset, count); }
};

struct Comment : public CharacterData { DERIVE_NODE_FROM(Comment, CharacterData, COMMENT_NODE) {} 
  virtual DOMString nodeName() const { return "#comment"; }
};

struct Text : public CharacterData { DERIVE_NODE_FROM(Text, CharacterData, TEXT_NODE) {}
  Text(unsigned short t, Document *doc) : CharacterData(t, doc) {}
  Text splitText(unsigned long offset);
  virtual DOMString nodeName() const { return "#text"; }
};

struct CDATASection : public Text { DERIVE_NODE_FROM(CDATASection, Text, CDATA_SECTION_NODE) {}
  virtual DOMString nodeName() const { return "#cdata-section"; }
};

struct Document : public Node { DERIVE_NODE_IMPL(Document, DOCUMENT_NODE);
  DocumentType        doctype;
  DOMImplementation   implementation;

  DocumentParser   *parser;
  Allocator        *alloc;
  StyleContext     *style_context=0, *inline_style_context=0;

  Document(DocumentParser *p, Allocator *a) : Node(DOCUMENT_NODE, this), doctype(this), parser(p), alloc(a) {}

  virtual DOMString nodeName()  const { return "#document"; }
  virtual DOMString nodeValue() const { return DOMStringNull; }
  virtual DocumentFragment *createDocumentFragment() { return AllocatorNew(ownerDocument->alloc, (DocumentFragment), (ownerDocument)); }
  virtual Attr                  *createAttribute(const DOMString &name) { return AllocatorNew(ownerDocument->alloc, (Attr), (ownerDocument)); }
  virtual EntityReference *createEntityReference(const DOMString &name) { return AllocatorNew(ownerDocument->alloc, (EntityReference), (ownerDocument)); }
  virtual Element                 *createElement(const DOMString &name) { Element      *ret = AllocatorNew(ownerDocument->alloc, (Element),      (ownerDocument)); ret->tagName = name; return ret; }
  virtual Text                   *createTextNode(const DOMString &data) { Text         *ret = AllocatorNew(ownerDocument->alloc, (Text),         (ownerDocument)); ret->data    = data; return ret; }
  virtual Comment                 *createComment(const DOMString &data) { Comment      *ret = AllocatorNew(ownerDocument->alloc, (Comment),      (ownerDocument)); ret->data    = data; return ret; }
  virtual CDATASection       *createCDATASection(const DOMString &data) { CDATASection *ret = AllocatorNew(ownerDocument->alloc, (CDATASection), (ownerDocument)); ret->data    = data; return ret; }
  virtual ProcessingInstruction *createProcessingInstruction(const DOMString &target, const DOMString &data) { return AllocatorNew(ownerDocument->alloc, (ProcessingInstruction), (ownerDocument)); }
  virtual Element              *documentElement() { for (int i=0, l=childNodes.length(); i<l; i++) if (childNodes.item(i)->htmlElementType == HTML_HTML_ELEMENT) return childNodes.item(i)->AsElement(); return NULL; }
  virtual NodeList              getElementsByTagName(const DOMString &tagname) { return Element::getElementsByTagName(this, tagname); }
};

struct HTMLCollection : public NodeList {
  Node *namedItem(const DOMString &name) {
    for (vector<Node*>::iterator it = data.begin(); it != data.end(); it++) {
      Node *n = (*it)->getAttributeNode("id");
      if (n && n->nodeValue() == name) return n;
    }
    for (vector<Node*>::iterator it = data.begin(); it != data.end(); it++) {
      if ((*it)->nodeName() == name) return *it; 
    }
    return 0;
  }
};

#define DERIVE_ELEMENT_IMPL(name, type) \
  static const int Type = type; \
  name* As ## name() { return this; }
#define DERIVE_ELEMENT_FROM(name, from, type) DERIVE_ELEMENT_IMPL(name, type); name(Document *doc) : from(type, doc)
#define DERIVE_ELEMENT(name, type) DERIVE_ELEMENT_FROM(name, HTMLElement, type)

#define DECLARE_TAG_NAMES(...) static const char **TagNames() { static const char *TN[] = { __VA_ARGS__, 0 }; return TN; }

#define PARSE_ATTR_BEGIN(parent) virtual void ParseAttributes(const map<DOMString, DOMString> &attr) { \
  parent::ParseAttributes(attr); \
  map<DOMString, DOMString>::const_iterator i;
#define PARSE_ATTR(x, y) if ((i = attr.find(#x)) != attr.end()) setAttribute(i->first, i->second);
#define PARSE_int(x)    PARSE_ATTR(x, int)
#define PARSE_bool(x)   PARSE_ATTR(x, bool)
#define PARSE_string(x) PARSE_ATTR(x, DOMString)
#define PARSE_ATTR_END }

struct HTMLElement : public Element {
  HTMLElement(int et, Document *doc) : Element(doc) { htmlElementType=et; }
  PARSE_ATTR_BEGIN(Element);
  PARSE_string(accesskey); PARSE_string(class); PARSE_string(contenteditable); PARSE_string(contextmenu);
  PARSE_string(dir); PARSE_string(draggable); PARSE_string(dropzone); PARSE_string(hidden); PARSE_string(id);
  PARSE_string(inert); PARSE_string(itemid); PARSE_string(itemprop); PARSE_string(itemref);
  PARSE_string(itemscope); PARSE_string(itemtype); PARSE_string(lang); PARSE_string(spellcheck); 
  PARSE_string(style); PARSE_string(tabindex); PARSE_string(title); PARSE_string(translate);
  PARSE_ATTR_END;
};

struct HTMLFormElement : public HTMLElement { DERIVE_ELEMENT(HTMLFormElement, HTML_FORM_ELEMENT) {}
  DECLARE_TAG_NAMES("form");
  HTMLCollection elements;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(name); PARSE_string(acceptcharset); PARSE_string(action); PARSE_string(enctype);
  PARSE_string(method); PARSE_string(target);
  PARSE_ATTR_END;

  long length() const { return elements.length(); }
  void submit();
  void reset();
};

struct HTMLSelectElement : public HTMLElement { DERIVE_ELEMENT(HTMLSelectElement, HTML_SELECT_ELEMENT), form(0) {}
  DECLARE_TAG_NAMES("select", "keygen");
  HTMLCollection options;
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(type); PARSE_string(value); PARSE_string(name); PARSE_int(selectedindex); PARSE_int(size);
  PARSE_int(tabindex); PARSE_bool(disabled); PARSE_bool(multiple);
  PARSE_ATTR_END;

  long length() const { return options.length(); }        
  void add(HTMLElement *element, HTMLElement *before) {}
  void remove(long index) {}
  void blur() {}
  void focus() {}
};

struct HTMLOptionElement : public HTMLElement { DERIVE_ELEMENT(HTMLOptionElement, HTML_OPTION_ELEMENT), form(0) {}
  DECLARE_TAG_NAMES("option");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(text); PARSE_string(label); PARSE_string(value); PARSE_int(index); PARSE_bool(defaultselected);
  PARSE_int(disabled); PARSE_bool(selected);
  PARSE_ATTR_END;
};

struct HTMLInputElement : public HTMLElement { DERIVE_ELEMENT(HTMLInputElement, HTML_INPUT_ELEMENT), form(0), image_tex(0) {}
  DECLARE_TAG_NAMES("input");
  HTMLFormElement *form;
  unique_ptr<TilesTextGUI> text;
  shared_ptr<Texture> image_tex;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(defaultvalue); PARSE_string(accept); PARSE_string(accesskey); PARSE_string(align);
  PARSE_string(alt); PARSE_string(name); PARSE_string(size); PARSE_string(src); PARSE_string(usemap);
  PARSE_string(value); PARSE_string(type); PARSE_bool(defaultchecked); PARSE_bool(checked);
  PARSE_bool(disabled); PARSE_bool(tabindex); PARSE_bool(readonly); PARSE_int(maxlength);
  PARSE_ATTR_END;

  void blur();
  void focus();
  void select();
  void click();
};

struct HTMLTextAreaElement : public HTMLElement { DERIVE_ELEMENT(HTMLTextAreaElement, HTML_TEXT_AREA_ELEMENT), form(0) {}
  DECLARE_TAG_NAMES("textarea");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(defaultvalue); PARSE_string(accesskey); PARSE_string(name); PARSE_string(type);
  PARSE_string(value); PARSE_bool(disabled); PARSE_bool(readonly); PARSE_int(cols); PARSE_int(rows);
  PARSE_int(tabIndex);
  PARSE_ATTR_END;

  void blur();
  void focus();
  void select();
};

struct HTMLButtonElement : public HTMLElement { DERIVE_ELEMENT(HTMLButtonElement, HTML_BUTTON_ELEMENT), form(0) {}
  DECLARE_TAG_NAMES("button");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(accessKey); PARSE_string(name); PARSE_string(type); PARSE_string(value);
  PARSE_bool(disabled); PARSE_int(tabindex);
  PARSE_ATTR_END;
};

struct HTMLAnchorElement : public HTMLElement { DERIVE_ELEMENT(HTMLAnchorElement, HTML_ANCHOR_ELEMENT) {}
  DECLARE_TAG_NAMES("a");

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(accesskey); PARSE_string(charset); PARSE_string(coords); PARSE_string(href);
  PARSE_string(hreflang); PARSE_string(name); PARSE_string(rel); PARSE_string(rev); PARSE_string(shape);
  PARSE_string(target); PARSE_string(type);
  PARSE_ATTR_END;

  long tabIndex;
  void focus();
  void blur();
};

struct HTMLImageElement : public HTMLElement { DERIVE_ELEMENT(HTMLImageElement, HTML_IMAGE_ELEMENT), tex(0) {}
  DECLARE_TAG_NAMES("img", "image");
  shared_ptr<Texture> tex;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(lowsrc); PARSE_string(name); PARSE_string(align); PARSE_string(alt); PARSE_string(border);
  PARSE_string(height); PARSE_string(hspace); PARSE_string(longdesc); PARSE_string(src); PARSE_string(useMap);
  PARSE_string(vspace); PARSE_string(width); PARSE_bool(ismap);
  PARSE_ATTR_END;
};

struct HTMLObjectElement : public HTMLElement { DERIVE_ELEMENT(HTMLObjectElement, HTML_OBJECT_ELEMENT), form(0) {}
  DECLARE_TAG_NAMES("object");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(code); PARSE_string(align); PARSE_string(archive); PARSE_string(border); PARSE_string(codebase);
  PARSE_string(codetype); PARSE_string(data); PARSE_string(height); PARSE_string(hspace); PARSE_string(name);
  PARSE_string(standby); PARSE_string(type); PARSE_string(usemap); PARSE_string(vspace); PARSE_string(width);
  PARSE_int(tabindex); PARSE_bool(declare);
  PARSE_ATTR_END;
};

struct HTMLTableCaptionElement : public HTMLElement {
  HTMLTableElement *parentTable;
  HTMLTableCaptionElement(Document *doc, HTMLTableElement *par=0) : HTMLElement(HTML_TABLE_CAPTION_ELEMENT, doc), parentTable(par) {}
  DERIVE_ELEMENT_IMPL(HTMLTableCaptionElement, HTML_TABLE_CAPTION_ELEMENT);
  DECLARE_TAG_NAMES("caption");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align);
  PARSE_ATTR_END;
};

struct HTMLTableCellElement : public HTMLElement {
  HTMLTableRowElement *parentRow;
  HTMLTableCellElement(Document *doc, HTMLTableRowElement *par=0) : HTMLElement(HTML_TABLE_CELL_ELEMENT, doc), parentRow(par) {}
  DERIVE_ELEMENT_IMPL(HTMLTableCellElement, HTML_TABLE_CELL_ELEMENT);
  DECLARE_TAG_NAMES("td", "th");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_int(cellindex); PARSE_int(colspan); PARSE_int(rowspan); PARSE_string(abbr); PARSE_string(align);
  PARSE_string(axis); PARSE_string(bgcolor); PARSE_string(ch); PARSE_string(choff); PARSE_string(headers);
  PARSE_string(height); PARSE_string(scope); PARSE_string(valign); PARSE_string(width); PARSE_bool(nowrap);
  PARSE_ATTR_END;
};

struct HTMLTableColElement : public HTMLElement { DERIVE_ELEMENT(HTMLTableColElement, HTML_TABLE_COL_ELEMENT) {}
  DECLARE_TAG_NAMES("col", "colgroup");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(char_); PARSE_string(charOff); PARSE_int(span); PARSE_string(valign); PARSE_string(width);
  PARSE_ATTR_END;
};

struct HTMLTableRowElement : public HTMLElement {
  HTMLCollection cells;
  HTMLTableElement *parentTable;
  HTMLTableRowElement(Document *doc, HTMLTableElement *par=0) : HTMLElement(HTML_TABLE_ROW_ELEMENT, doc), parentTable(par) {}
  DERIVE_ELEMENT_IMPL(HTMLTableRowElement, HTML_TABLE_ROW_ELEMENT);

  DECLARE_TAG_NAMES("tr");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(bgcolor); PARSE_string(ch); PARSE_string(choff); PARSE_string(valign);
  PARSE_int(rowIndex); PARSE_int(sectionrowindex);
  PARSE_ATTR_END;

  void deleteCell(long index) { cells.EraseAt(index); }
  HTMLElement *insertCell(long index) { return (HTMLElement*)cells.InsertAt(index, AllocatorNew(ownerDocument->alloc, (HTMLTableCellElement), (ownerDocument, this))); }
};

struct HTMLTableSectionElement : public HTMLElement {
  HTMLTableElement *parentTable;
  HTMLTableSectionElement(Document *doc, HTMLTableElement *par=0) : HTMLElement(HTML_TABLE_SECTION_ELEMENT, doc), parentTable(par) {}
  DERIVE_ELEMENT_IMPL(HTMLTableSectionElement, HTML_TABLE_SECTION_ELEMENT);
  DECLARE_TAG_NAMES("thead", "tbody", "tfoot");

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(ch); PARSE_string(choff); PARSE_string(valign);
  PARSE_ATTR_END;

  void deleteRow(long index);
  HTMLElement *insertRow(long index);
};

#define HTMLTableIterColGroup(t) for (int j=0, cols = (t)->colgroup ? (t)->colgroup->childNodes.length() : 0; j<cols; j++) if (LFL::DOM::HTMLTableColElement *col = n->colgroup->childNodes.item(j)->AsHTMLTableColElement())
#define HTMLTableRowIter(t) for (int i=0; i<(t)->rows .length(); i++) if (LFL::DOM::HTMLTableRowElement  *row  = (t)->rows .item(i)->AsHTMLTableRowElement())
#define HTMLTableColIter(t) for (int j=0; j<(t)->cells.length(); j++) if (LFL::DOM::HTMLTableCellElement *cell = (t)->cells.item(j)->AsHTMLTableCellElement())
#define HTMLTableIter(t) HTMLTableRowIter(t) HTMLTableColIter(row)
struct HTMLTableElement : public HTMLElement { DERIVE_ELEMENT(HTMLTableElement, HTML_TABLE_ELEMENT), tHead(0), tFoot(0), tBodies(0), caption(0), colgroup(0) {}
  DECLARE_TAG_NAMES("table");
  HTMLCollection rows;
  HTMLTableSectionElement *tHead, *tFoot, *tBodies;
  HTMLTableCaptionElement *caption;
  HTMLTableColElement *colgroup;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(bgcolor); PARSE_string(border); PARSE_string(cellpadding); PARSE_string(cellspacing);
  PARSE_string(frame); PARSE_string(rules); PARSE_string(summary); PARSE_string(width);
  PARSE_ATTR_END;

  void deleteTHead() { tHead=0; }
  void deleteTFoot() { tFoot=0; }
  void deleteTBody() { tBodies=0; }
  void deleteCaption() { caption=0; }
  void deleteRow(long index) { rows.EraseAt(index); }
  HTMLElement *createTHead   () { if (!tHead)   tHead    = AllocatorNew(ownerDocument->alloc, (HTMLTableSectionElement), (ownerDocument, this)); return tHead; }
  HTMLElement *createTFoot   () { if (!tFoot)   tFoot    = AllocatorNew(ownerDocument->alloc, (HTMLTableSectionElement), (ownerDocument, this)); return tFoot; }
  HTMLElement *createTBody   () { if (!tBodies) tBodies  = AllocatorNew(ownerDocument->alloc, (HTMLTableSectionElement), (ownerDocument, this)); return tBodies; }
  HTMLElement *createCaption () { if (!caption) caption  = AllocatorNew(ownerDocument->alloc, (HTMLTableCaptionElement), (ownerDocument, this)); return caption; }
  HTMLElement *createColGroup() { if (!caption) colgroup = AllocatorNew(ownerDocument->alloc, (HTMLTableColElement),     (ownerDocument));       return colgroup; }
  HTMLElement *insertRow(long index) { return (HTMLElement*)rows.InsertAt(index, AllocatorNew(ownerDocument->alloc, (HTMLTableRowElement), (ownerDocument, this))); }
};

struct HTMLHtmlElement : public HTMLElement { DERIVE_ELEMENT(HTMLHtmlElement, HTML_HTML_ELEMENT) {}
  DECLARE_TAG_NAMES("html");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(version);
  PARSE_ATTR_END;
};

struct HTMLHeadElement : public HTMLElement { DERIVE_ELEMENT(HTMLHeadElement, HTML_HEAD_ELEMENT) {}
  DECLARE_TAG_NAMES("head");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(profile);
  PARSE_ATTR_END;
};

struct HTMLOptGroupElement : public HTMLElement { DERIVE_ELEMENT(HTMLOptGroupElement, HTML_OPT_GROUP_ELEMENT) {}
  DECLARE_TAG_NAMES("optgroup");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(disabled); PARSE_string(label);
  PARSE_ATTR_END;
};

struct HTMLStyleElement : public HTMLElement { DERIVE_ELEMENT(HTMLStyleElement, HTML_STYLE_ELEMENT) {}
  DECLARE_TAG_NAMES("style");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(disabled); PARSE_string(media); PARSE_string(type);
  PARSE_ATTR_END;
};

struct HTMLLinkElement : public HTMLElement { DERIVE_ELEMENT(HTMLLinkElement, HTML_LINK_ELEMENT) {}
  DECLARE_TAG_NAMES("link");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(disabled); PARSE_string(charset); PARSE_string(href); PARSE_string(hreflang); PARSE_string(media);
  PARSE_string(rel); PARSE_string(rev); PARSE_string(target); PARSE_string(type);
  PARSE_ATTR_END;
};

struct HTMLTitleElement : public HTMLElement { DERIVE_ELEMENT(HTMLTitleElement, HTML_TITLE_ELEMENT) {}
  DECLARE_TAG_NAMES("title");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(text);
  PARSE_ATTR_END;
};

struct HTMLMetaElement : public HTMLElement { DERIVE_ELEMENT(HTMLMetaElement, HTML_META_ELEMENT) {}
  DECLARE_TAG_NAMES("meta");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(content); PARSE_string(http-equiv); PARSE_string(name); PARSE_string(scheme);
  PARSE_ATTR_END;
};

struct HTMLBaseElement : public HTMLElement { DERIVE_ELEMENT(HTMLBaseElement, HTML_BASE_ELEMENT) {}
  DECLARE_TAG_NAMES("base");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(href); PARSE_string(target);
  PARSE_ATTR_END;
};

struct HTMLIsIndexElement : public HTMLElement { DERIVE_ELEMENT(HTMLIsIndexElement, HTML_IS_INDEX_ELEMENT) {}
  DECLARE_TAG_NAMES("isindex");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(prompt);
  PARSE_ATTR_END;
};

struct HTMLBodyElement : public HTMLElement { DERIVE_ELEMENT(HTMLBodyElement, HTML_BODY_ELEMENT) {}
  DECLARE_TAG_NAMES("body");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(alink); PARSE_string(background); PARSE_string(bgcolor); PARSE_string(link); PARSE_string(text);
  PARSE_string(vlink);
  PARSE_ATTR_END;

  string HTML4PseudoClassStyle() {
    string ret;
    if (Node *a = getAttributeNode("alink")) StrAppend(&ret, "a:active  { color: ", a->nodeValue(), "; }\n");
    if (Node *a = getAttributeNode("vlink")) StrAppend(&ret, "a:visited { color: ", a->nodeValue(), "; }\n");
    if (Node *a = getAttributeNode("link"))  StrAppend(&ret, "a:link    { color: ", a->nodeValue(), "; }\n");
    return ret;
  }
};

struct HTMLLabelElement : public HTMLElement { DERIVE_ELEMENT(HTMLLabelElement, HTML_LABEL_ELEMENT) {}
  DECLARE_TAG_NAMES("label");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(accesskey); PARSE_string(htmlfor);
  PARSE_ATTR_END;
};

struct HTMLFieldSetElement : public HTMLElement { DERIVE_ELEMENT(HTMLFieldSetElement, HTML_FIELD_SET_ELEMENT) {}
  DECLARE_TAG_NAMES("fieldset");
  HTMLFormElement *form;
};

struct HTMLLegendElement : public HTMLElement { DERIVE_ELEMENT(HTMLLegendElement, HTML_LEGEND_ELEMENT) {}
  DECLARE_TAG_NAMES("legend");
  HTMLFormElement *form;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(accesskey); PARSE_string(align);
  PARSE_ATTR_END;
};

struct HTMLUListElement : public HTMLElement { DERIVE_ELEMENT(HTMLUListElement, HTML_U_LIST_ELEMENT) {}
  DECLARE_TAG_NAMES("ul");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(compact); PARSE_string(type);
  PARSE_ATTR_END;
};

struct HTMLOListElement : public HTMLElement { DERIVE_ELEMENT(HTMLOListElement, HTML_O_LIST_ELEMENT) {}
  DECLARE_TAG_NAMES("ol");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(compact); PARSE_int(start); PARSE_string(type);
  PARSE_ATTR_END;
};

struct HTMLDListElement : public HTMLElement { DERIVE_ELEMENT(HTMLDListElement, HTML_D_LIST_ELEMENT) {}
  DECLARE_TAG_NAMES("dl");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(compact);
  PARSE_ATTR_END;
};

struct HTMLDirectoryElement : public HTMLElement { DERIVE_ELEMENT(HTMLDirectoryElement, HTML_DIRECTORY_ELEMENT) {}
  DECLARE_TAG_NAMES("dir");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(compact);
  PARSE_ATTR_END;
};

struct HTMLMenuElement : public HTMLElement { DERIVE_ELEMENT(HTMLMenuElement, HTML_MENU_ELEMENT) {}
  DECLARE_TAG_NAMES("menu");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_bool(compact);
  PARSE_ATTR_END;
};

struct HTMLLIElement : public HTMLElement { DERIVE_ELEMENT(HTMLLIElement, HTML_LI_ELEMENT) {}
  DECLARE_TAG_NAMES("li");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(type); PARSE_int(value);
  PARSE_ATTR_END;
};

struct HTMLDivElement : public HTMLElement { DERIVE_ELEMENT(HTMLDivElement, HTML_DIV_ELEMENT) {}
  DECLARE_TAG_NAMES("div");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align);
  PARSE_ATTR_END;
};

struct HTMLParagraphElement : public HTMLElement { DERIVE_ELEMENT(HTMLParagraphElement, HTML_PARAGRAPH_ELEMENT) {}
  DECLARE_TAG_NAMES("p");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align);
  PARSE_ATTR_END;
};

struct HTMLHeadingElement : public HTMLElement { DERIVE_ELEMENT(HTMLHeadingElement, HTML_HEADING_ELEMENT) {}
  DECLARE_TAG_NAMES("h1", "h2", "h3", "h4", "h5", "h6");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align);
  PARSE_ATTR_END;
};

struct HTMLQuoteElement : public HTMLElement { DERIVE_ELEMENT(HTMLQuoteElement, HTML_QUOTE_ELEMENT) {}
  DECLARE_TAG_NAMES("q");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(cite);
  PARSE_ATTR_END;
};

struct HTMLPreElement : public HTMLElement { DERIVE_ELEMENT(HTMLPreElement, HTML_PRE_ELEMENT) {}
  DECLARE_TAG_NAMES("pre");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_int(width);
  PARSE_ATTR_END;
};

struct HTMLBRElement : public HTMLElement { DERIVE_ELEMENT(HTMLBRElement, HTML_BR_ELEMENT) {}
  DECLARE_TAG_NAMES("br");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(clear);
  PARSE_ATTR_END;
};

struct HTMLBaseFontElement : public HTMLElement { DERIVE_ELEMENT(HTMLBaseFontElement, HTML_BASE_FONT_ELEMENT) {}
  DECLARE_TAG_NAMES("basefont");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(color); PARSE_string(face); PARSE_string(size);
  PARSE_ATTR_END;
};

struct HTMLFontElement : public HTMLElement { DERIVE_ELEMENT(HTMLFontElement, HTML_FONT_ELEMENT) {}
  DECLARE_TAG_NAMES("font");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(color); PARSE_string(face); PARSE_string(size);
  PARSE_ATTR_END;
};

struct HTMLHRElement : public HTMLElement { DERIVE_ELEMENT(HTMLHRElement, HTML_HR_ELEMENT) {}
  DECLARE_TAG_NAMES("hr");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(size); PARSE_string(width); PARSE_bool(noshade);
  PARSE_ATTR_END;
};

struct HTMLModElement : public HTMLElement { DERIVE_ELEMENT(HTMLModElement, HTML_MOD_ELEMENT) {}
  DECLARE_TAG_NAMES("del", "ins");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(cite); PARSE_string(datetime);
  PARSE_ATTR_END;
};

struct HTMLParamElement : public HTMLElement { DERIVE_ELEMENT(HTMLParamElement, HTML_PARAM_ELEMENT) {}
  DECLARE_TAG_NAMES("param");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(name); PARSE_string(type); PARSE_string(value); PARSE_string(valuetype);
  PARSE_ATTR_END;
};

struct HTMLAppletElement : public HTMLElement { DERIVE_ELEMENT(HTMLAppletElement, HTML_APPLET_ELEMENT) {}
  DECLARE_TAG_NAMES("applet");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(alt); PARSE_string(archive); PARSE_string(code); PARSE_string(codebase);
  PARSE_string(height); PARSE_string(hspace); PARSE_string(name); PARSE_string(object); PARSE_string(vspace);
  PARSE_string(width);
  PARSE_ATTR_END;
};

struct HTMLMapElement : public HTMLElement { DERIVE_ELEMENT(HTMLMapElement, HTML_MAP_ELEMENT) {}
  DECLARE_TAG_NAMES("map");
  HTMLCollection areas;

  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(name);
  PARSE_ATTR_END;
};

struct HTMLAreaElement : public HTMLElement { DERIVE_ELEMENT(HTMLAreaElement, HTML_AREA_ELEMENT) {}
  DECLARE_TAG_NAMES("area");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(accesskey); PARSE_string(alt); PARSE_string(coords); PARSE_string(href); PARSE_string(shape);
  PARSE_string(target); PARSE_int(tabindex); PARSE_bool(nohref);
  PARSE_ATTR_END;
};

struct HTMLScriptElement : public HTMLElement { DERIVE_ELEMENT(HTMLScriptElement, HTML_SCRIPT_ELEMENT) {}
  DECLARE_TAG_NAMES("script");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(text); PARSE_string(htmlfor); PARSE_string(event); PARSE_string(charset); PARSE_string(defer);
  PARSE_string(src); PARSE_string(type);
  PARSE_ATTR_END;
};

struct HTMLFrameSetElement : public HTMLElement { DERIVE_ELEMENT(HTMLFrameSetElement, HTML_FRAME_SET_ELEMENT) {}
  DECLARE_TAG_NAMES("frameset");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(cols); PARSE_string(rows);
  PARSE_ATTR_END;
};

struct HTMLFrameElement : public HTMLElement { DERIVE_ELEMENT(HTMLFrameElement, HTML_FRAME_ELEMENT) {}
  DECLARE_TAG_NAMES("frame");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(frameborder); PARSE_string(longdesc); PARSE_string(marginheight); PARSE_string(marginwidth);
  PARSE_string(name); PARSE_string(scrolling); PARSE_string(src); PARSE_bool(noresize);
  PARSE_ATTR_END;
};

struct HTMLIFrameElement : public HTMLElement { DERIVE_ELEMENT(HTMLIFrameElement, HTML_I_FRAME_ELEMENT) {}
  DECLARE_TAG_NAMES("iframe");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_string(align); PARSE_string(frameborder); PARSE_string(height); PARSE_string(longdesc);
  PARSE_string(marginheight); PARSE_string(marginwidth); PARSE_string(name); PARSE_string(scrolling);
  PARSE_string(src); PARSE_string(width);
  PARSE_ATTR_END;
};

struct HTMLBlockQuoteElement : public HTMLElement { DERIVE_ELEMENT(HTMLBlockQuoteElement, HTML_BLOCK_QUOTE_ELEMENT) {}
  DECLARE_TAG_NAMES("blockquote");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_ATTR_END;
};

struct HTMLCanvasElement : public HTMLElement { DERIVE_ELEMENT(HTMLCanvasElement, HTML_CANVAS_ELEMENT) {}
  DECLARE_TAG_NAMES("canvas");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_ATTR_END;
};

struct HTMLSpanElement : public HTMLElement { DERIVE_ELEMENT(HTMLSpanElement, HTML_SPAN_ELEMENT) {}
  DECLARE_TAG_NAMES("span");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_ATTR_END;
};

struct HTMLEmbedElement : public HTMLElement { DERIVE_ELEMENT(HTMLEmbedElement, HTML_EMBED_ELEMENT) {}
  DECLARE_TAG_NAMES("embed");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_ATTR_END;
};

struct HTMLMarqueeElement : public HTMLElement { DERIVE_ELEMENT(HTMLMarqueeElement, HTML_MARQUEE_ELEMENT) {}
  DECLARE_TAG_NAMES("marquee");
  PARSE_ATTR_BEGIN(HTMLElement);
  PARSE_ATTR_END;
};

struct HTMLDocument : public Document {
  HTMLDocument(DocumentParser *p, Allocator *a) : Document(p, a) {}
  DOMString title, referrer, domain, URL, baseURI, cookie;
  HTMLElement *body;
  HTMLCollection images, applets, links, forms, anchors;

  virtual HTMLDocument* AsHTMLDocument() { return this; }
  void setURL(const string &v) { URL = v; }
  void setDomain(const string &v) { domain = v; }
  void setBaseURI(const string &v) { baseURI = v; }
  void open() {}
  void close() {}
  void write(const DOMString &text) {}
  void writeln(const DOMString &text) {}
  Element *getElementById(const DOMString &elementId) { return NULL; }
  NodeList *getElementsByName(const DOMString &elementName) { return NULL; }

# undef  XX
# define XX(n) HTMLElement *Create ## n () { return AllocatorNew(alloc, (HTML ## n), (this)); }
# include "web/html_elements.h"

  typedef HTMLElement *(HTMLDocument::*CreateElementCB)();
  struct CreateElementCallbacks : public unordered_map<string, CreateElementCB> {
    CreateElementCallbacks() {
#     undef  XX
#     define XX(n) for (const char **tn = HTML ## n::TagNames(); *tn; tn++) (*this)[*tn] = &HTMLDocument::Create ## n;
#     include "web/html_elements.h"
    }
  };

  virtual Element *createElement(const DOMString &name) {
    static CreateElementCallbacks cb;
    CreateElementCB create_cb = FindOrNull(cb, String::ToAscii(name));
    Element *ret = create_cb ? (this->*create_cb)() : AllocatorNew(ownerDocument->alloc, (HTMLElement), (0, ownerDocument));
    ret->tagName = name;
    return ret;
  }
};

template <class X> struct ObjectWrapperAlloc : public X {
  vector<void*> allocs;
  virtual ~ObjectWrapperAlloc() { Reset(); }
  virtual void *Malloc(int n)           { void *ret = X::Malloc(n);     allocs.push_back(ret); return ret; }
  virtual void *Realloc(void *p, int n) { void *ret = X::Realloc(p, n); allocs.push_back(ret); return ret; }
  virtual void Reset() {
    for (vector<void*>::iterator i = allocs.begin(); i != allocs.end(); i++) ((LFL::DOM::Object*)*i)->~Object();
    X::Reset();
    allocs.clear();
  }
};

template <int X> struct FixedObjectAlloc : public ObjectWrapperAlloc<FixedAlloc<X> > {};
struct BlockChainObjectAlloc : public ObjectWrapperAlloc<BlockChainAlloc> {
  BlockChainObjectAlloc(int s) { block_size = s; }
};

}; // namespace DOM
typedef DOM::DOMString DOMString;
}; // namespace LFL
#endif // LFL_LFAPP_DOM_H__
