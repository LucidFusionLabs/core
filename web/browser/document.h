/*
 * $Id$
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

#ifndef LFL_CORE_WEB_DOCUMENT_H__
#define LFL_CORE_WEB_DOCUMENT_H__
namespace LFL {

struct DocumentParser {
  struct Parser {
    string url;
    DocumentParser *parent;
    Parser(DocumentParser *P, const string &URL) : url(URL), parent(P) {}
    virtual ~Parser() {}
    virtual void Complete(void *self) {
      if (parent->outstanding.erase(self)) {
        parent->completed++;
        if (auto html = parent->doc->DocElement()) html->SetLayoutDirty();
      }
      delete this;
    }
  };

  struct StyleParser : public Parser {
    string charset;
    StyleSheet *style_target;
    StyleParser(DocumentParser *p, const string &url, const DOM::DOMString &cs) : Parser(p, url), charset(String::ToUTF8(cs)),
    style_target(new StyleSheet(parent->doc->node, "", charset.empty() ? "UTF-8" : charset.c_str(), 0, 0)) {
      p->doc->style_sheet.push_back(unique_ptr<StyleSheet>(style_target));
    }

    virtual void Complete(void *self) { 
      if (parent->Running(self)) {
        style_target->Done();
        parent->doc->node->style_context->AppendSheet(style_target);
        if (auto html = parent->doc->DocElement()) html->SetStyleDirty();
      }
      Parser::Complete(self);
    }

    int WGetResponseCB(Connection*, const char *h, const string &ct, const char *cb, int cl) {
      if (!h && (!cb || !cl)) { Complete(this); return 1; }
      else if (!h) style_target->Parse(string(cb, cl));
      return 0;
    }
  };

  struct HTMLParser : public ::LFL::HTMLParser, public StyleParser {
    vector<DOM::Node*> target;
    HTMLParser(DocumentParser *p, const string &url, DOM::Node *t) : 
      ::LFL::HTMLParser(p->doc->dispatch), StyleParser(p, url, "") { target.push_back(t); }

    void ParseChunkCB(const char *content, int content_len) { if (auto html = parent->doc->DocElement()) html->SetLayoutDirty(); }
    void WGetContentBegin(Connection *c, const char *h, const string &ct) { parent->doc->content_type=ct; }
    void WGetContentEnd(Connection *c) { StyleParser::Complete(this); }
    void CloseTag(const String &tag, const KV &attr, const TagStack &stack) {
      if (target.size() && tag == target.back()->nodeName()) target.pop_back();
    }

    void OpenTag(const String &input_tag, const KV &attr, const TagStack &stack) {
      string tag = LFL::String::ToAscii(input_tag);
      DOM::Node *parentNode = target.back();
      DOM::Element *addNode = 0;
      if (tag == "col") { CHECK(parentNode->AsHTMLTableColElement()); }
      if (tag == "td" || tag == "th") {
        DOM::HTMLTableRowElement *row = parentNode->AsHTMLTableRowElement();
        CHECK(row);
        addNode = row->insertCell(row->cells.length());
      } else if (tag == "tr") {
        DOM::HTMLTableElement        *table   = parentNode->AsHTMLTableElement();
        DOM::HTMLTableSectionElement *section = parentNode->AsHTMLTableSectionElement();
        CHECK(table || section);
        if (table) addNode = table  ->insertRow(table               ->rows.length());
        else       addNode = section->insertRow(section->parentTable->rows.length());
      } else if (HTMLParser::TableDependent(tag) && tag != "col") {
        DOM::HTMLTableElement *table = parentNode->AsHTMLTableElement();
        CHECK(table);
        if      (tag == "thead")    addNode = table->createTHead();
        else if (tag == "tbody")    addNode = table->createTBody();
        else if (tag == "tfoot")    addNode = table->createTFoot();
        else if (tag == "caption")  addNode = table->createCaption();
        else if (tag == "colgroup") addNode = table->createColGroup();
      } else {
        addNode = parentNode->document()->createElement(tag);
      }
      if (!addNode) return;
      addNode->tagName = tag;
      addNode->ParseAttributes(attr);
      addNode->AttachRender();
      parentNode->appendChild(addNode);
      target.push_back(addNode);
      if (tag == "body") {
        string pseudo_style = addNode->AsHTMLBodyElement()->HTML4PseudoClassStyle();
        if (!pseudo_style.empty() && style_target) style_target->Parse(pseudo_style);
      } else if (tag == "img" || (tag == "input" && StringEquals(addNode->getAttribute("type"), "image"))) {
        parent->Open(LFL::String::ToUTF8(addNode->getAttribute("src")), addNode);
      } else if (tag == "link") {
        if (StringEquals       (addNode->getAttribute("rel"),   "stylesheet") &&
            StringEmptyOrEquals(addNode->getAttribute("media"), "screen", "all")) {
          parent->Open(LFL::String::ToUTF8(addNode->getAttribute("href")), addNode);
        }
      }
    }

    void Text(const String &input_text, const TagStack &stack) {
      if (!parent->Running(this)) return;
      DOM::Node *parentNode = target.back();
      DOM::Text *ret = parentNode->document()->createTextNode(DOM::DOMString());
      ret->AttachRender();
      parentNode->appendChild(ret);

      if (!inpre) {
        for (const char *p = text.data(), *e = p + text.size(); p < e; /**/) {
          if (isspace(*p)) {
            ret->appendData(DOM::StringPiece::Space());
            while (p<e && isspace(*p)) p++;
          } else {
            const char *b = p;
            while (p<e && !isspace(*p)) p++;
            ret->appendData(HTMLParser::ReplaceEntitySymbols(string(b, p-b)));
          }
        }
      } else ret->appendData(text);
    }

    void Script(const String &text, const TagStack &stack) {}
    void Style(const String &text, const TagStack &stack) {
      if (style_target) style_target->Parse(LFL::String::ToUTF8(text));
    }
  };

  struct ImageParser : public Parser {
    shared_ptr<Texture> target;
    string content;
    int content_length=0;
    ProcessAPIClient::TextureCB loadasset_cb;
    ProcessAPIServer::LoadTextureIPC::CB loadtex_cb;
    ImageParser(DocumentParser *p, const string &url, const shared_ptr<Texture> &t) :
      Parser(p, url), target(t), loadasset_cb(bind(&ImageParser::LoadAssetResponseCB, this, _1)),
      loadtex_cb(bind(&ImageParser::LoadTextureResponseCB, this, _1, _2)) {}

    int WGetResponseCB(Connection*, const char *h, const string &ct, const char *cb, int cl) {
      if      (h)                  content_length = cl;
      if      (!h && (!cb || !cl)) { WGetComplete(ct); return 1; }
      else if (!h)                 content.append(cb, cl);
      return 0;
    }

    void WGetComplete(const string &content_type) {
      if (!content.empty() && content_length == content.size() && parent->Running(this)) {
        string fn = BaseName(url);
        if      (MIMEType::Jpg(content_type) && !FileSuffix::Jpg(fn)) fn += ".jpg";
        else if (MIMEType::Png(content_type) && !FileSuffix::Png(fn)) fn += ".png";
        else if (PrefixMatch(content_type, "text/html")) INFO("ImageParser content='", content, "'");

        auto dispatch = parent->doc->dispatch;
        if (dispatch->render_process) {
          dispatch->render_process->LoadAsset(content, fn, loadasset_cb);
          return;
        } else if (dispatch->main_process) {
          parent->doc->loader->LoadTexture(content.data(), fn.c_str(), content.size(), target.get(), 0);
          dispatch->main_process->LoadTexture(target.get(), loadtex_cb);
          return;
        } else {
          parent->doc->loader->LoadTexture(content.data(), fn.c_str(), content.size(), target.get());
          INFO("ImageParser ", content_type, ": ", url, " ", fn, " ", content.size(), " ", target->width, " ", target->height);
        }
      }
      Parser::Complete(this);
    }

    void LoadAssetResponseCB(const MultiProcessTextureResource &tex) {
      if (tex.width && tex.height) { target->LoadGL(tex); target->owner = true; }
      parent->doc->dispatch->RunInNetworkThread([=]() { Parser::Complete(this); });
    }

    int LoadTextureResponseCB(const IPC::LoadTextureResponse *res, Void) {
      if (res) target.get()->ID.v = res->tex_id();
      Parser::Complete(this);
      return IPC::Done;
    }
  };

  StringCB redirect_cb;
  Browser::Document *doc;
  int requested=0, completed=0;
  unordered_set<void*> outstanding;
  LRUCache<string, shared_ptr<Texture> > image_cache;
  GraphicsDeviceHolder *gd;
  SocketServices *net;
  DocumentParser(GraphicsDeviceHolder *G, SocketServices *N, Browser::Document *D) :
    doc(D), image_cache(64), gd(G), net(N) {}

  bool Running(void *h) const { return outstanding.find(h) != outstanding.end(); }
  void Clear() { requested=completed=0; outstanding.clear(); doc->Clear(); }

  void OpenHTML(const string &content) {
    Clear();
    HTMLParser *html_handler = new HTMLParser(this, "", doc->node);
    outstanding.insert(html_handler);
    html_handler->WGetCB(0, 0, string(), content.c_str(), content.size());
    html_handler->WGetCB(0, 0, string(), 0,               0);
  }

  void OpenFrame(const string &url, DOM::Frame*) {
    Clear();
    doc->node->setURL(url);
    doc->node->setDomain(HTTP::HostURL(url.c_str()).c_str());
    string urldir = url.substr(0, DirNameLen(url, 1));
    doc->node->setBaseURI((urldir.size() >= 3 && urldir.substr(urldir.size()-3) == "://") ? url : urldir);
    Open(url, doc->node);
  }

  void Open(const string &input_url, DOM::Node *target) {
    if (input_url.empty()) return;
    string url = input_url;
    bool data_url = PrefixMatch(url, "data:");
    if (data_url) { /**/
    } else if (PrefixMatch(url, "/")) {
      if (PrefixMatch(url, "//")) url = "http:"                           + url;
      else if (doc->node)         url = String::ToUTF8(doc->node->domain) + url;
    } else if (url.substr(0, 8).find("://") == string::npos) {
      if (doc->node) url = StrCat(doc->node->baseURI, "/", url);
    }

    void *handler = 0;
    HTTPClient::ResponseCB callback;
    DOM::HTMLDocument     *html  = target->AsHTMLDocument();
    DOM::HTMLImageElement *image = target->AsHTMLImageElement();
    DOM::HTMLLinkElement  *link  = target->AsHTMLLinkElement();
    DOM::HTMLInputElement *input = target->AsHTMLInputElement();
    bool input_image = input && StringEquals(input->getAttribute("type"), "image");
    shared_ptr<Texture> *image_tex = input_image ? &input->image_tex : (image ? &image->tex : 0);

    if (image_tex) {
      CHECK_EQ(image_tex->get(), 0);
      if (shared_ptr<Texture> *cached = image_cache.Get(url)) { *image_tex = *cached; return; }
      *image_tex = make_shared<Texture>(gd);
      image_cache.EvictUnique();
      image_cache.Insert(url, *image_tex);
    }

    if (data_url) {
      int semicolon=url.find(";"), comma=url.find(",");
      if (semicolon != string::npos && comma != string::npos && semicolon < comma) {
        string mimetype=url.substr(5, semicolon-5), encoding=url.substr(semicolon+1, comma-semicolon-1);
        if (PrefixMatch(mimetype, "image/") && image_tex) {
          string fn = toconvert(mimetype, tochar<'/', '.'>);
          if (encoding == "base64") {
            string content = Singleton<Base64>::Set()->Decode(url.c_str()+comma+1, url.size()-comma-1);
            doc->loader->LoadTexture(content.data(), fn.c_str(), content.size(), image_tex->get());
          } else INFO("unhandled: data:url encoding ", encoding);
        } else INFO("unhandled data:url mimetype ", mimetype);
      } else INFO("unhandled data:url ", input_url);
      return;
    }

    if      (html)      { auto h = new HTMLParser (this, url, html);                          handler = h; callback = bind(&HTMLParser::WGetCB,          h, _1, _2, _3, _4, _5); }
    else if (image_tex) { auto h = new ImageParser(this, url, *image_tex);                    handler = h; callback = bind(&ImageParser::WGetResponseCB, h, _1, _2, _3, _4, _5); }
    else if (link)      { auto h = new StyleParser(this, url, link->getAttribute("charset")); handler = h; callback = bind(&StyleParser::WGetResponseCB, h, _1, _2, _3, _4, _5); }

    if (handler) {
      INFO("Browser open '", url, "'");
      requested++;
      outstanding.insert(handler);
      bool root = target == doc->node;
      auto dispatch = doc->dispatch;
      if (dispatch->main_process) dispatch->main_process->WGet(        url, callback, root ? redirect_cb : StringCB());
      else                                    HTTPClient::WGet(net, 0, url, 0, callback, root ? redirect_cb : StringCB());
    } else {
      ERROR("unknown mimetype for url ", url);
    }
  }

  shared_ptr<Texture> OpenImage(const string &url) {
    DOM::HTMLImageElement image(0);
    Open(url, &image);
    return image.tex;
  }

  void OpenStyleImport(const string &url) {
    DOM::HTMLLinkElement link(0);
    Open(url, &link);
  }
};

}; // namespace LFL
#endif // LFL_CORE_WEB_DOCUMENT_H__
