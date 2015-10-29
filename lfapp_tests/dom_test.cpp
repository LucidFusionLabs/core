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

#include "gtest/gtest.h"
#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/ipc.h"
#include "lfapp/browser.h"
#include "../crawler/html.h"
#include "../crawler/document.h"

namespace LFL {

TEST(DOMTest, DOMNode) {
    LFL::DOM::FixedObjectAlloc<65536> alloc;
    LFL::DOM::HTMLDocument *doc = AllocatorNew(&alloc, (LFL::DOM::HTMLDocument), (0, &alloc));
    LFL::DOM::Text A(doc), B(doc), C(doc), D(doc);
    EXPECT_EQ(0, doc->firstChild()); EXPECT_EQ(0, doc->lastChild());

    EXPECT_EQ(0, A.parentNode);
    doc->appendChild(&A);
    EXPECT_EQ(doc, A.parentNode);
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(0,  A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&A, doc->lastChild());

    EXPECT_EQ(0, B.parentNode);
    doc->appendChild(&B);
    EXPECT_EQ(doc, B.parentNode);
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(0,  B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&B, doc->lastChild());

    EXPECT_EQ(0, C.parentNode);
    doc->appendChild(&C);
    EXPECT_EQ(doc, C.parentNode);
    EXPECT_EQ(&B,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(&C, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    EXPECT_EQ(0, D.parentNode);
    doc->insertBefore(&D, &C);
    EXPECT_EQ(doc, D.parentNode);
    EXPECT_EQ(&B,  D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(&A,  B.previousSibling()); EXPECT_EQ(&D, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(&B, A.nextSibling());
    EXPECT_EQ(&A,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    doc->removeChild(&B);
    EXPECT_EQ(0, B.parentNode);
    EXPECT_EQ(&A, D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D, C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(0,  B.previousSibling()); EXPECT_EQ(0,  B.nextSibling());
    EXPECT_EQ(0,  A.previousSibling()); EXPECT_EQ(&D, A.nextSibling());
    EXPECT_EQ(&A, doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());

    doc->replaceChild(&B, &A);
    EXPECT_EQ(doc, B.parentNode);
    EXPECT_EQ(0,   A.parentNode);
    EXPECT_EQ(&B,  D.previousSibling()); EXPECT_EQ(&C, D.nextSibling());
    EXPECT_EQ(&D,  C.previousSibling()); EXPECT_EQ(0,  C.nextSibling());
    EXPECT_EQ(0,   B.previousSibling()); EXPECT_EQ(&D, B.nextSibling());
    EXPECT_EQ(0,   A.previousSibling()); EXPECT_EQ(0,  A.nextSibling());
    EXPECT_EQ(&B,  doc->firstChild());   EXPECT_EQ(&C, doc->lastChild());
}

TEST(DOMTest, DOMTree) {
    GUI sb_gui;
    Browser sb(&sb_gui, screen->Box());
    sb.doc.parser->OpenHTML("<html>\n"
                            "<head><style> h1 { background-color: #111111; } </style></head>\n"
                            "<body style=\"background-color: #00ff00\">\n"
                            "<H1 class=\"foo  bar\">Very header</h1>\n"
                            "<P id=cat>In  the  begining  was  fun.</p>\n"
                            "</body>\n");
    Box viewport = screen->Box();
    sb.v_scrollbar.menuicon = Fonts::Fake();
    sb.h_scrollbar.menuicon = Fonts::Fake();
    sb.Draw(viewport);
    CHECK(sb.doc.node);
    EXPECT_EQ("#document", String::ToUTF8(sb.doc.node->nodeName()));
    EXPECT_EQ(1, sb.doc.node->childNodes.length());
    EXPECT_EQ(0, sb.doc.node->AsHTMLImageElement());
    EXPECT_EQ(sb.doc.node, sb.doc.node->AsHTMLDocument());

    LFL::DOM::Node *html_node = sb.doc.node->firstChild();
    EXPECT_EQ("html", String::ToUTF8(html_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, html_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_HTML_ELEMENT, html_node->htmlElementType);
    LFL::DOM::HTMLElement *html_element = (LFL::DOM::HTMLElement*)html_node;
    EXPECT_EQ(html_node->AsElement(), html_element);
    EXPECT_EQ("html", String::ToUTF8(html_element->tagName));
    EXPECT_EQ(2, html_element->childNodes.length());
#ifdef LFL_LIBCSS
    void *vnode = 0; bool matched = 0; int count = 0;
    css_qname html_qn = { 0, 0 }, html_query = { 0, LibCSS_String::Intern("html") };
    css_qname foo_query = { 0, LibCSS_String::Intern("foo") };
    StyleContext::NodeName(0, html_node, &html_qn);
    EXPECT_EQ("html", LibCSS_String::ToUTF8String(html_qn.name));
    lwc_string **html_classes = 0; uint32_t n_html_classes = 0;
    StyleContext::NodeClasses(0, html_node, &html_classes, &n_html_classes);
    EXPECT_EQ(0, n_html_classes);
    EXPECT_EQ(0, html_classes);
    lwc_string *html_id = 0;
    StyleContext::NodeId(0, html_node, &html_id);
    EXPECT_EQ(0, html_id);
    StyleContext::NodeIsRoot(0, html_node, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NamedParentNode(0, html_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::NodeHasName(0, html_node, &html_query, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasName(0, html_node, &foo_query, &matched);
    EXPECT_EQ(0, matched);
    CHECK(html_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, html_node->render->style.Display().v);
#endif

    LFL::DOM::Node *head_node = html_element->firstChild();
    EXPECT_EQ("head", String::ToUTF8(head_node->nodeName()));

    LFL::DOM::Node *body_node = html_element->childNodes.item(1);
    EXPECT_EQ("body", String::ToUTF8(body_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, body_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_BODY_ELEMENT, body_node->htmlElementType);
    LFL::DOM::HTMLBodyElement *body_element = (LFL::DOM::HTMLBodyElement*)body_node;
    EXPECT_EQ(body_node->AsElement(), body_element);
    EXPECT_EQ(body_node->AsHTMLBodyElement(), body_element);
    EXPECT_EQ("body", String::ToUTF8(body_element->tagName));
    EXPECT_EQ("background-color: #00ff00", String::ToUTF8(body_element->getAttribute("style")));
    EXPECT_EQ(2, body_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname body_qn = { 0, 0 }, body_query = { 0, LibCSS_String::Intern("body") };
    css_qname style_query = { 0, LibCSS_String::Intern("style") };
    StyleContext::NodeName(0, body_node, &body_qn);
    EXPECT_EQ("body", LibCSS_String::ToUTF8String(body_qn.name));
    StyleContext::NodeIsRoot(0, body_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NamedAncestorNode(0, body_node, &html_query, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::NamedParentNode(0, body_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::NamedParentNode(0, body_node, &html_query, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::ParentNode(0, body_node, &vnode);
    EXPECT_EQ(html_node, vnode);
    StyleContext::SiblingNode(0, body_node, &vnode);
    EXPECT_EQ(head_node, vnode);
    StyleContext::NodeHasAttribute(0, body_node, &foo_query, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttribute(0, body_node, &style_query, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributeEqual(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributeEqual(0, body_node, &style_query, LibCSS_String::Intern("background-color: #00ff00"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributeDashmatch(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributeDashmatch(0, body_node, &style_query, LibCSS_String::Intern("background-"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributeIncludes(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributeIncludes(0, body_node, &style_query, LibCSS_String::Intern("background-color:"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributePrefix(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributePrefix(0, body_node, &style_query, LibCSS_String::Intern("background-color: #0"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributeSuffix(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributeSuffix(0, body_node, &style_query, LibCSS_String::Intern("0ff00"), &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasAttributeSubstring(0, body_node, &style_query, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasAttributeSubstring(0, body_node, &style_query, LibCSS_String::Intern("ground-color"), &matched);
    EXPECT_EQ(1, matched);
    CHECK(body_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, body_node->render->style.Display().v);
    EXPECT_EQ("00FF00", String::ToUTF8(body_node->render->style.BackgroundColor().cssText()));
#endif

    LFL::DOM::Node *h_node = body_element->firstChild();
    EXPECT_EQ("h1", String::ToUTF8(h_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, h_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_HEADING_ELEMENT, h_node->htmlElementType);
    LFL::DOM::HTMLHeadingElement *h_element = (LFL::DOM::HTMLHeadingElement*)h_node;
    EXPECT_EQ(h_node->AsElement(), h_element);
    EXPECT_EQ(h_node->AsHTMLHeadingElement(), h_element);
    EXPECT_EQ("h1", String::ToUTF8(h_element->tagName));
    EXPECT_EQ("foo  bar", String::ToUTF8(h_element->getAttribute("class")));
    EXPECT_EQ(1, h_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname h_qn = { 0, 0 }, h_query = { 0, LibCSS_String::Intern("h1") }, p_query = { 0, LibCSS_String::Intern("p") };
    css_qname bar_query = { 0, LibCSS_String::Intern("bar") }, cat_query = { 0, LibCSS_String::Intern("cat") };
    StyleContext::NodeName(0, h_node, &h_qn);
    EXPECT_EQ("h1", LibCSS_String::ToUTF8String(h_qn.name));
    lwc_string **h_classes = 0; uint32_t n_h_classes = 0;
    StyleContext::NodeClasses(0, h_node, &h_classes, &n_h_classes);
    EXPECT_EQ(2, n_h_classes);
    EXPECT_EQ("foo", LibCSS_String::ToUTF8String(h_classes[0]));
    EXPECT_EQ("bar", LibCSS_String::ToUTF8String(h_classes[1]));
    StyleContext::NodeIsRoot(0, h_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::SiblingNode(0, h_node, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::NamedSiblingNode(0, h_node, &p_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::NodeHasClass(0, h_node, foo_query.name, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeHasClass(0, h_node, cat_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasClass(0, h_node, bar_query.name, &matched);
    EXPECT_EQ(1, matched);
    StyleContext::NodeCountSiblings(0, h_node, false, false, &count);
    EXPECT_EQ(0, count);
    StyleContext::NodeCountSiblings(0, h_node, false, true, &count);
    EXPECT_EQ(1, count);
    StyleContext::NodeIsEmpty(0, h_node, &matched);
    EXPECT_EQ(0, matched);
    CHECK(h_node->render);
    EXPECT_EQ(LFL::DOM::Display::Block, h_node->render->style.Display().v);
    EXPECT_EQ("000000", String::ToUTF8(h_node->render->style.Color().cssText()));
    EXPECT_EQ("111111", String::ToUTF8(h_node->render->style.BackgroundColor().cssText()));
#endif

    LFL::DOM::Node *h1_text = h_element->firstChild();
    EXPECT_EQ(LFL::DOM::TEXT_NODE, h1_text->nodeType);
    EXPECT_EQ("#text", String::ToUTF8(h1_text->nodeName()));
    EXPECT_EQ("Very header", String::ToUTF8(h1_text->nodeValue()));
    CHECK_NE(h1_text->AsText(), 0);
#ifdef LFL_LIBCSS
    StyleContext::NodeIsEmpty(0, h1_text, &matched);
    EXPECT_EQ(1, matched);
#endif

    LFL::DOM::Node *p_node = body_element->childNodes.item(1);
    EXPECT_EQ("p", String::ToUTF8(p_node->nodeName()));
    EXPECT_EQ(LFL::DOM::ELEMENT_NODE, p_node->nodeType);
    EXPECT_EQ(LFL::DOM::HTML_PARAGRAPH_ELEMENT, p_node->htmlElementType);
    LFL::DOM::HTMLParagraphElement *p_element = (LFL::DOM::HTMLParagraphElement*)p_node;
    EXPECT_EQ(p_node->AsElement(), p_element);
    EXPECT_EQ(p_node->AsHTMLParagraphElement(), p_element);
    EXPECT_EQ("p", String::ToUTF8(p_element->tagName));
    EXPECT_EQ("cat", String::ToUTF8(p_element->getAttribute("id")));
    EXPECT_EQ(1, p_element->childNodes.length());
#ifdef LFL_LIBCSS
    css_qname p_qn = { 0, 0 };
    StyleContext::NodeName(0, p_node, &p_qn);
    EXPECT_EQ("p", LibCSS_String::ToUTF8String(p_qn.name));
    StyleContext::NodeIsRoot(0, p_node, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::SiblingNode(0, p_node, &vnode);
    EXPECT_EQ(h_node, vnode);
    StyleContext::NamedSiblingNode(0, p_node, &foo_query, &vnode);
    EXPECT_EQ(0, vnode);
    StyleContext::NamedSiblingNode(0, p_node, &h_query, &vnode);
    EXPECT_EQ(h_node, vnode);
    StyleContext::NodeHasId(0, p_node, foo_query.name, &matched);
    EXPECT_EQ(0, matched);
    StyleContext::NodeHasId(0, p_node, cat_query.name, &matched);
    EXPECT_EQ(1, matched);
#endif

    LFL::DOM::Node *p_text = p_element->firstChild();
    EXPECT_EQ(LFL::DOM::TEXT_NODE, p_text->nodeType);
    EXPECT_EQ("#text", String::ToUTF8(p_text->nodeName()));
    EXPECT_EQ("In the begining was fun.", String::ToUTF8(p_text->nodeValue()));

    LFL::DOM::NodeList node_list = sb.doc.node->getElementsByTagName("html");
    EXPECT_EQ(1, node_list.length());
    EXPECT_EQ(html_node, node_list.item(0));
    node_list.clear();
    node_list = body_element->getElementsByTagName("h1");
    EXPECT_EQ(1, node_list.length());
    EXPECT_EQ(h_node, node_list.item(0));
}

#ifdef LFL_LIBCSS
#include "lfapp/css.h"
TEST(DOMTest, CSS) {
    StyleSheet sheet(0, 0);
    sheet.Parse("h1 { color: red }\n"
                "em { color: blue }\n"
                "h1 em { color: green }");
    sheet.Done();
}
#endif

}; // namespace LFL
