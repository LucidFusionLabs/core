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

#ifndef LFL_CORE_APP_BINDINGS_LIBCSS_H__
#define LFL_CORE_APP_BINDINGS_LIBCSS_H__

namespace LFL {
struct LibCSS_String {
  static lwc_string *Intern(const string &s);
  static lwc_string *Intern(const String16 &s);
  static LFL::DOM::DOMString ToString(lwc_string *s);
  static string ToUTF8String(lwc_string *s);
};

struct LibCSS_Unit {
  static int ToPrimitive(int n);
};

struct LibCSS_StyleSheet {
  static css_error Resolve(void *pw, const char *base, lwc_string *rel, lwc_string **abs);
  static css_error Import(void *pw, css_stylesheet *parent, lwc_string *url, uint64_t media);
  static css_error Color(void *pw, lwc_string *name, css_color *color);
  static css_error Font(void *pw, lwc_string *name, css_system_font *system_font);
};

struct LibCSS_StyleContext {
  static css_select_handler *SelectHandler();
  static css_error NodeName(void *pw, void *n, css_qname *qname);
  static css_error NodeClasses(void *pw, void *n, lwc_string ***classes_out, uint32_t *n_classes);
  static css_error NodeId(void *pw, void *n, lwc_string **id);
  static css_error NamedAncestorNode(void *pw, void *n, const css_qname *qname, void **ancestor);
  static css_error NamedParentNode(void *pw, void *n, const css_qname *qname, void **parent);
  static css_error NamedSiblingNode(void *pw, void *n, const css_qname *qname, void **sibling);
  static css_error NamedGenericSiblingNode(void *pw, void *n, const css_qname *qname, void **sibling);
  static css_error ParentNode(void *pw, void *n, void **parent);
  static css_error SiblingNode(void *pw, void *n, void **sibling);
  static css_error NodeHasName(void *pw, void *n, const css_qname *qname, bool *match);
  static css_error NodeHasClass(void *pw, void *n, lwc_string *name, bool *match);
  static css_error NodeHasId(void *pw, void *n, lwc_string *name, bool *match);
  static css_error NodeHasAttribute(void *pw, void *n, const css_qname *qname, bool *match);
  static css_error NodeHasAttributeEqual(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeHasAttributeDashmatch(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeHasAttributeIncludes(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeHasAttributePrefix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeHasAttributeSuffix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeHasAttributeSubstring(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match);
  static css_error NodeIsRoot(void *pw, void *n, bool *match);
  static css_error NodeCountSiblings(void *pw, void *n, bool same_name, bool after, int32_t *count);
  static css_error NodeIsEmpty(void *pw, void *n, bool *match);
  static css_error NodeIsLink(void *pw, void *n, bool *match);
  static css_error NodeIsVisited (void *pw, void *n, bool *match);
  static css_error NodeIsHover   (void *pw, void *n, bool *match);
  static css_error NodeIsActive  (void *pw, void *n, bool *match);
  static css_error NodeIsFocus   (void *pw, void *n, bool *match);
  static css_error NodeIsEnabled (void *pw, void *n, bool *match);
  static css_error NodeIsDisabled(void *pw, void *n, bool *match);
  static css_error NodeIsChecked (void *pw, void *n, bool *match);
  static css_error NodeIsTarget  (void *pw, void *n, bool *match);
  static css_error NodeIsLang    (void *pw, void *n, lwc_string *lang, bool *match);
  static css_error NodePresentationalHint(void *pw, void *node, uint32_t property, css_hint *hint);
  static css_error UADefaultForProperty(void *pw, uint32_t property, css_hint *hint);
  static css_error ComputeFontSize(void *pw, const css_hint *parent, css_hint *size);
  static css_error SetLibCSSNodeData(void *pw, void *n, void *libcss_node_data);
  static css_error GetLibCSSNodeData(void *pw, void *n, void **libcss_node_data);
};

}; // namespace LFL
#endif // LFL_CORE_APP_BINDINGS_LIBCSS_H__
