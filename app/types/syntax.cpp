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

#include "core/app/gui.h"
#include "core/app/types/syntax.h"

namespace LFL {
SyntaxMatcher::SyntaxMatcher(const vector<SyntaxMatcher::Rule> &rules_in, SyntaxStyleInterface *style, int default_attr) {
  rules.push_back({ string(), 0, 0, 0, vector<int>(), vector<pair<int,int>>() });
  string last_name = "";

  for (auto &r : rules_in) {
    CHECK(!r.name.empty());
    CHECK(!r.match_beg.empty());
    auto name_iter = rulenames.find(r.name);
    bool first = name_iter == rulenames.end();
    if (first) {
      name_iter = Insert(rulenames, r.name, rules.size());
      if (!(r.flag & Contained)) rules[0].within.push_back(name_iter->second);
      rules.push_back({ r.name, bool(r.flag & Display), bool(r.flag & Transparent), 
                      0, vector<int>(), vector<pair<int,int>>() });
    } else { CHECK_EQ(last_name, r.name); }
    last_name = r.name;

    if (r.flag & Regexp) {
      if (r.match_end.empty()) rules.back().index.emplace_back
        (CompiledRule::RegexMatch, PushBackIndex(regex_match_rule, { Regex(r.match_beg) }));
      else rules.back().index.emplace_back
        (CompiledRule::RegexRegion, PushBackIndex(regex_region_rule, { Regex(r.match_beg), Regex(r.match_end), r.skip.empty() ? Regex() : Regex(r.skip) })); 
    } else {
      CHECK(!r.match_end.empty());
      rules.back().index.emplace_back
        (CompiledRule::Region, PushBackIndex(region_rule, { String::ToUTF16(r.match_beg), String::ToUTF16(r.match_end) }));
    }
  }

  int rule_ind = 0;
  for (auto &r : rules_in) {
    if (last_name != r.name) { last_name = r.name; rule_ind++; }
    CompiledRule *rule = VectorCheckElement(rules, rule_ind);
    for (auto &i : r.match_within)
      if (int id = FindOrDefault(rulenames, i, 0)) rule->within.push_back(id);
      else if (i == "All")                         rule->within_type = CompiledRule::WithinAll;
      else if (i == "AllBut")                      rule->within_type = CompiledRule::WithinAllBut;
  }
  CHECK_EQ(rules.size()-1, rule_ind);
  if (style) LoadStyle(style, default_attr);
}

void SyntaxMatcher::LoadStyle(SyntaxStyleInterface *style, int default_attr) {
  style_ind.clear();
  for (auto &r : rules) style_ind.push_back(r.transparent ? -1 : style->GetSyntaxStyle(r.name, default_attr));
}

SyntaxParseState *SyntaxMatcher::GetAnchorParseState(Editor *editor, const pair<int, int> &p) {
  return VectorCheckElement(CheckPointer(editor->file_line.GetAnchorVal(p.first))->syntax_buf, p.second);
}

void SyntaxMatcher::UpdateAnnotation(Editor *editor, DrawableAnnotation *out, int out_size) {
  String16 buf;
  int current_state = 0;
  pair<int, int> current_parent;
  for (Editor::LineMap::Iterator i = editor->file_line.Begin(); i.ind; ++i) {
    CHECK_GE(i.val->annotation_ind, 0);
    AnnotateLine(editor, i, *editor->ReadLine(i, &buf), &current_state, &current_parent, &out[i.val->annotation_ind]);
  }
}

void SyntaxMatcher::GetLineAnnotation(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &t,
                                      int *parsed_line_index, int *parsed_anchor, DrawableAnnotation *out) {
  Editor::LineMap::Iterator pi;
  if (sync_minlines) {
    // walk backward to find first line with non '#' character in position 0 where prev line doesnt end w '\\'
  } else {
    int anno_line_index = i.GetIndex();
    if (*parsed_anchor && anno_line_index <= *parsed_line_index) {
      // parse display
      return;
    }
    pi = *parsed_anchor ? editor->file_line.GetAnchorIter(*parsed_anchor) : editor->file_line.Begin();
    // CHECK_EQ(*parsed_line_index, pi.GetIndex());
  }

  String16 buf;
  pair<int, int> current_parent = *parsed_anchor ? pi.val->syntax_parent : pair<int, int>();
  int current_state = current_parent.first ? GetAnchorParseState(editor, current_parent)->state : 0;

  for (/**/; pi.ind; ++pi) {
    bool last = pi.ind == i.ind;
    CHECK_GE(pi.val->annotation_ind, 0);
    AnnotateLine(editor, pi, last ? t : *editor->ReadLine(pi, &buf),
                 &current_state, &current_parent, &out[pi.val->annotation_ind]);
    if (last) break;
  }

  *parsed_anchor = i.GetAnchor();
  *parsed_line_index = i.GetIndex();
  // parse display
}

void SyntaxMatcher::AnnotateLine(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &text,
                                 int *current_state, pair<int,int> *current_parent, DrawableAnnotation *out) {
  CHECK_RANGE(*current_state, 0, rules.size());
  out->clear();
  out->ExtendBack({0, style_ind[*current_state]});
  i.val->syntax_parent = *current_parent;
  i.val->syntax_buf.clear();
  int current_anchor = i.GetAnchor();
  String16Piece remaining_line(text);

  while (remaining_line.len > 0) {
    auto &rule = rules[*current_state];
    int offset = remaining_line.buf - text.data(), consumed = 0;
    Regex::Result first_match, m;
    pair<int, int> match_id;

    // find end
    if (*current_state) {
      auto parent = GetAnchorParseState(editor, *current_parent);
      auto &parent_ind = rule.index[parent->substate];
      switch (parent_ind.first) {
        case CompiledRule::Region:      m = Regex::Result(FindStringIndex(remaining_line, String16Piece(region_rule[parent_ind.second].end))); break;
        case CompiledRule::RegexMatch:  m = Regex::Result(PieceIndex(parent->end.second - offset, 0)); break;
        case CompiledRule::RegexRegion: m = regex_region_rule[parent_ind.second].end.MatchOne(remaining_line); break;
      }
      if (!!m && (!first_match || m < first_match)) { first_match=m; match_id=make_pair(-1, 0); }
    } else { CHECK_EQ(0, *current_state); }

    // find next
    auto &within = rule.within_type == CompiledRule::WithinAll ? rules[0].within : rule.within;
    for (auto b = within.begin(), i = b, e = within.end(); i != e; ++i) {
      for (auto ib = rules[*i].index.begin(), ii = ib, ie = rules[*i].index.end(); ii != ie; ++ii) {
        switch(ii->first) {
          case CompiledRule::Region:      m = Regex::Result(FindStringIndex(remaining_line, String16Piece(region_rule[ii->second].beg))); break;
          case CompiledRule::RegexMatch:  m = regex_match_rule [ii->second].match.MatchOne(remaining_line); break;
          case CompiledRule::RegexRegion: m = regex_region_rule[ii->second].beg.  MatchOne(remaining_line); break;
        }
        if (!!m && (!first_match || m < first_match)) { first_match=m; match_id=make_pair(i-b+1, ii-ib); }
      }
    }

    if (!first_match) break;
    CHECK(match_id.first);
    // push or pop
    if (match_id.first == -1) {
      *current_parent = GetAnchorParseState(editor, *current_parent)->parent;
      *current_state = current_parent->first ? GetAnchorParseState(editor, *current_parent)->state : 0;
      out->ExtendBack({ offset + first_match.end, style_ind[*current_state] });
      consumed = first_match.end;
    } else { 
      i.val->syntax_buf.push_back
        ({ *current_parent, 
         make_pair(current_anchor, offset + first_match.begin),
         make_pair(current_anchor, offset + first_match.end),
         within[match_id.first-1], match_id.second });

      *current_parent = make_pair(i.GetAnchor(), i.val->syntax_buf.size()-1);
      *current_state = i.val->syntax_buf.back().state;
      int a = style_ind[*current_state];
      if (a >= 0) out->ExtendBack({ offset + first_match.begin, a });
      consumed = min(first_match.begin+1, remaining_line.len);
    }
    remaining_line = String16Piece(remaining_line.buf + consumed, remaining_line.len - consumed);
  }

  // end line
  CHECK_GE(remaining_line.len, 0);
  while(current_parent->first) {
    auto parent = GetAnchorParseState(editor, *current_parent);
    int type = rules[parent->state].index[parent->substate].first;
    bool line_based = type == CompiledRule::RegexMatch;
    if (!line_based) break;
    *current_parent = parent->parent;
    *current_state = current_parent->first ? GetAnchorParseState(editor, *current_parent)->state : 0;
  }
}

// \\ -> \\\\             \+ -> +      +  -> \\+
// \o -> [0-7]            \( -> (      (  -> \\(
// \x -> [0-9|A-F|a-f]    \) -> )      )  -> \\)
// \d -> \\d              \| -> |      |  -> \\|
// \< -> \\b              \{ -> {      {  -> \\{
// \> -> \\b              \} -  }      }  -> \\}
// \[ -> \\[              \. -> \\.    .  -> .
// \] -> \\]              \= -> ?      =  -> =

RegexCPlusPlusHighlighter::RegexCPlusPlusHighlighter
(SyntaxStyleInterface *style, int default_attr) : SyntaxMatcher({
  Rule{ "Special", "\\\\(x[0-9|A-F|a-f]+|[0-7]{1,3}|.|$)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Special", "\\\\(u[0-9|A-F|a-f]{4}|U[0-9|A-F|a-f]{8})", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "String", "(L?\")", "(\")", "(\\\\\\\\|\\\\\")", Regexp, StringVec{"Special", "SpecialChar"} },
  Rule{ "Character", "(L?'[^\\\\]')", "", "", Regexp, StringVec{} },
  Rule{ "Character", "(L?'[^']*')", "", "", Regexp, StringVec{"Special"} },
  Rule{ "SpecialError", "(L?'\\\\[^'\"\\?\\\\abefnrtv]')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "(L?'\\\\['\"\\?\\\\abefnrtv]')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "(L?'\\\\[0-7]{1,3}')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "(L?'\\\\x[0-9|A-F|a-f]{1,2}')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "Numbers", "(\\b\\d|\\.\\d)", "", "", RegexpDisplay, StringVec{"Number", "Float", "Octal"} },
  Rule{ "Number", "(\\d+(u?l{0,2}|ll?u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Number", "(0x[0-9|A-F|a-f]+(u?l{0,2}|ll?u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Octal", "(0[0-7]+(u?l{0,2}|ll?u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\d+f)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\d+\\.\\d*(e[-+]?\\d+)?[fl]?)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\.\\d+(e[-+]?\\d+)?[fl]?\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\d+e[-+]?\\d+[fl]?\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Comment", "/*", "*/", "", 0, StringVec{"String", "Number", "Special", "SpecialChar"} },
  Rule{ "PreCondition", "(^\\s*(%:|#)\\s*(if|ifdef|ifndef|elif)\\b)", "$", "", RegexpDisplay, StringVec{"All"} },
  Rule{ "PreCondition", "(^\\s*(%:|#)\\s*(else|endif)\\b)", "$", "", RegexpDisplay, StringVec{"All"} },
  Rule{ "Included", "\"", "\"", "(\\\\\\\\|\\\\\")", DisplayContained, StringVec{} },
  Rule{ "Included", "(<[^>]*>)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Include", "(^\\s*(%:|#)\\s*include\\b\\s*[\"<])", "", "", RegexpDisplay, StringVec{"Included"} },
  Rule{ "Define", "(^\\s*(%:|#)\\s*(define|undef)\\b)", "", "", Regexp, StringVec{"All"} },
  Rule{ "PreProc", "^\\s*(%:|#)\\s*(pragma\\b|line\\b|warning\\b|warn\\b|error\\b)", "$", "", Regexp, StringVec{"All"} },
}, style, default_attr) {}

}; // namespace LFL
