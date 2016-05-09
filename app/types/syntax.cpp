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
  rules.push_back({ string(), 0, 0, 0, vector<uint16_t>(), vector<RulePointer>() });
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
                      0, vector<uint16_t>(), vector<RulePointer>() });
    } else { CHECK_EQ(last_name, r.name); }
    last_name = r.name;

    if (r.flag & Regexp) {
      if (r.match_end.empty()) rules.back().subrule.push_back
        ({ RuleType::RegexMatch, PushBackIndex(regex_match_rule, { Regex(r.match_beg) }) });
      else rules.back().subrule.push_back
        ({ RuleType::RegexRegion, PushBackIndex(regex_region_rule, { Regex(r.match_beg), Regex(r.match_end), r.skip.empty() ? Regex() : Regex(r.skip) }) }); 
    } else {
      CHECK(!r.match_end.empty());
      rules.back().subrule.push_back
        ({ RuleType::Region, PushBackIndex(region_rule, { String::ToUTF16(r.match_beg), String::ToUTF16(r.match_end) }) });
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

SyntaxMatch::List *SyntaxMatcher::GetSyntaxMatchList(Editor *editor, const SyntaxMatch::ListPointer &p) {
  return VectorCheckElement(CheckPointer(editor->file_line.GetAnchorVal(p.anchor))
                            ->syntax_ancestor_list_storage, p.subindex);
}

void SyntaxMatcher::UpdateAnnotation(Editor *editor, DrawableAnnotation *out, int out_size) {
  String16 buf;
  SyntaxMatch::State current_state = 0;
  vector<SyntaxMatch::State> current_state_stack;
  SyntaxMatch::ListPointer current_parent = {0, 0};
  for (Editor::LineMap::Iterator i = editor->file_line.Begin(); i.ind; ++i) {
    CHECK_GE(i.val->annotation_ind, 0);
    AnnotateLine(editor, i, *editor->ReadLine(i, &buf), &current_state, &current_state_stack,
                 &current_parent, &out[i.val->annotation_ind]);
  }
}

void SyntaxMatcher::GetLineAnnotation(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &t,
                                      int *parsed_line_index, int *parsed_anchor, DrawableAnnotation *out) {
  Editor::LineMap::Iterator pi;
  if (sync_minlines) {
    // walk backward to find first line with non '#' character in position 0 where prev line doesnt end w '\\'
  } else {
    int anno_line_index = i.GetIndex();
    if (*parsed_anchor && anno_line_index < *parsed_line_index) {
      // parse display
      return;
    }
    pi = *parsed_anchor ? editor->file_line.GetAnchorIter(*parsed_anchor) : editor->file_line.Begin();
    CHECK_EQ(*parsed_line_index, pi.GetIndex());
  }

  String16 buf;
  SyntaxMatch::ListPointer current_parent = *parsed_anchor ? pi.val->syntax_parent : SyntaxMatch::ListPointer();
  SyntaxMatch::State current_state = current_parent.anchor ? GetSyntaxMatchList(editor, current_parent)->state : 0;
  vector<SyntaxMatch::State> current_state_stack = *parsed_anchor ? pi.val->syntax_state : vector<SyntaxMatch::State>();

  for (/**/; pi.ind; ++pi) {
    bool last = pi.ind == i.ind;
    CHECK_GE(pi.val->annotation_ind, 0);
    AnnotateLine(editor, pi, last ? t : *editor->ReadLine(pi, &buf),
                 &current_state, &current_state_stack, &current_parent, &out[pi.val->annotation_ind]);
    if (last) break;
  }

  *parsed_anchor = i.GetAnchor();
  *parsed_line_index = i.GetIndex();
  // parse display
}

void SyntaxMatcher::AnnotateLine(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &text,
                                 SyntaxMatch::State *current_state, vector<SyntaxMatch::State> *current_state_stack,
                                 SyntaxMatch::ListPointer *current_parent, DrawableAnnotation *out) {
  int current_state_ind = SyntaxMatch::GetStateIndex(*current_state);
  CHECK_RANGE(current_state_ind, 0, rules.size());
  out->clear();
  out->ExtendBack({0, style_ind[current_state_ind]});
  i.val->syntax_parent = *current_parent;
  i.val->syntax_state = *current_state_stack;
  i.val->syntax_ancestor_list_storage.clear();
  int current_anchor = i.GetAnchor(), last_linematch_end = -1;
  String16Piece remaining_line(text);

  while (remaining_line.len > 0) {
    auto &rule = rules[current_state_ind];
    int offset = remaining_line.buf - text.data(), consumed = 0;
    Regex::Result first_match, m;
    pair<long, long> match_id;

    // find end
    if (current_state_ind) {
      auto parent = GetSyntaxMatchList(editor, *current_parent);
      auto &parent_ind = rule.subrule[SyntaxMatch::GetSubStateIndex(parent->state)];
      switch (parent_ind.type) {
        case RuleType::Region:      m = Regex::Result(FindStringIndex(remaining_line, String16Piece(region_rule[parent_ind.index].end))); break;
        case RuleType::RegexMatch:  m = Regex::Result(PieceIndex(last_linematch_end - offset, 0)); break;
        case RuleType::RegexRegion: m = regex_region_rule[parent_ind.index].end.MatchOne(remaining_line); break;
      }
      if (!!m && (!first_match || m < first_match)) { first_match=m; match_id={-1, 0}; }
    } else { CHECK_EQ(0, current_state_ind); }

    // find next
    auto &within = rule.within_type == CompiledRule::WithinAll ? rules[0].within : rule.within;
    for (auto b = within.begin(), i = b, e = within.end(); i != e; ++i) {
      for (auto ib = rules[*i].subrule.begin(), ii = ib, ie = rules[*i].subrule.end(); ii != ie; ++ii) {
        switch(ii->type) {
          case RuleType::Region:      m = Regex::Result(FindStringIndex(remaining_line, String16Piece(region_rule[ii->index].beg))); break;
          case RuleType::RegexMatch:  m = regex_match_rule [ii->index].match.MatchOne(remaining_line); break;
          case RuleType::RegexRegion: m = regex_region_rule[ii->index].beg.  MatchOne(remaining_line); break;
        }
        if (!!m && (!first_match || m < first_match)) { first_match=m; match_id={i-b+1, ii-ib}; }
      }
    }

    if (!first_match) break;
    CHECK(match_id.first);
    // push or pop
    if (match_id.first == -1) {
      *current_parent = GetSyntaxMatchList(editor, *current_parent)->parent;
      *current_state = current_parent->anchor ? GetSyntaxMatchList(editor, *current_parent)->state : 0;
      current_state_stack->pop_back();
      CHECK_EQ(*current_state, (current_state_stack->size() ? current_state_stack->back() : 0));
      current_state_ind = SyntaxMatch::GetStateIndex(*current_state);
      out->ExtendBack({ offset + first_match.end, style_ind[current_state_ind] });
      consumed = first_match.end;
    } else { 
      last_linematch_end = offset + first_match.end;
      i.val->syntax_ancestor_list_storage.push_back
        ({ SyntaxMatch::MakeState(within[match_id.first-1], match_id.second), *current_parent });
      *current_parent = { i.GetAnchor(), unsigned(i.val->syntax_ancestor_list_storage.size()-1) };
      *current_state = i.val->syntax_ancestor_list_storage.back().state;
      current_state_stack->push_back(*current_state);
      current_state_ind = SyntaxMatch::GetStateIndex(*current_state);
      int a = style_ind[current_state_ind];
      if (a >= 0) out->ExtendBack({ offset + first_match.begin, a });
      consumed = min(first_match.begin+1, remaining_line.len);
    }
    remaining_line = String16Piece(remaining_line.buf + consumed, remaining_line.len - consumed);
  }

  // end line
  CHECK_GE(remaining_line.len, 0);
  while(current_parent->anchor) {
    auto parent = GetSyntaxMatchList(editor, *current_parent);
    int type = rules[SyntaxMatch::GetStateIndex(parent->state)].
      subrule[SyntaxMatch::GetSubStateIndex(parent->state)].type;
    bool line_based = type == RuleType::RegexMatch;
    if (!line_based) break;
    *current_parent = parent->parent;
    *current_state = current_parent->anchor ? GetSyntaxMatchList(editor, *current_parent)->state : 0;
    current_state_stack->pop_back();
    CHECK_EQ(*current_state, (current_state_stack->size() ? current_state_stack->back() : 0));
    current_state_ind = SyntaxMatch::GetStateIndex(*current_state);
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
