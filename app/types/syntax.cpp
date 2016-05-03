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
      rules.back().index.emplace_back(CompiledRule::Region, PushBackIndex(region_rule, { r.match_beg, r.match_end }));
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

void SyntaxMatcher::UpdateAnnotation(const string &text, DrawableAnnotation *out, int out_size) {
  vector<SyntaxParseState> state_stack;
  const char *iter = text.data();
  int current_state = 0, current_line_number = -1;
  StringPiece current_line, remaining_line;
  StringLineIter lines(text, StringLineIter::Flag::BlankLines);

  for (current_line.buf = lines.Next(); current_line.buf; current_line.buf = lines.Next()) {
    current_line_number++;
    current_line.len = lines.CurrentLength();
    remaining_line = current_line;
    CHECK_RANGE(current_line_number, 0, out_size);
    CHECK_RANGE(current_state, 0, rules.size());
    auto &out_line = out[current_line_number];
    out_line.clear();
    out_line.ExtendBack({0, style_ind[current_state]});

    while (remaining_line.len > 0) {
      auto &rule = rules[current_state];
      int offset = remaining_line.buf - current_line.buf, consumed = 0;
      Regex::Result first_match, m;
      pair<int, int> match_id;

      // find end
      if (current_state) {
        auto &parent_ind = rule.index[state_stack.back().index_offset];
        switch (parent_ind.first) {
          case CompiledRule::Region:      m = Regex::Result(FindStringIndex(remaining_line, region_rule[parent_ind.second].end)); break;
          case CompiledRule::RegexMatch:  m = Regex::Result(PieceIndex(state_stack.back().end.second - offset, 0)); break;
          case CompiledRule::RegexRegion: m = regex_region_rule[parent_ind.second].end.MatchOne(remaining_line); break;
        }
        if (!!m && (!first_match || m < first_match)) { first_match=m; match_id=make_pair(-1, 0); }
      } else { CHECK_EQ(0, current_state); }

      // find next
      auto &within = rule.within_type == CompiledRule::WithinAll ? rules[0].within : rule.within;
      for (auto b = within.begin(), i = b, e = within.end(); i != e; ++i) {
        for (auto ib = rules[*i].index.begin(), ii = ib, ie = rules[*i].index.end(); ii != ie; ++ii) {
          switch(ii->first) {
            case CompiledRule::Region:      m = Regex::Result(FindStringIndex(remaining_line, region_rule[ii->second].beg)); break;
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
        auto done = PopBack(state_stack);
        current_state = state_stack.size() ? state_stack.back().id : 0;
        out_line.ExtendBack({ offset + first_match.end, style_ind[current_state] });
        consumed = first_match.end;
      } else { 
        state_stack.push_back
          ({ within[match_id.first-1], match_id.second,
           make_pair(current_line_number, offset + first_match.begin),
           make_pair(current_line_number, offset + first_match.end) });
        current_state = state_stack.back().id;
        int a = style_ind[current_state];
        if (a >= 0) out_line.ExtendBack({ offset + first_match.begin, a });
        consumed = min(first_match.begin+1, remaining_line.len);
      }
      remaining_line = StringPiece(remaining_line.buf + consumed, remaining_line.len - consumed);
    }

    // end line
    CHECK_GE(remaining_line.len, 0);
    while(state_stack.size()) {
      const auto &b = state_stack.back();
      int type = rules[b.id].index[b.index_offset].first;
      bool line_based = type == CompiledRule::RegexMatch;
      if (!line_based) break;
      state_stack.pop_back();
      current_state = state_stack.size() ? state_stack.back().id : 0;
    }
  }
}

void SyntaxMatcher::GetLineAnnotation(const Editor::LineMap::Iterator &i, const String16 &t, DrawableAnnotation *out) {
  // if ind < max_up_to_date_line just return o->main_annotation
}

// \o -> [0-7]            \( -> (
// \x -> [0-9|A-F|a-f]    \) -> )
// \d -> \\d              \[ -> [
// \< -> \\b              \] -> ]
// \> -> \\b              \| -> |
// \= -> ?                \. -> .

RegexCPlusPlusHighlighter::RegexCPlusPlusHighlighter
(SyntaxStyleInterface *style, int default_attr) : SyntaxMatcher({
  Rule{ "Special", "\\\\(x[0-9|A-F|a-f]+|[0-7]{1,3}|.)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Special", "\\\\(u[0-9|A-F|a-f]{4}|U[0-9|A-F|a-f]{8})", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "String", "\"", "\"", "(\\\\\\\\|\\\\\")", 0, StringVec{"Special", "SpecialChar"} },
  Rule{ "Character", "('[^\\\\]')", "", "", Regexp, StringVec{} },
  Rule{ "Character", "('[^']*')", "", "", Regexp, StringVec{"Special"} },
  Rule{ "SpecialError", "('\\\\[^'\"?\\\\abefnrtv]')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "('\\\\['\"?\\\\abefnrtv]')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "('\\\\[0-7]{1,3}')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "SpecialChar", "('\\\\x[0-9|A-F|a-f]{1,2}')", "", "", RegexpDisplay, StringVec{} },
  Rule{ "Numbers", "(\\\\b\\d|.\\d)", "", "", RegexpDisplay, StringVec{"Number", "Float", "Octal"} },
  Rule{ "Number", "(\\d+(u?l{0,2}|ll?u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Number", "(0x[0-9|A-F|a-f]+(u?l{0,2}|ll?u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Octal", "(0[0-7]+(u?l{0,2}|ll=u)\\b)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\d+f)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(\\d+.\\d*(e[-+]?\\d+)?[fl]?)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Float", "(.\\d+(e[-+]?\\d+)?[fl]?\\b)", "", "", RegexpDisplayContained, StringVec{} },
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
