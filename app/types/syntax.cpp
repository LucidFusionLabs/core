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

// #define LFL_SYTNAX_DEBUG
#ifdef LFL_SYTNAX_DEBUG
#define SyntaxDebug(...) ERRORf(__VA_ARGS__)
#else
#define SyntaxDebug(...)
#endif

namespace LFL {
void SyntaxMatcher::RegionContext::LoadLineStartRegionContext(Editor *editor,
                                                              const Editor::LineMap::Iterator &i) {
  sig = i.val->syntax_region_start_sig;
  parent = i.val->syntax_region_start_parent;
  if (parent.anchor) state = GetSyntaxMatchListNode(editor, parent)->state;
}

void SyntaxMatcher::RegionContext::SaveLineStartRegionContext(const Editor::LineMap::Iterator &i) {
  i.val->syntax_region_start_parent = parent;
  i.val->syntax_region_start_sig = sig;
  i.val->syntax_region_ancestor_list_storage.clear();
}

void SyntaxMatcher::RegionContext::SaveLineEndRegionContext(const Editor::LineMap::Iterator &i) {
  i.val->syntax_region_end_parent = parent;
  i.val->syntax_region_end_sig = sig;
  i.val->colored = true;
}

void SyntaxMatcher::RegionContext::PushRegionContext(Editor *editor, const Editor::LineMap::Iterator &i,
                                                     SyntaxMatch::State next_state) {
  i.val->syntax_region_ancestor_list_storage.push_back({ next_state, parent });
  parent = { i.GetAnchor(), unsigned(i.val->syntax_region_ancestor_list_storage.size()-1) };
  state = next_state;
  sig.push_back(next_state);
}

void SyntaxMatcher::RegionContext::PopRegionContext(Editor *editor) {
  parent = GetSyntaxMatchListNode(editor, parent)->parent;
  state = parent.anchor ? GetSyntaxMatchListNode(editor, parent)->state : 0;
  sig.pop_back();
  CHECK_EQ(state, (sig.size() ? sig.back() : 0));
}

SyntaxMatch::List*
SyntaxMatcher::RegionContext::GetSyntaxMatchListNode(Editor *editor, const SyntaxMatch::ListPointer &p) {
  return VectorCheckElement(CheckPointer(editor->file_line.GetAnchorVal(p.anchor))
                            ->syntax_region_ancestor_list_storage, p.storage_index);
}

SyntaxMatcher::SyntaxMatcher(const vector<SyntaxMatcher::Rule> &rules_in,
                             SyntaxStyleInterface *style, int default_attr) {
  groups.push_back({ string(), 0, 0, 0, vector<uint16_t>(), vector<uint16_t>(), vector<CompiledRule>() });
  string last_name = "";

  for (auto &r : rules_in) {
    CHECK(!r.name.empty());
    CHECK(!r.match_beg.empty());
    auto name_iter = group_name.find(r.name);
    bool first = name_iter == group_name.end();
    if (first) {
      name_iter = Insert(group_name, r.name, groups.size());
      if (!(r.flag & Contained)) groups[0].within.push_back(name_iter->second);
      groups.push_back({ r.name, bool(r.flag & Display), bool(r.flag & Transparent), 
                       0, vector<uint16_t>(), vector<uint16_t>(), vector<CompiledRule>() });
    } else { CHECK_EQ(last_name, r.name); }

    last_name = r.name;
    if (r.flag & Regexp) {
      groups.back().rule.emplace_back
        (r.match_end.empty() ? CompiledRule::RegexMatch : CompiledRule::RegexRegion,
         r.match_beg, r.match_end, r.skip);
    } else {
      CHECK(!r.match_end.empty());
      groups.back().rule.emplace_back
        (CompiledRule::Region, String::ToUTF16(r.match_beg), String::ToUTF16(r.match_end));
    }
  }

  int group_ind = 0;
  for (auto &r : rules_in) {
    if (last_name != r.name) { last_name = r.name; group_ind++; }
    CompiledGroup *g = VectorCheckElement(groups, group_ind);
    for (auto &i : r.match_within)
      if (int id = FindOrDefault(group_name, i, 0)) g->within.push_back(id);
      else if (i == "All")                          g->within_type = CompiledGroup::WithinAll;
      else if (i == "AllBut")                       g->within_type = CompiledGroup::WithinAllBut;
    sort(g->within.begin(), g->within.end());
    if (g->within_type == CompiledGroup::WithinAllBut) SetComplement(&g->within, groups[0].within);
  }
  CHECK_EQ(groups.size()-1, group_ind);

  for (auto &g : groups)
    for (auto &i : g.within) {
      CHECK_RANGE(i, 0, groups.size());
      if (!groups[i].display) g.non_display_within.push_back(i);
    }

  if (style) LoadStyle(style, default_attr);
}

void SyntaxMatcher::LoadStyle(SyntaxStyleInterface *style, int default_attr) {
  style_ind.clear();
  for (auto &g : groups)
    style_ind.push_back(g.transparent ? -1 : style->GetSyntaxStyle(g.name, default_attr));
}

void SyntaxMatcher::UpdateAnnotation(Editor *editor, DrawableAnnotation *out, int out_size) {
  String16 buf;
  RegionContext current_region;
  for (Editor::LineMap::Iterator i = editor->file_line.Begin(); i.ind; ++i) {
    CHECK_GE(i.val->annotation_ind, 0);
    AnnotateLine(editor, i, *editor->ReadLine(i, &buf), false, &current_region, &out[i.val->annotation_ind]);
  }
}

void SyntaxMatcher::GetLineAnnotation(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &t,
                                      int *parsed_line_index, int *parsed_anchor, DrawableAnnotation *out) {
  Editor::LineMap::Iterator pi;
  if (sync_minlines) {
    FATAL("not implemented");
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
  RegionContext current_region;
  if (*parsed_anchor) current_region.LoadLineStartRegionContext(editor, pi);

  for (/**/; pi.ind; ++pi) {
    bool last = pi.ind == i.ind;
    CHECK_GE(pi.val->annotation_ind, 0);
#if 0
    if (pi.val->colored && pi.val->syntax_region_start_sig == current_region.sig) continue;
    else { 
      current_region.sig = pi.val->syntax_region_end_sig;
      current_region.parent = pi.val->syntax_region_end_parent;
    }
#endif
    AnnotateLine(editor, pi, last ? t : *editor->ReadLine(pi, &buf),
                 pi.GetBegKey() >= editor->last_first_line, &current_region, &out[pi.val->annotation_ind]);
    if (last) break;
  }

  *parsed_anchor = i.GetAnchor();
  *parsed_line_index = i.GetIndex();
  // parse display
}

void SyntaxMatcher::AnnotateLine(Editor *editor, const Editor::LineMap::Iterator &i, const String16 &text,
                                 bool displayed_line, RegionContext *current_region, DrawableAnnotation *out) {
  String16Piece remaining_line(text);
  vector<LineContext> line_context;
  int current_anchor = i.GetAnchor(), current_group_ind, current_rule_ind;
  SyntaxMatch::GetStateIndices(current_region->state, &current_group_ind, &current_rule_ind);
  CHECK_RANGE(current_group_ind, 0, groups.size());
  out->clear();
  out->ExtendBack({0, style_ind[current_group_ind]});
  current_region->SaveLineStartRegionContext(i);
  SyntaxDebug("AnnotateLine crs=%d %s\n", current_region->state, String::ToUTF8(text).c_str());

  while (remaining_line.len > 0) {
    int offset = remaining_line.buf - text.data(), consumed = 0;
    auto &group = groups[current_group_ind];
    static pair<long, long> match_id_end(-1, 0), match_id_skip(-2, 0);
    pair<long, long> match_id;
    Regex::Result first_match, m;

    // find skip or end
    if (current_group_ind) {
      auto &parent_rule = group.rule[current_rule_ind];
      bool plus1 = line_context.size() && offset == line_context.back().begin;
      String16Piece remaining(remaining_line.buf + plus1, remaining_line.len - plus1);

      if (parent_rule.type == CompiledRule::RegexRegion && parent_rule.skip.impl) {
        m = parent_rule.skip.MatchOne(remaining);
        if (!!m && (!first_match || m < first_match)) { first_match=m; match_id=match_id_skip; }
      }

      switch (parent_rule.type) {
        case CompiledRule::Region:      m = FindStringIndex(remaining, String16Piece(parent_rule.end_pat)); break;
        case CompiledRule::RegexRegion: m = parent_rule.end.MatchOne(remaining); break;
        case CompiledRule::RegexMatch:  m = Regex::Result(PieceIndex(VectorBack(line_context)->end - offset-1, 0)); break;
      }
      if (!!m && (!first_match || m < first_match)) { first_match=m; match_id=match_id_end; }
      if (!!first_match && plus1) first_match += 1;
    }

    // find next
    auto &within = group.within_type == CompiledGroup::WithinAll ? groups[0].within : group.within;
    for (auto b = within.begin(), i = b, e = within.end(); i != e; ++i) {
      for (auto ib = groups[*i].rule.begin(), ii = ib, ie = groups[*i].rule.end(); ii != ie; ++ii) {
        switch(ii->type) {
          case CompiledRule::Region:      m = FindStringIndex(remaining_line, String16Piece(ii->beg_pat)); break;
          case CompiledRule::RegexRegion: m = ii->beg.MatchOne(remaining_line); break;
          case CompiledRule::RegexMatch:  m = ii->beg.MatchOne(remaining_line); break;
        }
        if (!!m && (!first_match || m < first_match)) { first_match=m; match_id={i-b+1, ii-ib}; }
      }
    }

    if (!first_match) break;
    CHECK(match_id.first);

    // skip, pop or push
    if      (match_id == match_id_skip) consumed = first_match.end;
    else if (match_id == match_id_end) {
      SyntaxMatch::State next_state = 0;
      bool have_next_state = 0, region = 1;
      if (line_context.size()) {
        region = PopBack(line_context).region;
        if ((have_next_state = line_context.size())) next_state = line_context.back().state;
      }
      SyntaxDebug("Match-end (r=%d) %s %s\n", region, groups[current_group_ind].name.c_str(),
                  String::ToUTF8(text.substr(offset+first_match.begin, first_match.end-first_match.begin)).c_str());

      if (region) current_region->PopRegionContext(editor);
      if (!have_next_state) next_state = current_region->state;

      consumed = first_match.end;
      SyntaxMatch::GetStateIndices(next_state, &current_group_ind, &current_rule_ind);
      out->ExtendBack({ offset + first_match.end, style_ind[current_group_ind] });
    } else {
      consumed = first_match.begin;
      SyntaxMatch::State next_state = SyntaxMatch::MakeState(within[match_id.first-1], match_id.second);
      SyntaxMatch::GetStateIndices(next_state, &current_group_ind, &current_rule_ind);

      bool region = groups[current_group_ind].rule[current_rule_ind].type != CompiledRule::RegexMatch; 
      line_context.push_back({ next_state, offset + first_match.begin, offset + first_match.end, region });
      if (region) current_region->PushRegionContext(editor, i, next_state);

      int a = style_ind[current_group_ind];
      if (a >= 0) out->ExtendBack({ offset + first_match.begin, a });
      SyntaxDebug("Match-begin (r=%d) %s %s\n", region, groups[current_group_ind].name.c_str(),
                  String::ToUTF8(text.substr(offset+first_match.begin, first_match.end-first_match.begin)).c_str());
    }
    remaining_line = String16Piece(remaining_line.buf + consumed, remaining_line.len - consumed);
  }
  CHECK_GE(remaining_line.len, 0);
  current_region->SaveLineEndRegionContext(i);
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
  Rule{ "String", "(L?\")", "(\")", "(\\\\\\\\|\\\\\")", Regexp, StringVec{"Special"} },
  Rule{ "Character", "(L?'[^\\\\]')", "", "", Regexp, StringVec{"Special"} },
  Rule{ "Character", "(L'[^']*')", "", "", Regexp, StringVec{} },
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
  Rule{ "Comment", "(//)", "(.$)", "(\\\\$)", Regexp, StringVec{"String", "Number", "Special", "SpecialChar"} },
  Rule{ "PreCondition", "(^\\s*(%:|#)\\s*(if|ifdef|ifndef|elif)\\b)", "(.$)", "", RegexpDisplay, StringVec{"Comment", "String", "Character", "Numbers"} },
  Rule{ "PreCondition", "(^\\s*(%:|#)\\s*(else|endif)\\b)", "(.$)", "", RegexpDisplay, StringVec{} },
  Rule{ "Included", "\"", "\"", "(\\\\\\\\|\\\\\")", DisplayContained, StringVec{} },
  Rule{ "Included", "(<[^>]*>)", "", "", RegexpDisplayContained, StringVec{} },
  Rule{ "Include", "(^\\s*(%:|#)\\s*include\\b\\s*[\"<])", "", "", RegexpDisplay, StringVec{"Included"} },
  Rule{ "Define",  "(^\\s*(%:|#)\\s*(define|undef)\\b)", "(.$)", "(\\\\$)", Regexp, StringVec{"AllBut", "PreCondition", "Include", "Define", "Special", "Numbers", "Comment" } },
  Rule{ "PreProc", "^\\s*(%:|#)\\s*(pragma\\b|line\\b|warning\\b|warn\\b|error\\b)", "(.$)", "", Regexp, StringVec{"AllBut", "PreCondition", "PreProc", "Include", "Define", "Special", "Numbers", "Comment"} },
}, style, default_attr) {}

}; // namespace LFL
