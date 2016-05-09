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

#ifndef LFL_CORE_APP_TYPES_SYNTAX_H__
#define LFL_CORE_APP_TYPES_SYNTAX_H__
namespace LFL {

struct SyntaxMatcher {
  enum Flag { Regexp=1, Display=2, Contained=4, Transparent=8,
    RegexpDisplay=Regexp|Display, RegexpContained=Regexp|Contained, RegexpTransparent=Regexp|Transparent,
    DisplayContained=Display|Contained, DisplayTransparent=Display|Transparent,
    ContainedTransparent=Contained|Transparent, RegexpDisplayContained=Regexp|Display|Contained,
    RegexpDisplayTransparent=Regexp|Display|Transparent, RegexpContainedTransparent=Regexp|Contained|Transparent,
    DisplayContainedTransparent=Display|Contained|Transparent,
    RegexpDisplayContainedTransparent=Regexp|Display|Contained|Transparent
  };

  struct Rule {
    string name, match_beg, match_end, skip;
    int flag;
    vector<string> match_within;
  };

  struct RegionRule { String16 beg, end; };
  struct RegexRegionRule { Regex beg, end, skip; };
  struct RegexMatchRule { Regex match; };
  struct RuleType { enum { Region=1, RegexRegion=2, RegexMatch=3 }; };
  struct RulePointer { int type; size_t index; };

  struct CompiledRule {
    enum WithinType { WithinAll=1, WithinAllBut=2 };
    string name;
    bool display, transparent;
    int within_type;
    vector<uint16_t> within;
    vector<RulePointer> subrule;
  };

  vector<CompiledRule> rules;
  map<string, int> rulenames;
  vector<RegionRule> region_rule;
  vector<RegexRegionRule> regex_region_rule;
  vector<RegexMatchRule> regex_match_rule;
  vector<int> style_ind;
  bool sync_minlines=0, sync_maxlines=0;
  SyntaxMatcher(const vector<Rule>&, SyntaxStyleInterface *s=0, int da=0);

  void LoadStyle(SyntaxStyleInterface*, int default_attr);
  SyntaxMatch::List *GetSyntaxMatchList(Editor*, const SyntaxMatch::ListPointer&);
  void UpdateAnnotation(Editor*, DrawableAnnotation *out, int out_size);
  void GetLineAnnotation(Editor*, const Editor::LineMap::Iterator &i, const String16 &t,
                         int *parsed_line_index, int *parsed_anchor, DrawableAnnotation *out);
  void AnnotateLine(Editor*, const Editor::LineMap::Iterator &i, const String16 &t,
                    SyntaxMatch::State *current_state, vector<SyntaxMatch::State> *current_state_stack,
                    SyntaxMatch::ListPointer *current_parent, DrawableAnnotation *out);
};

struct RegexCPlusPlusHighlighter : public SyntaxMatcher {
  RegexCPlusPlusHighlighter(SyntaxStyleInterface *style=0, int default_attr=0);
};

}; // namespace LFL
#endif // LFL_CORE_APP_TYPES_SYNTAX_H__
