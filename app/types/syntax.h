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

  struct Keyword {
    string name;
    vector<string> word;
  };

  struct CompiledRule {
    enum { RegexMatch=1, RegexRegion=2, Region=3 };
    String16 beg_pat, end_pat;
    Regex beg, end, skip;
    int type=0;
    CompiledRule() {}
    CompiledRule(int T, const String16 &b, const String16 &e) : type(T), beg_pat(b), end_pat(e) {}
    CompiledRule(int T, const string &b, const string &e, const string &s) : type(T), beg(b), end(e), skip(s) {}
  };

  struct CompiledGroup {
    enum WithinType { WithinAll=1, WithinAllBut=2 };
    string name;
    bool display, transparent, keywords;
    int within_type;
    vector<uint16_t> within, non_display_within;
    vector<CompiledRule> rule;
  };

  struct RegionContext {
    SyntaxMatch::State state=0;
    SyntaxMatch::ListPointer parent={0,0};
    vector<SyntaxMatch::State> sig;
    void LoadLineStartRegionContext(Editor*, const Editor::LineMap::Iterator&);
    void LoadLineEndRegionContext(const Editor::LineMap::Iterator&);
    void SaveLineStartRegionContext(const Editor::LineMap::Iterator&);
    void SaveLineEndRegionContext(const Editor::LineMap::Iterator&);
    void PushRegionContext(Editor*, const Editor::LineMap::Iterator&, SyntaxMatch::State);
    void PopRegionContext(Editor*);
    static SyntaxMatch::List *GetSyntaxMatchListNode(Editor*, const SyntaxMatch::ListPointer&);
  };

  struct LineContext {
    SyntaxMatch::State state;
    int begin, end;
    bool region;
  };

  unordered_map<String16, pair<string, int>> keywords;
  vector<CompiledGroup> groups;
  map<string, int> group_name;
  vector<int> style_ind;
  bool sync_minlines=1, sync_maxlines=0;
  SyntaxMatcher(const vector<Rule>&, const vector<Keyword>&, SyntaxStyleInterface *s=0, int da=0);

  void LoadStyle(SyntaxStyleInterface*, int default_attr);
  void UpdateAnnotation(Editor*, DrawableAnnotation *out, int out_size);
  void GetLineAnnotation(Editor*, const Editor::LineMap::Iterator &i, const String16 &t,
                         bool first_line, int *parsed_line_index, int *parsed_anchor, DrawableAnnotation *out);
  void AnnotateLine(Editor*, const Editor::LineMap::Iterator &i, const String16 &t,
                    bool displayed_line, RegionContext *current, DrawableAnnotation *out);
};

struct RegexCPlusPlusHighlighter : public SyntaxMatcher {
  RegexCPlusPlusHighlighter(SyntaxStyleInterface *style=0, int default_attr=0);
};

}; // namespace LFL
#endif // LFL_CORE_APP_TYPES_SYNTAX_H__
