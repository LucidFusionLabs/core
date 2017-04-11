/*
 * $Id: shell.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef LFL_CORE_APP_SHELL_H__
#define LFL_CORE_APP_SHELL_H__
namespace LFL {

struct Shell {
  typedef function<void(const vector<string>&)> CB;
  struct Command { 
    string name;
    CB cb;
    Command(const string &N, const CB &Cb) : name(N), cb(Cb) {}
  };
  Window *parent;
  vector<Command> command;
  Shell(Window *W);

  template <class... Args> void Add(Args&&... args) { command.emplace_back(forward<Args>(args)...); }
  void AddSceneCommands(Scene *scene);
  void AddBrowserCommands(Browser*);

  bool FGets();
  void Run(const string &text);

  void quit(const vector<string>&);
  void mousein(const vector<string>&);
  void mouseout(const vector<string>&);
  void console(const vector<string>&);
  void consolecolor(const vector<string>&);
  void showkeyboard(const vector<string>&);
  void clipboard(const vector<string>&);
  void startcmd(const vector<string>&);
  void savedir(const vector<string>&);
  void savesettings(const vector<string>&);
  void screenshot(const vector<string>&);
  void testcrash(const StringVec&);

  void fillmode(const vector<string>&);
  void grabmode(const vector<string>&);
  void texmode (const vector<string>&);
  void swapaxis(const vector<string>&);
  void play     (const vector<string>&);
  void playmovie(const vector<string>&);
  void loadsound(const vector<string>&);
  void loadmovie(const vector<string>&);
  void copy(const vector<string>&);
  void snap(const vector<string>&);
  void filter   (const vector<string>&);
  void fftfilter(const vector<string>&);
  void f0(const vector<string>&);
  void sinth(const vector<string>&);
  void writesnap(const vector<string>&);
  void FPS(const vector<string>&);
  void NetworkStats(const vector<string>&);
  void WGet(const vector<string>&);
  void MessageBox(const vector<string>&);
  void TextureBox(const vector<string>&);
  void Slider    (const vector<string>&);
  void Edit      (const vector<string>&);

  void cmds (const vector<string>&);
  void flags(const vector<string>&);
  void binds(const vector<string>&);
  void constants(const vector<string>&);
};

}; // namespace LFL
#endif // LFL_CORE_APP_INPUT_H__
