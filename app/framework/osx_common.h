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

@interface GameView : NSView
  @property (nonatomic, retain) NSMutableDictionary *main_wait_fh;
  @property (nonatomic, assign) LFL::Window *screen;
  @property (nonatomic, assign) BOOL frame_on_keyboard_input, frame_on_mouse_input, should_close;

  + (NSOpenGLPixelFormat*)defaultPixelFormat;
  - (void)clearKeyModifiers;
  - (void)update;
@end

@interface GameContainerView : NSView<NSWindowDelegate, ObjcWindow>
  @property (nonatomic, retain) GameView *gameView;
@end

namespace LFL {
struct OSXWindow : public Window {
  GameView *view=0;
  NSOpenGLContext *gl=0;
  OSXWindow(Application *app) : Window(app) {}
  virtual ~OSXWindow() { ClearChildren(); }
  void SetCaption(const string &c) override;
  void SetResizeIncrements(float x, float y) override;
  void SetTransparency(float v) override;
  bool Reshape(int w, int h) override;
  void Wakeup(int flag=0) override;
  int Swap() override;
};
}; // namespace LFL
