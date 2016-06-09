/*
 * $Id: ios_common.h 1336 2014-12-08 09:29:59Z justin $
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

@interface MyTouchView : UIView {}
@end

@interface MyTextField : UITextField {}
@end

@interface LFViewController : GLKViewController<GLKViewControllerDelegate, UIActionSheetDelegate> {}
  @property BOOL autorotate;
  - (void)updateToolbarFrame;
  - (CGRect)getKeyboardToolbarFrame;
@end

@interface LFUIApplication : NSObject <UIApplicationDelegate, GLKViewDelegate, UITextFieldDelegate> {}
  @property (nonatomic, retain) UIWindow *window;
  @property (nonatomic, retain) LFViewController *controller;
  @property (nonatomic, retain) GLKView *view;
  @property (nonatomic, retain) UIView *lview, *rview;
  @property (nonatomic, retain) MyTextField *textField;
  @property BOOL resign_textfield_on_return, frame_on_keyboard_input, frame_on_mouse_input, downscale_animation;
  @property int screen_y, screen_width, screen_height;
  + (LFUIApplication *) sharedAppDelegate;
  - (CGFloat)getScale;
  - (CGRect)getFrame;
  - (CGRect)getKeyboardFrame;
@end
