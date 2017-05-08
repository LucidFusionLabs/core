/*
 * $Id: apple_common.h 1336 2014-12-08 09:29:59Z justin $
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

@interface ObjcTimer : NSObject
@end

@interface ObjcCallback : NSObject {}
  - (id)initWithCB:(LFL::Callback)cb;
  - (void)run;
  - (void)runAndRelease;
@end

@interface ObjcStringCallback : NSObject {}
  - (id)initWithStringCB:(LFL::StringCB)cb;
  - (void)run:(const LFL::string&)v;
@end

@protocol ObjcWindow<NSObject>
  - (void)objcWindowSelect;
  - (void)objcWindowFrame;
@end

@interface ObjcFileHandleCallback : NSObject {}
  @property (nonatomic, retain) NSFileHandle *fh;
  - (id)initWithCB:(std::function<bool()>)cb forWindow:(id<ObjcWindow>)w fileDescriptor:(int)fd;
  - (void)fileDataAvailable: (NSNotification *)notification;
@end

namespace LFL {
void NSLogString(const string &text);
string GetNSDocumentDirectory();

inline string GetNSString(NSString *x) { return x ? string([x UTF8String]) : string(); }
inline NSString *MakeNSString(const string &x) { return [[[NSString alloc] initWithBytes:x.data() length:x.size() encoding:NSUTF8StringEncoding] autorelease]; }
inline NSData *MakeNSData(const string &x) { return [NSData dataWithBytes:x.data() length:x.size()]; }
inline NSData *MakeNSData(const void *b, int l) { return [NSData dataWithBytes:b length:l]; }
inline CGPoint MakeCGPoint(const point &p) { return CGPointMake(p.x, p.y); }
inline CGPoint MakeCGPoint(const v2 &p) { return CGPointMake(p.x, p.y); }
inline v2 GetCGPoint(CGPoint p) { return v2(p.x, p.y); }
inline Box GetCGRect(CGRect r) { return Box(r.origin.x, r.origin.y, r.size.width, r.size.height); }
NSArray *MakeNSStringArray(const vector<string>&);
NSSet *MakeNSStringSet(const vector<string>&);
CGImageRef MakeCGImage(Texture&);

template<typename T> inline T* objc_dynamic_cast(id from) {
  return ([from isKindOfClass:[T class]]) ? static_cast<T*>(from) : nil;
}

struct AppleTimer : public TimerInterface {
  ObjcTimer *timer;
  ~AppleTimer();
  AppleTimer(Callback c);
  void Run(Time interval, bool force=false);
  bool Clear();
};

struct NSURLSessionStreamConnection : public Connection {
  static NSURLSession *session;
  NSURLSessionStreamTask *stream=0;
  Connection::CB connected_cb;
  function<bool()> readable_cb;
  bool connected=0, done=0;
  int outstanding_read=0, outstanding_write=0;
  string buf;

  virtual ~NSURLSessionStreamConnection();
  NSURLSessionStreamConnection(const string &hostport, int default_port, Connection::CB s_cb);

  void AddToMainWait(Window*, function<bool()> r_cb) override { readable_cb = move(r_cb); }
  void RemoveFromMainWait(Window*)                   override { readable_cb = function<bool()>(); }

  void Close() override;
  int Read() override { return done ? -1 : 0; }
  int WriteFlush(const char *buf, int len) override;

  void ConnectedCB();
  void ReadableCB(bool init, bool error, bool eof);
};

}; // namespace LFL
