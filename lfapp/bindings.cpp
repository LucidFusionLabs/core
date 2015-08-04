/*
 * $Id: bindings.cpp 1335 2014-12-02 04:13:46Z justin $
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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/browser.h"

#ifdef LFL_CUDA
#include <cuda_runtime.h>
#include "lfcuda/lfcuda.h"
#include "speech/hmm.h"
#include "speech/speech.h"
#endif

extern "C" {
#ifdef LFL_LUA
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#endif
};

#ifdef LFL_V8JS
#include <v8.h>
#endif

namespace LFL {
#ifdef LFL_CUDA
void PrintCUDAProperties(cudaDeviceProp *prop) {
  DEBUGf("Major revision number:         %d", prop->major);
  DEBUGf("Minor revision number:         %d", prop->minor);
  DEBUGf("Name:                          %s", prop->name);
  DEBUGf("Total global memory:           %u", prop->totalGlobalMem);
  DEBUGf("Total shared memory per block: %u", prop->sharedMemPerBlock);
  DEBUGf("Total registers per block:     %d", prop->regsPerBlock);
  DEBUGf("Warp size:                     %d", prop->warpSize);
  DEBUGf("Maximum memory pitch:          %u", prop->memPitch);
  DEBUGf("Maximum threads per block:     %d", prop->maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of block: %d", i, prop->maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of grid:  %d", i, prop->maxGridSize[i]);
  DEBUGf("Clock rate:                    %d", prop->clockRate);
  DEBUGf("Total constant memory:         %u", prop->totalConstMem);
  DEBUGf("Texture alignment:             %u", prop->textureAlignment);
  DEBUGf("Concurrent copy and execution: %s", (prop->deviceOverlap ? "Yes" : "No"));
  DEBUGf("Number of multiprocessors:     %d", prop->multiProcessorCount);
  DEBUGf("Kernel execution timeout:      %s", (prop->kernelExecTimeoutEnabled ? "Yes" : "No"));
}

int CUDA::Init() {
  INFO("CUDA::Init()");
  FLAGS_lfapp_cuda = 0;

  int cuda_devices = 0;
  cudaError_t err;
  if ((err = cudaGetDeviceCount(&cuda_devices)) != cudaSuccess)
  { ERROR("cudaGetDeviceCount error ", cudaGetErrorString(err)); return 0; }

  cudaDeviceProp prop;
  for (int i=0; i<cuda_devices; i++) {
    if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) { ERROR("cudaGetDeviceProperties error ", err); return 0; }
    if (FLAGS_lfapp_debug) PrintCUDAProperties(&prop);
    if (strstr(prop.name, "Emulation")) continue;
    FLAGS_lfapp_cuda=1;
  }

  if (FLAGS_lfapp_cuda) {
    INFO("CUDA device detected, enabling acceleration: lfapp_cuda(", FLAGS_lfapp_cuda, ") devices ", cuda_devices);
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    cuda_init_hook();
  }
  else INFO("no CUDA devices detected ", cuda_devices);
  return 0;
}
#else
int CUDA::Init() { FLAGS_lfapp_cuda=0; INFO("CUDA not supported lfapp_cuda(", FLAGS_lfapp_cuda, ")"); return 0; }
#endif /* LFL_CUDA */

#ifdef LFL_LUA
struct MyLuaContext : public LuaContext {
  lua_State *L;
  ~MyLuaContext() { lua_close(L); }
  MyLuaContext() : L(luaL_newstate()) {
    luaopen_base(L);
    luaopen_table(L);
    luaopen_io(L);
    luaopen_string(L);
    luaopen_math(L);
  }
  string Execute(const string &s) {
    if (luaL_loadbuffer(L, s.data(), s.size(), "MyLuaExec")) { ERROR("luaL_loadstring ", lua_tostring(L, -1)); return ""; }
    if (lua_pcall(L, 0, LUA_MULTRET, 0))                     { ERROR("lua_pcall ",       lua_tostring(L, -1)); return ""; }
    return "";
  }
};
LuaContext *CreateLuaContext() { return new MyLuaContext(); }
#else /* LFL_LUA */
LuaContext *CreateLuaContext() { return 0; }
#endif /* LFL_LUA */

#ifdef LFL_V8JS
v8::Local<v8::String> NewV8String(v8::Isolate *I, const char  *s) { return v8::String::NewFromUtf8(I, s); }
v8::Local<v8::String> NewV8String(v8::Isolate *I, const short *s) { return v8::String::NewFromTwoByte(I, (const uint16_t *)s); }
template <class X> inline X CastV8InternalFieldTo(v8::Local<v8::Object> &self, int field_index) {
  return static_cast<X>(v8::Local<v8::External>::Cast(self->GetInternalField(field_index))->Value());
}
#define V8_SimpleMemberReturn(X, type, ret) \
  v8::Local<v8::Object> self = args.Holder(); \
  X *inst = CastV8InternalFieldTo<X*>(self, 1); \
  args.GetReturnValue().Set(type(args.GetIsolate(), (ret)));

#define V8_ObjectMemberReturn(X, Y, OT, ret) \
  v8::Local<v8::Object> self = args.Holder(); \
  X *impl = CastV8InternalFieldTo<X*>(self, 1); \
  Y *val = (ret); \
  if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; } \
  MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0); \
  v8::Local<v8::Object> impl_obj = (js_context->*OT)->NewInstance(); \
  impl_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context)); \
  impl_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val)); \
  impl_obj->SetInternalField(2, v8::Integer ::New(args.GetIsolate(), TypeId(val))); \
  args.GetReturnValue().Set(impl_obj);

template <typename X, int (X::*Y)() const> void MemberIntFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
  V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, int (X::*Y)() /***/> void MemberIntGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
  V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, int (X::*Y)() const> void MemberIntGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
  V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, DOM::DOMString (X::*Y)() const> void MemberStringFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
  V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)().c_str());
}
template <typename X, DOM::DOMString (X::*Y)() const> void MemberStringFuncGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
  V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)().c_str());
}
template <typename X, DOM::DOMString (X::*Y)(int)> void MemberStringFuncInt(const v8::FunctionCallbackInfo<v8::Value> &args) {
  if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
  V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(args[0]->Int32Value()).c_str());
}
template <typename X, DOM::DOMString (X::*Y)(int)> void IndexedMemberStringProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
  V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(index).c_str());
}
template <typename X, DOM::DOMString (X::*Y)(const DOM::DOMString &)> void NamedMemberStringProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
  string v = BlankNull(*v8::String::Utf8Value(name));
  if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
  V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(DOM::DOMString(v)).c_str());
}

struct MyV8JSInit { MyV8JSInit() { v8::V8::Initialize(); } };

struct MyV8JSContext : public JSContext {
  v8::Isolate*                   isolate;
  v8::Isolate::Scope             isolate_scope;
  v8::HandleScope                handle_scope;
  v8::Handle<v8::Context>        context;
  v8::Context::Scope             context_scope;
  v8::Handle<v8::ObjectTemplate> global, console, window, node, node_list, named_node_map, css_style_declaration;
  Console*                       js_console;

  virtual ~MyV8JSContext() {}
  MyV8JSContext(Console *C, DOM::Node *D) : isolate(v8::Isolate::New()), isolate_scope(isolate),
  handle_scope(isolate), context(v8::Context::New(isolate)), context_scope(context), global(v8::ObjectTemplate::New()),
  console(v8::ObjectTemplate::New()), window(v8::ObjectTemplate::New()), node(v8::ObjectTemplate::New()),
  node_list(v8::ObjectTemplate::New()), named_node_map(v8::ObjectTemplate::New()), css_style_declaration(v8::ObjectTemplate::New()),
  js_console(C) {
    console->SetInternalFieldCount(1);
    console->Set(v8::String::NewFromUtf8(isolate, "log"), v8::FunctionTemplate::New(isolate, consoleLog));
    v8::Local<v8::Object> console_obj = console->NewInstance();
    console_obj->SetInternalField(0, v8::External::New(isolate, this));
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);

    window->SetInternalFieldCount(1);
    window->Set(v8::String::NewFromUtf8(isolate, "getComputedStyle"), v8::FunctionTemplate::New(isolate, windowGetComputedStyle));
    v8::Local<v8::Object> window_obj = window->NewInstance();
    window_obj->SetInternalField(0, v8::External::New(isolate, this));
    window_obj->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "window"), window_obj);

    node->SetInternalFieldCount(3);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeName"),
                      MemberStringFuncGetter<DOM::Node, &DOM::Node::nodeName>, donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeValue"),
                      MemberStringFuncGetter<DOM::Node, &DOM::Node::nodeValue>, donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "childNodes"), 
                      MemberObjectGetter<DOM::Node, DOM::NodeList,
                      &DOM::Node::childNodes, &MyV8JSContext::node_list>, donothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "attributes"), 
                      ElementObjectGetter<DOM::Node, DOM::NamedNodeMap,
                      &DOM::Element::attributes, &MyV8JSContext::named_node_map>, donothingSetter);

    node_list->SetInternalFieldCount(3);
    node_list->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                           MemberIntGetter<DOM::NodeList, &DOM::NodeList::length>, donothingSetter);
    node_list->Set(v8::String::NewFromUtf8(isolate, "item"),
                   v8::FunctionTemplate::New(isolate, MemberObjectFuncInt<DOM::NodeList, DOM::Node, 
                                             &DOM::NodeList::item, &MyV8JSContext::node>));
    node_list->SetIndexedPropertyHandler(IndexedMemberObjectProperty<DOM::NodeList, DOM::Node,
                                         &DOM::NodeList::item, &MyV8JSContext::node>);

    named_node_map->SetInternalFieldCount(3);
    named_node_map->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                MemberIntGetter<DOM::NamedNodeMap, &DOM::NamedNodeMap::length>, donothingSetter);
    named_node_map->Set(v8::String::NewFromUtf8(isolate, "item"),
                        v8::FunctionTemplate::New(isolate, MemberObjectFuncInt<DOM::NamedNodeMap, DOM::Node, 
                                                  &DOM::NamedNodeMap::item, &MyV8JSContext::node>));
    named_node_map->SetIndexedPropertyHandler(IndexedMemberObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                              &DOM::NamedNodeMap::item, &MyV8JSContext::node>);
    named_node_map->SetNamedPropertyHandler(NamedMemberObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                            &DOM::NamedNodeMap::getNamedItem, &MyV8JSContext::node>);

    css_style_declaration->SetInternalFieldCount(3);
    css_style_declaration->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                       MemberIntGetter<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::length>, donothingSetter);
    css_style_declaration->Set(v8::String::NewFromUtf8(isolate, "item"),
                               v8::FunctionTemplate::New(isolate, MemberStringFuncInt<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>));
    css_style_declaration->SetIndexedPropertyHandler(IndexedMemberStringProperty<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>);
    css_style_declaration->SetNamedPropertyHandler(NamedMemberStringProperty<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::getPropertyValue>);

    if (D) {
      v8::Local<v8::Object> node_obj = node->NewInstance();
      node_obj->SetInternalField(0, v8::External::New(isolate, this));
      node_obj->SetInternalField(1, v8::External::New(isolate, D));
      node_obj->SetInternalField(2, v8::Integer::New(isolate, TypeId(D)));
      context->Global()->Set(v8::String::NewFromUtf8(isolate, "document"), node_obj);
    }
  }
  string Execute(const string &s) {
    v8::Handle<v8::String> source = v8::String::NewFromUtf8(isolate, s.c_str());
    v8::Handle<v8::Script> script = v8::Script::Compile(source);
    { v8::TryCatch trycatch;
      v8::Handle<v8::Value> result = script->Run();
      if (!result.IsEmpty()) {
        if (result->IsObject() && js_console) {
          v8::Local<v8::Object> obj = result->ToObject();
          if (obj->InternalFieldCount() >= 3) {
            if (obj->GetInternalField(2)->Int32Value() == TypeId<DOM::Node>()) {
              js_console->Write(CastV8InternalFieldTo<DOM::Node*>(obj, 1)->DebugString());
            }
          }
        }
      } else result = trycatch.Exception();
      return BlankNull(*v8::String::Utf8Value(result));
    }
  }
  static void donothingSetter(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<void>& args) {}
  static void windowGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    args.GetReturnValue().Set(args.Holder());
  }
  static void consoleLog(const v8::FunctionCallbackInfo<v8::Value> &args) {
    v8::Local<v8::Object> self = args.Holder(); string msg;
    MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
    for (int i=0; i < args.Length(); i++) StrAppend(&msg, BlankNull(*v8::String::Utf8Value(args[i]->ToString())));
    if (js_context->js_console) js_context->js_console->Write(msg);
    else INFO("VSJ8(", (void*)js_context, ") console.log: ", msg);
    args.GetReturnValue().Set(v8::Null(args.GetIsolate()));
  };
  static void windowGetComputedStyle(const v8::FunctionCallbackInfo<v8::Value> &args) {
    v8::Local<v8::Object> self = args.Holder();
    if (args.Length() < 1 || !args[0]->IsObject()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    v8::Local<v8::Object> arg_obj = args[0]->ToObject();
    DOM::Node *impl = CastV8InternalFieldTo<DOM::Node*>(arg_obj, 1);
    DOM::CSSStyleDeclaration *val = impl->render ? &impl->render->style : 0;
    if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
    v8::Local<v8::Object> impl_obj = js_context->css_style_declaration->NewInstance();
    impl_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context));
    impl_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val));
    impl_obj->SetInternalField(2, v8::Integer ::New(args.GetIsolate(), TypeId(val)));
    args.GetReturnValue().Set(impl_obj);
  }
  template <typename X, typename Y, Y (X::*Z), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void MemberObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
      V8_ObjectMemberReturn(X, Y, OT, &(impl->*Z));
    }
  template <typename X, typename Y, Y (DOM::Element::*Z), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void ElementObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
      V8_ObjectMemberReturn(X, Y, OT, impl->AsElement() ? &(impl->AsElement()->*Z) : 0);
    }
  template <typename X, typename Y, Y *(X::*Z)(int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void MemberObjectFuncInt(const v8::FunctionCallbackInfo<v8::Value> &args) {
      if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
      V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(args[0]->Int32Value()));
    }
  template <typename X, typename Y, Y *(X::*Z)(int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void IndexedMemberObjectProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
      V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(index));
    }
  template <typename X, typename Y, Y *(X::*Z)(const DOM::DOMString &), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void NamedMemberObjectProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
      string v = BlankNull(*v8::String::Utf8Value(name));
      if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
      V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(DOM::DOMString(v)));
    }
};
JSContext *CreateV8JSContext(Console *js_console, DOM::Node *doc) { Singleton<MyV8JSInit>::Get(); return new MyV8JSContext(js_console, doc); }
#else /* LFL_V8JS */
JSContext *CreateV8JSContext(Console *js_console, DOM::Node *doc) { return 0; }
#endif /* LFL_V8JS */
}; // namespace LFL
