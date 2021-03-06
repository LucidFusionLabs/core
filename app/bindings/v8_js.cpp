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

#include "core/app/gl/view.h"
#include "core/web/browser/browser.h"
#include <libplatform/libplatform.h>
#include <v8.h>

namespace LFL {
v8::Local<v8::String> NewV8String(v8::Isolate *I, const char  *s) { return v8::String::NewFromUtf8(I, s); }
v8::Local<v8::String> NewV8String(v8::Isolate *I, const short *s) { return v8::String::NewFromTwoByte(I, MakeUnsigned(s)); }
template <class X> inline X CastV8InternalFieldTo(v8::Local<v8::Object> &self, int field_index) {
  return static_cast<X>(v8::Local<v8::External>::Cast(self->GetInternalField(field_index))->Value());
}

#define GET_V8_SELF() \
  v8::Local<v8::Object> self = args.Holder(); \
  X *inst = CastV8InternalFieldTo<X*>(self, 1);

#define RETURN_V8_TYPE(X, type, ret) \
  GET_V8_SELF(); \
  args.GetReturnValue().Set(type(args.GetIsolate(), (ret)));

#define RETURN_V8_OBJECT(X, OT, ret) \
  v8::Local<v8::Object> self = args.Holder(); \
  auto val = (ret); \
  if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; } \
  MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0); \
  v8::Local<v8::Object> ret_obj = (js_context->*OT)->NewInstance(); \
  ret_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context)); \
  ret_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val)); \
  ret_obj->SetInternalField(2, v8::External::New(args.GetIsolate(), TypeId<decltype(val)>())); \
  args.GetReturnValue().Set(ret_obj);

#define GET_V8_PROPERTY_STRING() \
  string n = BlankNull(*v8::String::Utf8Value(args.GetIsolate(), name)); \
  if (n == "toString" || n == "valueOf" || n == "length" || n == "item") return;

template <typename X, int       (*f)(const X*)> void IntGetter   (v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) { RETURN_V8_TYPE(X, v8::Integer::New, f(inst)); }
template <typename X, DOMString (*f)(const X*)> void StringGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) { RETURN_V8_TYPE(X, NewV8String,      f(inst).c_str()); }

template <typename X, DOMString (*f)(const X*, int)> void IndexedStringFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
  if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
  RETURN_V8_TYPE(X, NewV8String, f(inst, args[0]->Int32Value(args.GetIsolate()->GetCurrentContext()).FromJust()).c_str());
}
template <typename X, DOMString (*f)(const X*, int)> void IndexedStringProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
  RETURN_V8_TYPE(X, NewV8String, f(inst, index).c_str());
}
template <typename X, DOMString (*f)(const X*, const DOMString &)>
void NamedStringProperty(v8::Local<v8::Name> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
  GET_V8_PROPERTY_STRING();
  RETURN_V8_TYPE(X, NewV8String, f(inst, DOMString(n)).c_str());
}
template <typename X, void (*f)(X*, const DOMString &, const DOMString&)>
void SetNamedStringProperty(v8::Local<v8::Name> name, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<v8::Value> &args) {
  GET_V8_PROPERTY_STRING();
  GET_V8_SELF();
  auto isolate = args.GetIsolate();
  string v = BlankNull(*v8::String::Utf8Value(isolate, value->ToString(isolate->GetCurrentContext()).ToLocalChecked()));
  f(inst, n, v);
}

template <class X, int        (X::*Z)()                 const> int       CallIntMember          (const X *x)                     { return (x->*Z)(); }
template <class X, DOMString  (X::*Z)()                 const> DOMString CallStringMember       (const X *x)                     { return (x->*Z)(); }
template <class X, DOMString  (X::*Z)(int)              const> DOMString CallIndexedStringMember(const X *x, int i)              { return (x->*Z)(i); }
template <class X, DOMString  (X::*Z)(const DOMString&) const> DOMString CallNamedStringMember  (const X *x, const DOMString &n) { return (x->*Z)(n); }
template <class X, void (X::*Z)(const DOMString&, const DOMString&)> void CallNamedSetterMember(X *x, const DOMString &n, const DOMString &v) { return (x->*Z)(n, v); }

template <class X, class Y, Y  (X::*Z)                        > Y* GetObjectMember        (X *x)                     { return &(x->*Z); }
template <class X, class Y, Y  (DOM::Element::*Z)             > Y* GetObjectFromElement   (X *x)                     { return x->AsElement() ? &(x->AsElement()->*Z) : 0; }
template <class X, class Y, Y* (X::*Z)(int)              const> Y* CallIndexedObjectMember(X *x, int i)              { return (x->*Z)(i); }
template <class X, class Y, Y* (X::*Z)(const DOMString&) const> Y* CallNamedObjectMember  (X *x, const DOMString &n) { return (x->*Z)(n); }
template <class X> DOM::CSSStyleDeclaration* GetRenderStyle(X *x) { return x->render ? &x->render->style : 0; }
template <class X> DOM::CSSStyleDeclaration* GetInlineStyle(X *x) {
  if (!x->render) return 0;
  if (!x->render->inline_style.Computed()) x->render->ComputeStyle(x->ownerDocument->inline_style_context, &x->render->inline_style);
  return &x->render->inline_style;
}

struct MyV8JSInit {
  std::unique_ptr<v8::Platform> platform;
  MyV8JSInit() {
    // v8::V8::InitializeICUDefaultLocation("");
    // v8::V8::InitializeExternalStartupData("");
    platform = v8::platform::NewDefaultPlatform();
    v8::V8::InitializePlatform(platform.get());
    v8::V8::Initialize();
  }
  virtual ~MyV8JSInit() {
    v8::V8::Dispose();
    v8::V8::ShutdownPlatform();
  }
};

struct MyV8JSContext : public JSContext {
  struct IsolateDeleter { void operator()(v8::Isolate *isolate) { isolate->Dispose(); } };
  template <typename X, typename Y, Y* (*f)(X*), v8::Local<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void ObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1)));
  }
  template <typename X, typename Y, Y* (*f)(X*, int), v8::Local<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void IndexedObjectFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), args[0]->Int32Value(args.GetIsolate()->GetCurrentContext()).FromJust()));
  }
  template <typename X, typename Y, Y* (*f)(X*), v8::Local<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void ArgumentObjectFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    if (args.Length() < 1 || !args[0]->IsObject()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    v8::Local<v8::Object> arg_obj = args[0]->ToObject(args.GetIsolate()->GetCurrentContext()).ToLocalChecked();
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(arg_obj, 1)));
  }
  template <typename X, typename Y, Y *(*f)(X*, int), v8::Local<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void IndexedObjectProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), index));
  }
  template <typename X, typename Y, Y *(*f)(X*, const DOMString&), v8::Local<v8::ObjectTemplate> (MyV8JSContext::*OT)>
  static void NamedObjectProperty(v8::Local<v8::Name> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
    string v = BlankNull(*v8::String::Utf8Value(args.GetIsolate(), name));
    if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
    RETURN_V8_OBJECT(X, OT, f(CastV8InternalFieldTo<X*>(self, 1), DOMString(v)));
  }

  unique_ptr<v8::ArrayBuffer::Allocator>  allocator;
  unique_ptr<v8::Isolate, IsolateDeleter> isolate_holder;
  v8::Isolate*                            isolate;
  v8::Isolate::Scope                      isolate_scope;
  v8::HandleScope                         handle_scope;
  v8::Local<v8::Context>                  context;
  v8::Context::Scope                      context_scope;
  v8::Local<v8::ObjectTemplate>           global, console, window, node, node_list, named_node_map, css_style_declaration;
  Console*                                js_console;

  virtual ~MyV8JSContext() {}
  MyV8JSContext(Console *C, DOM::Node *D) :
    allocator(v8::ArrayBuffer::Allocator::NewDefaultAllocator()),
    isolate_holder(v8::Isolate::New(IsolateCreateParams(allocator.get()))), isolate(isolate_holder.get()),
    isolate_scope(isolate), handle_scope(isolate), context(v8::Context::New(isolate)), context_scope(context),
    global(v8::ObjectTemplate::New(isolate)), console(v8::ObjectTemplate::New(isolate)),
    window(v8::ObjectTemplate::New(isolate)), node(v8::ObjectTemplate::New(isolate)),
    node_list(v8::ObjectTemplate::New(isolate)), named_node_map(v8::ObjectTemplate::New(isolate)),
    css_style_declaration(v8::ObjectTemplate::New(isolate)),
    js_console(C) {

    console->SetInternalFieldCount(1);
    console->Set(v8::String::NewFromUtf8(isolate, "log"), v8::FunctionTemplate::New(isolate, ConsoleLog));
    v8::Local<v8::Object> console_obj = console->NewInstance();
    console_obj->SetInternalField(0, v8::External::New(isolate, this));
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);

    window->SetInternalFieldCount(1);
    window->Set(v8::String::NewFromUtf8(isolate, "getComputedStyle"), v8::FunctionTemplate::New
                (isolate, ArgumentObjectFunc<DOM::Node, DOM::CSSStyleDeclaration, &GetRenderStyle<DOM::Node>, &MyV8JSContext::css_style_declaration>));
    v8::Local<v8::Object> window_obj = window->NewInstance();
    window_obj->SetInternalField(0, v8::External::New(isolate, this));
    window_obj->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);
    context->Global()->Set(v8::String::NewFromUtf8(isolate, "window"), window_obj);

    node->SetInternalFieldCount(3);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeName"),
                      StringGetter<DOM::Node, CallStringMember<DOM::Node, &DOM::Node::nodeName>>, DoNothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeValue"),
                      StringGetter<DOM::Node, CallStringMember<DOM::Node, &DOM::Node::nodeValue>>, DoNothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "childNodes"),
                      ObjectGetter<DOM::Node, DOM::NodeList, &GetObjectMember<DOM::Node, DOM::NodeList, &DOM::Node::childNodes>, &MyV8JSContext::node_list>,
                      DoNothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "attributes"), 
                      ObjectGetter<DOM::Node, DOM::NamedNodeMap, &GetObjectFromElement<DOM::Node, DOM::NamedNodeMap, &DOM::Element::attributes>, &MyV8JSContext::named_node_map>,
                      DoNothingSetter);
    node->SetAccessor(v8::String::NewFromUtf8(isolate, "style"),
                      ObjectGetter<DOM::Node, DOM::CSSStyleDeclaration, &GetInlineStyle<DOM::Node>, &MyV8JSContext::css_style_declaration>,
                      DoNothingSetter);
    node_list->SetInternalFieldCount(3);
    node_list->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                           IntGetter<DOM::NodeList, &CallIntMember<DOM::NodeList, &DOM::NodeList::length>>, DoNothingSetter);
    node_list->Set(v8::String::NewFromUtf8(isolate, "item"),
                   v8::FunctionTemplate::New(isolate, IndexedObjectFunc<DOM::NodeList, DOM::Node, 
                                             &CallIndexedObjectMember<DOM::NodeList, DOM::Node, &DOM::NodeList::item>, &MyV8JSContext::node>));
    node_list->SetIndexedPropertyHandler(IndexedObjectProperty<DOM::NodeList, DOM::Node,
                                         &CallIndexedObjectMember<DOM::NodeList, DOM::Node, &DOM::NodeList::item>, &MyV8JSContext::node>);

    named_node_map->SetInternalFieldCount(3);
    named_node_map->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                IntGetter<DOM::NamedNodeMap, &CallIntMember<DOM::NamedNodeMap, &DOM::NamedNodeMap::length>>, DoNothingSetter);
    named_node_map->Set(v8::String::NewFromUtf8(isolate, "item"),
                        v8::FunctionTemplate::New(isolate, IndexedObjectFunc<DOM::NamedNodeMap, DOM::Node, 
                                                  CallIndexedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::item>, &MyV8JSContext::node>));
    named_node_map->SetIndexedPropertyHandler(IndexedObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                              &CallIndexedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::item>, &MyV8JSContext::node>);
    named_node_map->SetHandler
      (v8::NamedPropertyHandlerConfiguration
       (NamedObjectProperty<DOM::NamedNodeMap, DOM::Node,
        &CallNamedObjectMember<DOM::NamedNodeMap, DOM::Node, &DOM::NamedNodeMap::getNamedItem>, &MyV8JSContext::node>));

    css_style_declaration->SetInternalFieldCount(3);
    css_style_declaration->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                       IntGetter<DOM::CSSStyleDeclaration, &CallIntMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::length>>, DoNothingSetter);
    css_style_declaration->Set(v8::String::NewFromUtf8(isolate, "item"),
                               v8::FunctionTemplate::New(isolate, IndexedStringFunc<DOM::CSSStyleDeclaration, CallIndexedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>>));
    css_style_declaration->SetIndexedPropertyHandler(IndexedStringProperty<DOM::CSSStyleDeclaration, &CallIndexedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>>);
    css_style_declaration->SetHandler
      (v8::NamedPropertyHandlerConfiguration
       (   NamedStringProperty<DOM::CSSStyleDeclaration, &CallNamedStringMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::getPropertyValue>>,
        SetNamedStringProperty<DOM::CSSStyleDeclaration, &CallNamedSetterMember<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::setPropertyValue>>));

    if (D) {
      v8::Local<v8::Object> node_obj = node->NewInstance();
      node_obj->SetInternalField(0, v8::External::New(isolate, this));
      node_obj->SetInternalField(1, v8::External::New(isolate, D));
      node_obj->SetInternalField(2, v8::External::New(isolate, TypeId<DOM::Node*>()));
      context->Global()->Set(v8::String::NewFromUtf8(isolate, "document"), node_obj);
    }
  }

  string Execute(const string &s) {
    v8::Local<v8::String> source = v8::String::NewFromUtf8(isolate, s.c_str());
    v8::Local<v8::Script> script = v8::Script::Compile(isolate->GetCurrentContext(), source).ToLocalChecked();
    { v8::TryCatch trycatch(isolate);
      v8::Local<v8::Value> result;
      auto maybe_result = script->Run(isolate->GetCurrentContext());
      if (maybe_result.ToLocal(&result) && !result.IsEmpty()) {
        if (result->IsObject() && js_console) {
          v8::Local<v8::Object> obj;
          auto maybe_obj = result->ToObject(isolate->GetCurrentContext());
          if (maybe_obj.ToLocal(&obj) && obj->InternalFieldCount() >= 3) {
            if (CastV8InternalFieldTo<Void>(obj, 2) == TypeId<DOM::Node*>()) {
              js_console->Write(CastV8InternalFieldTo<DOM::Node*>(obj, 1)->DebugString());
            }
          }
        }
      } else result = trycatch.Exception();
      return BlankNull(*v8::String::Utf8Value(isolate, result));
    }
  }

  static v8::Isolate::CreateParams IsolateCreateParams(v8::ArrayBuffer::Allocator *alloc) {
    v8::Isolate::CreateParams create_params;
    create_params.array_buffer_allocator = alloc;
    return create_params;
  }

  static void DoNothingSetter(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<void>& args) {}

  static void WindowGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    args.GetReturnValue().Set(args.Holder());
  }

  static void ConsoleLog(const v8::FunctionCallbackInfo<v8::Value> &args) {
    v8::Local<v8::Object> self = args.Holder(); string msg;
    MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
    for (int i=0; i < args.Length(); i++) StrAppend(&msg, BlankNull(*v8::String::Utf8Value(args.GetIsolate(), args[i]->ToString(js_context->isolate->GetCurrentContext()).ToLocalChecked())));
    if (js_context->js_console) js_context->js_console->Write(msg);
    else INFO("VSJ8(", Void(js_context), ") console.log: ", msg);
    args.GetReturnValue().Set(v8::Null(args.GetIsolate()));
  };
};

unique_ptr<JSContext> JSContext::Create(Console *js_console, DOM::Node *doc) {
  Singleton<MyV8JSInit>::Get();
  return make_unique<MyV8JSContext>(js_console, doc);
}

}; // namespace LFL
