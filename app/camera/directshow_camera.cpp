/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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
#include <comdef.h>
#include <dshow.h>

#ifndef __ISampleGrabberCB_INTERFACE_DEFINED__
#define __ISampleGrabberCB_INTERFACE_DEFINED__
/* interface ISampleGrabberCB */
/* [unique][helpstring][local][uuid][object] */ 
EXTERN_C const IID IID_ISampleGrabberCB;
MIDL_INTERFACE("0579154A-2B53-4994-B0D0-E773148EFF85")
ISampleGrabberCB : public IUnknown {
  public:
    virtual HRESULT STDMETHODCALLTYPE SampleCB(double SampleTime, IMediaSample *pSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};
#endif 	/* __ISampleGrabberCB_INTERFACE_DEFINED__ */

#ifndef __ISampleGrabber_INTERFACE_DEFINED__
#define __ISampleGrabber_INTERFACE_DEFINED__
/* interface ISampleGrabber */
/* [unique][helpstring][local][uuid][object] */ 
EXTERN_C const IID IID_ISampleGrabber;
MIDL_INTERFACE("6B652FFF-11FE-4fce-92AD-0266B5D7C78F")
ISampleGrabber : public IUnknown {
  public:
    virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};
#endif /* __ISampleGrabber_INTERFACE_DEFINED__ */

EXTERN_C const CLSID CLSID_SampleGrabber;
class DECLSPEC_UUID("C1F400A0-3F08-11d3-9F0B-006008039E37")
SampleGrabber;

namespace LFL {
template <class T> struct CComPtr {
  T *p;
  CComPtr() : p(0) {}
  CComPtr(T *P) { if ((p = P)) p->AddRef(); }
  CComPtr(const CComPtr<T>& P) { if ((p = P.p)) p->AddRef(); }
  ~CComPtr() { if (p) p->Release(); }
  void Release() { if (p) p->Release(); p=0; }
  operator T*() { return (T*)p; }
  T& operator*() { return *p; }
  T** operator&() { return &p; }
  T* operator->() { return p; }    
};

template <class T> struct SampleGrabberThunker : public ISampleGrabberCB {
  T *tp; int refcount;
  SampleGrabberThunker(T *TP) : tp(TP), refcount(0) {}
  ULONG WINAPI AddRef() { return ++refcount; }
  ULONG WINAPI Release() { return --refcount; }
  HRESULT WINAPI QueryInterface(REFIID iid, void **out) {
    if (iid != IID_IUnknown && iid != IID_ISampleGrabberCB) return E_NOINTERFACE;
    *out = (void *)this; 
    AddRef();
    return S_OK;
  }
  HRESULT WINAPI BufferCB(double time, BYTE *buf, long len) { return E_NOTIMPL; }
  HRESULT WINAPI SampleCB(double time, IMediaSample *pSample) {
    return tp->SampleGrabberCB(time, pSample);
  }
};

struct DirectShowCameraModule : public Module {
  CComPtr<IGraphBuilder> pGraph;
  CComPtr<IMediaControl> pMC;
  CComPtr<IMediaEventEx> pME;

  CComPtr<ICaptureGraphBuilder2> pCapture;
  CComPtr<IAMStreamConfig> pStreamConfig;

  CComPtr<IBaseFilter> pVideoSource;
  CComPtr<IPin> pCapturePin;

  CComPtr<ISampleGrabber> pGrabber;
  SampleGrabberThunker<DirectShowCamera> thunker;

  unique_ptr<RingSampler> frames;
  int next=0;
  mutex lock;
  bool invert=0;

  CameraState *camera;
  DirectShowCameraModule(CameraState *C) : thunker(this), camera(C) {}

  int Free() {
    pMC->Stop();
    return 0;
  }

  int Init() {
    FLAGS_lfapp_camera = 0;
    if (FAILED(CoInitialize(0))) return 0;

    /* create filter graph manager */
    if (FAILED(CoCreateInstance(CLSID_FilterGraph, 0, CLSCTX_INPROC,
                                IID_IGraphBuilder, (void**)&pGraph))) return 0;

    /* get media control interface */
    if (FAILED(pGraph->QueryInterface(IID_IMediaControl, (void**)&pMC))) return 0;

    /* get media event interface */
    if (FAILED(pGraph->QueryInterface(IID_IMediaEvent, (void**)&pME))) return -1;

    /* create capture graph builder */
    if (FAILED(CoCreateInstance(CLSID_CaptureGraphBuilder2, 0, CLSCTX_INPROC,
                                IID_ICaptureGraphBuilder2, (void **)&pCapture))) return -1;

    /* provide capture graph builter with filter graph */
    if (FAILED(pCapture->SetFiltergraph(pGraph))) return -1;

    /* create system device enumerator */
    CComPtr<ICreateDevEnum> pDevEnum;
    if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, 0, CLSCTX_INPROC,
                                IID_ICreateDevEnum, (void**)&pDevEnum))) return 0;

    /* create video input device enumerator */
    CComPtr<IEnumMoniker> pVidEnum;
    if (FAILED(pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                               &pVidEnum, 0))) return 0;

    /* no video input devices */
    if (!pVidEnum) return 0;

    /* optionally log video device list */
    CComPtr<IMoniker> pMoniker; ULONG cFetched;
    if (FLAGS_print_camera_devices) {
      for (int i=0; /**/; i++) {
        pMoniker.Release();
        if (pVidEnum->Next(1, &pMoniker, &cFetched) != S_OK) break;

        CComPtr<IPropertyBag> pProp;
        pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pProp);

        VARIANT name; VariantInit(&name);
        if (FAILED(pProp->Read(L"FriendlyName", &name, 0))) continue;

        _bstr_t n = name;
        INFO("Camera device ", i, " = ", (char*)n);
      }
      pVidEnum->Reset();
    }

    /* iterate video input devices and retrieve device by index */
    for (int i=0; i<=FLAGS_camera_device; i++) {
      pMoniker.Release();
      if (pVidEnum->Next(1, &pMoniker, &cFetched) != S_OK) return 0;
      if (i != FLAGS_camera_device) continue;

      CComPtr<IPropertyBag> pProp;
      pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pProp);

      VARIANT name; VariantInit(&name);
      if (FAILED(pProp->Read(L"FriendlyName", &name, 0))) continue;

      _bstr_t n = name;
      INFO("Opened camera ", i, " = ", (char*)n);
    }

    /* open video input device */
    if (FAILED(pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&pVideoSource))) return 0;

    /* add video input device to filter graph */
    pGraph->AddFilter(pVideoSource, L"Video Capture");

    /* get video input device stream config */
    if (FAILED(pCapture->FindInterface(&PIN_CATEGORY_CAPTURE,NULL, pVideoSource,
                                       IID_IAMStreamConfig,(void**)&pStreamConfig))) return 0;

    /* get video input device output pin */
    CComPtr<IEnumPins> pPinEnum;
    if (FAILED(pVideoSource->EnumPins(&pPinEnum))) return 0;
    if (FAILED(pPinEnum->Next(1, &pCapturePin, 0))) return 0;

    /* create video input device output pin media type enumerator */
    CComPtr<IEnumMediaTypes> pMTEnum;
    if (FAILED(pCapturePin->EnumMediaTypes(&pMTEnum))) return 0;

    /* iterate media types and retrieve by matching width,height,fps,format */
    AM_MEDIA_TYPE *pMT; VIDEOINFOHEADER *vih=0;
    bool match=0, mediatype_enumerate=(FLAGS_print_camera_devices==true);
    while (pMTEnum->Next(1, &pMT, 0) == S_OK) {
      vih = (VIDEOINFOHEADER *)pMT->pbFormat;
      int fps = round_f(10000000.0/vih->AvgTimePerFrame);
      int fmt = MST2PF(pMT->subtype, MST2invert(pMT->subtype), mediatype_enumerate);

      if ((!FLAGS_camera_image_width  || vih->bmiHeader.biWidth  == FLAGS_camera_image_width) &&
          (!FLAGS_camera_image_height || vih->bmiHeader.biHeight == FLAGS_camera_image_height) &&
          (!FLAGS_camera_fps || fps >= FLAGS_camera_fps) &&
          ((!image_format && fmt) ||
           (image_format && image_format == fmt))
         )
      {
        match = true;                
      }

      if (mediatype_enumerate)
        INFO("mt = ", vih->bmiHeader.biWidth, " x ", vih->bmiHeader.biHeight, " @ ", fps, " fps, support=", fmt);

      if (match) break;
      else CoTaskMemFree((void*)pMT);
    }
    if (!match) { INFO("no matching media type found"); return 0; }

    /* set frame rate */
    if (FLAGS_camera_fps) vih->AvgTimePerFrame = 10000000.0/FLAGS_camera_fps;

    /* set media type on video input device's stream config */
    if (FAILED(pStreamConfig->SetFormat(pMT))) return 0;

    /* create sample grabber filter */
    CComPtr<IBaseFilter> pSampleGrabber;
    if (FAILED(CoCreateInstance(CLSID_SampleGrabber, 0, CLSCTX_INPROC,
                                IID_IBaseFilter, (void**)&pSampleGrabber))) return 0;

    /* get sample grabber interface */
    if (FAILED(pSampleGrabber->QueryInterface(IID_ISampleGrabber, (void**)&pGrabber))) return 0;

    /* add sample grabber to filter graph */
    pGraph->AddFilter(pSampleGrabber, L"Sample Grabber");

    /* get sample grabber input pin */ 
    CComPtr<IPin> pGrabberPin;
    pPinEnum.Release();
    if (FAILED(pSampleGrabber->EnumPins(&pPinEnum))) return 0;
    if (FAILED(pPinEnum->Next(1, &pGrabberPin, 0))) return 0;

    /* connect video device output pin to sample grabber input pin */
    if (FAILED(pGraph->Connect(pCapturePin, pGrabberPin))) return 0;

    /* get thunker sample grabber callback interface */
    CComPtr<ISampleGrabberCB> pThunkerCB;
    if (FAILED(thunker.QueryInterface(IID_ISampleGrabberCB, (void**)&pThunkerCB))) return -1;

    /* provide our callback to sample grabber */
    if (FAILED(pGrabber->SetCallback(pThunkerCB, 0))) return 0;

    /* start filter graph */
    if (FAILED(pMC->Run())) return 0;

    /* expose capture parameters */
    invert = MST2invert(pMT->subtype);
    FLAGS_camera_image_width = vih->bmiHeader.biWidth;
    FLAGS_camera_image_height = vih->bmiHeader.biHeight;
    FLAGS_camera_fps = round_f(10000000.0/vih->AvgTimePerFrame);
    image_format = MST2PF(pMT->subtype, invert);
    CoTaskMemFree((void*)pMT);

    image_linesize = FLAGS_camera_image_width * Pixel::size(image_format);

    /* create ring buffer */
    frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_height*image_linesize);

    /* success */
    FLAGS_lfapp_camera = 1;
    INFO("opened camera ", FLAGS_camera_image_width, "x", FLAGS_camera_image_height, " @ ", FLAGS_camera_fps, " fps");
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (camera->frames_read > camera->last_frames_read) {

        /* frame align stream handle */
        next = frames->ring.back;

        new_frame = true;
      }
      camera->last_frames_read = camera->frames_read;
    }
    if (!new_frame) return 0;

    camera->image = (unsigned char*)frames->read(-1, next);
    camera->image_timestamp = frames->readtimestamp(-1, next);
    return 1;
  }

  HRESULT SampleGrabberCB(double time, IMediaSample *pSample) {
    /* get sample buffer */
    char *buf;
    if (FAILED(pSample->GetPointer((BYTE**)&buf))) return S_OK;

    /* copy and optionally invert */ 
    char *out = (char*)frames->write(RingSampler::Peek | RingSampler::Stamp);
    if (invert) {
      for (int len=frames->width, i=0; i<frames->width; i++) out[len-1-i] = buf[i];
    }
    else {
      memcpy(out, buf, frames->width);
    }

    /* commit */
    {
      ScopedMutex ML(lock);
      frames->write();
      camera->frames_read++;
    }
    return S_OK;
  }

  static bool MST2invert(GUID mst) { return MST2PF(mst, false) != MST2PF(mst, true); }
  static int MST2PF(GUID mst, bool invert, bool logerror=false) {
    if      (mst == MEDIASUBTYPE_RGB24)  return invert ? Pixel::RGB24   : Pixel::BGR24;
    else if (mst == MEDIASUBTYPE_RGB32)  return invert ? Pixel::RGB32   : Pixel::BGR32;
    else if (mst == MEDIASUBTYPE_RGB555) return invert ? Pixel::RGB555  : Pixel::BGR555;
    else if (mst == MEDIASUBTYPE_RGB565) return invert ? Pixel::RGB565  : Pixel::BGR565;
    else if (mst == MEDIASUBTYPE_YUY2)   return invert ? Pixel::YUYV422 : Pixel::YUYV422;
    else {
      if (logerror) {
        OLECHAR *bstrGuid;
        StringFromCLSID(mst, &bstrGuid);
        _bstr_t n = bstrGuid;
        ERROR("unknown pixel format : ", (char*)n);
        CoTaskMemFree(bstrGuid);
      }
      return 0;
    }
  }

  static void PinEnum(IEnumPins *pPinEnum) {
    CComPtr<IPin> pPin; PIN_INFO pin; int pinId=0; 
    while (pPinEnum->Next(1, &pPin, 0) == S_OK) {
      pPin->QueryPinInfo(&pin);
      pin.pFilter->Release();
      pPin.Release();
      INFO("PinEnum ", pinId++, " ", pin.dir);
    }
    pPinEnum->Reset();
  }
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new DirectShowCameraModule(state); }

}; // namespace LFL
