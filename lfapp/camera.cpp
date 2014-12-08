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

#ifdef LFL_OPENCV_CAMERA
#include "opencv/highgui.h"
#endif

#include "lfapp/lfapp.h"

#ifdef LFL_CAMERA
#ifdef _WIN32
#define LFL_DIRECTSHOW_CAMERA
#endif

#ifdef LFL_IPHONE
#ifndef LFL_IPHONESIM
#define LFL_AVCAPTURE_CAMERA
void AVCaptureInit(int fps);
void AVCaptureShutdown();
#endif // LFL_IPHONESIM
#else // LFL_IPHONE
#ifdef __APPLE__
#ifdef LFL64
#define LFL_QTKIT_CAMERA
void QTKitInit(int fps);
void QTKitShutdown();
#else
#define LFL_QUICKTIME_CAMERA 
#endif // LFL64
#endif // __APPLE__
#endif // LFL_IPHONE

#if defined(__linux__) && !defined(LFL_ANDROID)
#define LFL_FFMPEG_CAMERA
#endif
#endif /* LFL_CAMERA */

extern "C" {
#ifdef LFL_FFMPEG_CAMERA
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#endif
};
#ifdef LFL_DSVL_CAMERA
#include "dsvl.h"
#endif
#ifdef LFL_QUICKTIME_CAMERA
#include <QuickTime/QuickTimeComponents.h>
#endif

namespace LFL {
DEFINE_int(camera_fps, 20, "Camera capture frames per second");
DEFINE_int(camera_device, 0, "Camera device index");
DEFINE_bool(print_camera_devices, false, "Print camera device list");
DEFINE_int(camera_image_width, 0, "Camera capture image width");
DEFINE_int(camera_image_height, 0, "Camera capture image height");

#ifdef LFL_OPENCV_CAMERA
LFL_IMPORT extern int OPENCV_FPS;

struct OpenCvCamera : public CameraInterface {
    Thread thread;
    Mutex lock;

    struct Camera { 
        CvCapture *capture;
        int width, height;
        void dimensions(int W, int H) { width=W; height=H; }

        RingBuf *frames;
        int next;

        Camera() : capture(0), width(0), height(0), frames(0), next(0) {}
    } L, R;

    virtual ~OpenCvCamera() {}
    OpenCvCamera() : thread(threadthunk, this) {}

    int init() {
        OPENCV_FPS = FLAGS_camera_fps;

        if (!(L.capture = cvCaptureFromCAM(0))) { FLAGS_lfapp_camera=0; return 0; }
        // if (!(R.capture = cvCaptureFromCAM(1))) { /**/ }

        if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }
        return 0;
    }

    static int threadthunk(void *p) { return ((OpenCvCamera *)p)->threadproc(); }

    int threadproc() {
        while (app->run) {
            /* grab */
            bool lg=0, rg=0;
            if (L.capture) lg = cvGrabFrame(L.capture);
            if (R.capture) rg = cvGrabFrame(R.capture);

            /* retrieve */
            IplImage *lf=0, *rf=0;
            if (lg) lf = cvRetrieveFrame(L.capture);
            if (rg) rf = cvRetrieveFrame(R.capture);

            /* 1-time cosntruct */
            if (lf && !L.frames) {
                L.dimensions(lf->width, lf->height);
                L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, lf->imageSize);
            }
            if (rf && !R.frames) {
                R.dimensions(rf->width, rf->height);
                R.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, rf->imageSize);
            }

            /* write */
            if (lf) memcpy(L.frames->write(RingBuf::Peek | RingBuf::Stamp), lf->imageData, lf->imageSize);
            if (rf) memcpy(R.frames->write(RingBuf::Peek | RingBuf::Stamp), rf->imageData, rf->imageSize);

            /* commit */  
            if (lf || rf) {
                ScopedMutex ML(lock);
                if (lf) L.frames->write();
                if (rf) R.frames->write();
                frames_read++;
            }
            else Msleep(1);
        }
        return 0;
    };

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.capture) L.next = L.frames->ring.back;
                if (R.capture) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }

        if (!new_frame) return 0;

        image = (unsigned char*)L.frames->read(-1, L.next);
        image_timestamp = L.frames->readtimestamp(-1, L.next);
        FLAGS_camera_image_width = L.width;
        FLAGS_camera_image_height = L.height;

        image_format = Pixel::BGR24;
        image_linesize = image_width*3;

        return 1;
    }

    int free() {
        if (thread.started) thread.Wait();
        if (&L.capture) cvReleaseCapture(&L.capture);
        if (&R.capture) cvReleaseCapture(&R.capture);
        return 0;
    }
};
#endif /* LFL_OPENCV_CAMERA */

#ifdef LFL_FFMPEG_CAMERA
struct FFmpegCamera : public CameraInterface {
    Thread thread;
    Mutex lock;

    struct Camera { 
        AVFormatContext *fctx;

        struct FramePtr {
            AVPacket data;
            bool dirty;

            FramePtr(bool D=0) : dirty(D) {}
            ~FramePtr() { if (dirty) free(); }
            void free() { av_free_packet(&data); clear(); }
            void clear() { dirty=0; }
        
            static void swap(FramePtr *A, FramePtr *B) { FramePtr C = *A; *A = *B; *B = C; C.clear(); }
        };

        RingBuf *frames;
        int next;

        Camera() : fctx(0), frames(0), next(0) {}
    } L, R;

    virtual ~FFmpegCamera() {}
    FFmpegCamera() : thread(threadthunk, this) {}

    int init() {
#ifdef _WIN32
        static const char *ifmtname = "vfwcap";
        static const char *ifilename[2] = { "0", "1" };
#endif
#ifdef __linux__
        static const char *ifmtname = "video4linux2";
        static const char *ifilename[2] = { "/dev/video0", "/dev/video1" };
#endif
        avdevice_register_all();
        AVInputFormat *ifmt = av_find_input_format(ifmtname);

        AVDictionary *options = 0;
        av_dict_set(&options, "video_size", "640x480", 0);

        L.fctx = 0;
        if (avformat_open_input(&L.fctx, ifilename[0], ifmt, &options) < 0) {
            FLAGS_lfapp_camera = 0;
            return 0;
        }
        L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, sizeof(Camera::FramePtr));
        av_dict_free(&options);

        if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }

        AVCodecContext *codec = L.fctx->streams[0]->codec;
        FLAGS_camera_image_width = codec->width;
        FLAGS_camera_image_height = codec->height;
        image_format = Pixel::FromFFMpegId(codec->pix_fmt);
        image_linesize = FLAGS_camera_image_width * Pixel::size(image_format);
        return 0;
    }

    static int threadthunk(void *p) { return ((FFmpegCamera *)p)->threadproc(); }

    int threadproc() {
        while (app->run) {
            /* grab */
            Camera::FramePtr Lframe(0);
            if (av_read_frame(L.fctx, &Lframe.data) < 0) { ERROR("av_read_frame"); return -1; }
            else Lframe.dirty = 1;

            Camera::FramePtr::swap((Camera::FramePtr*)L.frames->write(RingBuf::Peek | RingBuf::Stamp), &Lframe);
            
            /* commit */  
            {
                ScopedMutex ML(lock);
                L.frames->write();
                frames_read++;
            }
        }
        return 0;
    };

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.fctx) L.next = L.frames->ring.back;
                if (R.fctx) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        AVPacket *f = (AVPacket*)L.frames->read(-1, L.next);
        image = f->data;
        image_timestamp = L.frames->readtimestamp(-1, L.next);
        return 1;
    }

    int free() {
        if (thread.started) thread.Wait();
        return 0;
    }
};
#endif /* LFL_FFMPEG_CAMERA */

#ifdef LFL_DSVL_CAMERA
struct DsvlCamera : public CameraInterface {
    Thread thread;
    Mutex lock;

    struct Camera { 
        DSVL_VideoSource *vs;
        RingBuf *frames;
        int next;

        Camera() : vs(0), frames(0), next(0) {}
    } L, R;

    virtual ~DsvlCamera() {}
    DsvlCamera() : thread(threadthunk, this) {}

    int init() {
        char config[] = "<?xml version='1.0' encoding='UTF-8'?> <dsvl_input>"
            "<camera frame_rate='5.0' show_format_dialog='true'>"
            "<pixel_format><RGB24 flip_v='true'/></pixel_format>"
            "</camera></dsvl_input>\r\n";

        CoInitialize(0);
	    L.vs = new DSVL_VideoSource();

        LONG w, h; double fps; PIXELFORMAT pf;
        if (FAILED(L.vs->BuildGraphFromXMLString(config))) return -1;
        if (FAILED(L.vs->GetCurrentMediaFormat(&w, &h, &fps, &pf))) return -1;
        if (FAILED(L.vs->EnableMemoryBuffer())) return -1;
        if (FAILED(L.vs->Run())) return -1;

        FLAGS_camera_image_width = w;
        FLAGS_camera_image_height = h;
        FLAGS_camera_fps = fps;

        int depth;
        if (pf == PIXELFORMAT_RGB24) {
            depth = 3;
            image_format = Pixel::RGB24;
            image_linesize = FLAGS_camera_image_width * depth;
        }
        else { ERROR("unknown pixel format: ", pf); return -1; }

        L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_width*FLAGS_camera_image_height*depth);

        if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }

        INFO("opened camera ", FLAGS_camera_image_width, "x", FLAGS_camera_image_height, " @ ", FLAGS_camera_fps, "fps");
        return 0;
    }

    static int threadthunk(void *p) { return ((DsvlCamera *)p)->threadproc(); }

    int threadproc() {
        while (app->run) {
            DWORD ret = L.vs->WaitForNextSample(1000/FLAGS_camera_fps);
            if (ret != WAIT_OBJECT_0) continue;

            MemoryBufferHandle h; char *b;
            L.vs->CheckoutMemoryBuffer(&h, (BYTE**)&b);
            memcpy(L.frames->write(RingBuf::Peek | RingBuf::Stamp), b, L.frames->width);
            L.vs->CheckinMemoryBuffer(h);

            { /* commit */ 
                ScopedMutex ML(lock);
                L.frames->write();
                frames_read++;
            }
        }
        return 0;
    };

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.vs) L.next = L.frames->ring.back;
                if (R.vs) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        image = (unsigned char*)L.frames->read(-1, L.next);
        image_timestamp = L.frames->readtimestamp(-1, L.next);
        return 1;
    }

    int free() {
        if (thread.started) thread.Wait();
        return 0;
    }
};
#endif /* LFL_DSVL_CAMERA */

#ifdef LFL_DIRECTSHOW_CAMERA
}; // namespace LFL
#include <comdef.h>
#include <dshow.h>

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

struct DirectShowCamera : public CameraInterface {
    CComPtr<IGraphBuilder> pGraph;
    CComPtr<IMediaControl> pMC;
    CComPtr<IMediaEventEx> pME;

    CComPtr<ICaptureGraphBuilder2> pCapture;
    CComPtr<IAMStreamConfig> pStreamConfig;

    CComPtr<IBaseFilter> pVideoSource;
    CComPtr<IPin> pCapturePin;

    CComPtr<ISampleGrabber> pGrabber;
    SampleGrabberThunker<DirectShowCamera> thunker;

    RingBuf *frames;
    int next;
    Mutex lock;
    bool invert;

    virtual ~DirectShowCamera() { delete frames; }
    DirectShowCamera() : thunker(this), frames(0), next(0), invert(0) {}

    int init() {
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
            int fmt = mst2pf(pMT->subtype, mst2invert(pMT->subtype), mediatype_enumerate);

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
        invert = mst2invert(pMT->subtype);
        FLAGS_camera_image_width = vih->bmiHeader.biWidth;
        FLAGS_camera_image_height = vih->bmiHeader.biHeight;
        FLAGS_camera_fps = round_f(10000000.0/vih->AvgTimePerFrame);
        image_format = mst2pf(pMT->subtype, invert);
        CoTaskMemFree((void*)pMT);

        image_linesize = FLAGS_camera_image_width * Pixel::size(image_format);

        /* create ring buffer */
        frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_height*image_linesize);

        /* success */
        FLAGS_lfapp_camera = 1;
        INFO("opened camera ", FLAGS_camera_image_width, "x", FLAGS_camera_image_height, " @ ", FLAGS_camera_fps, " fps");
        return 0;
    }

    static int mst2pf(GUID mst, bool invert, bool logerror=false) {
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

    static bool mst2invert(GUID mst) { return mst2pf(mst, false) != mst2pf(mst, true); }

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

    HRESULT SampleGrabberCB(double time, IMediaSample *pSample) {
        /* get sample buffer */
        char *buf;
        if (FAILED(pSample->GetPointer((BYTE**)&buf))) return S_OK;

        /* copy and optionally invert */ 
        char *out = (char*)frames->write(RingBuf::Peek | RingBuf::Stamp);
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
            frames_read++;
        }
        return S_OK;
    }

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handle */
                next = frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        image = (unsigned char*)frames->read(-1, next);
        image_timestamp = frames->readtimestamp(-1, next);
        return 1;
    }

    int free() {
        pMC->Stop();
        return 0;
    }
};
#endif /* LFL_DIRECTSHOW_CAMERA */

#ifdef LFL_QUICKTIME_CAMERA
struct QuickTimeCamera : public CameraInterface {
    Thread thread;
    VideoResampler conv;
    Mutex lock;

    struct Camera {
        SeqGrabComponent grabber;
        SGChannel channel;
        GWorldPtr gworld;
        Rect rect;
        PixMapHandle pixmap;
        ImageSequence seq;

        RingBuf *frames;
        int next;
        Camera() : grabber(0), channel(0), gworld(0), seq(0), frames(0), next(0) {}
    } L, R;

    virtual ~QuickTimeCamera() {}
    QuickTimeCamera() : thread(threadthunk, this) {}

    int init() {
        image_format = Pixel::BGR24;
        int image_depth = Pixel::size(image_format);

        /* open device */
        SGDeviceList devlist; int devind=-1, di=0, ret;
        if ((ret = EnterMovies()) != noErr) { ERROR("EnterMovies: ", ret); return -1; }
        if (!(L.grabber = OpenDefaultComponent(SeqGrabComponentType, 0))) { ERROR("OpenDefaultComonent: ", L.grabber); return -1; }
        if ((ret = SGInitialize(L.grabber)) != noErr) { ERROR("SGInitialize: ", ret); return -1; }
        if ((ret = SGSetDataRef(L.grabber, 0, 0, seqGrabDontMakeMovie)) != noErr) { ERROR("SGSetDataRef: ", ret); return -1; }
        if ((ret = SGNewChannel(L.grabber, VideoMediaType, &L.channel)) != noErr) { ERROR("SGNewChannel: ", ret); return -1; }
        if ((ret = SGGetChannelDeviceList(L.channel, 0, &devlist)) != noErr) { ERROR("SGGetChannelDeviceList: ", ret); return -1; }
        for (int i=0; i<(*devlist)->count; i++) {
            SGDeviceName devname = (*devlist)->entry[i];
            if (devname.flags) continue;
            if (FLAGS_print_camera_devices) INFO("camera device ", di, " = ", devname.name);
            if (di != FLAGS_camera_device) { di++; continue; }
            devind = i;
            break;
        }
        if (devind < 0) { FLAGS_lfapp_camera=0; return 0; }
        SGDeviceName devname = (*devlist)->entry[devind];
        if ((ret = SGSetChannelDevice(L.channel, devname.name)) != noErr) { ERROR("SGSetChannelDevice: ", ret); return -1; }
        if ((ret = SGDisposeDeviceList(L.grabber, devlist)) != noErr) { ERROR("SGDisposeDeviceList: ", ret); return -1; }

        /* get resolution */
        ImageDescriptionHandle imgdesc = (ImageDescriptionHandle)NewHandle(0);
        if ((ret = SGGetSrcVideoBounds(L.channel, &L.rect)) != noErr) { ERROR("SGGetSrcVideoBounds: ", ret); return -1; }
        if ((ret = QTNewGWorld(&L.gworld, k32ARGBPixelFormat, &L.rect, 0, 0, 0)) != noErr) { ERROR("QTNewGworld: ", ret); return -1; }
        if ((ret = SGSetGWorld(L.grabber, L.gworld, 0)) != noErr) { ERROR("SGSetGWorld: ", ret); return -1; }
        if ((ret = SGSetChannelBounds(L.channel, &L.rect)) != noErr) { ERROR("SGSetChannelBounds: ", ret); return -1; }
        if ((ret = SGSetChannelUsage(L.channel, seqGrabRecord)) != noErr) { ERROR("SGSetChannelUsage: ", ret); return -1; }
        if ((ret = SGStartRecord(L.grabber)) != noErr) { ERROR("SGStartRecord: ", ret); return -1; }
        if ((ret = SGGetChannelSampleDescription(L.channel, (Handle)imgdesc)) != noErr) { ERROR("SGSetGWorld: ", ret); return -1; }
        if ((ret = SGStop(L.grabber)) != noErr) { ERROR("SGStop: ", ret); return -1; }
        L.rect.right = FLAGS_camera_image_width = (*imgdesc)->width;
        L.rect.bottom = FLAGS_camera_image_height = (*imgdesc)->height;
        DisposeHandle((Handle)imgdesc);
        GWorldPtr oldgworld = L.gworld;
        if ((ret = QTNewGWorld(&L.gworld, k32ARGBPixelFormat, &L.rect, 0, 0, 0)) != noErr) { ERROR("QTNewGworld: ", ret); return -1; }
        if ((ret = SGSetGWorld(L.grabber, L.gworld, 0)) != noErr) { ERROR("SGSetGWorld: ", ret); return -1; }
        if ((ret = SGSetChannelBounds(L.channel, &L.rect)) != noErr) { ERROR("SGSetChannelBounds: ", ret); return -1; }
        DisposeGWorld(oldgworld);
        image_linesize = FLAGS_camera_image_width * image_depth;
        conv.open(FLAGS_camera_image_width, FLAGS_camera_image_height, Pixel::BGR32,
                  FLAGS_camera_image_width, FLAGS_camera_image_height, image_format);

        /* start */
        L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_width*FLAGS_camera_image_height*image_depth);
        L.pixmap = GetGWorldPixMap(L.gworld);
        LockPixels(L.pixmap);
        if ((ret = SGSetDataProc(L.grabber, NewSGDataUPP(SGDataProcL), (long)this)) != noErr) { ERROR("SGSetDataProc: ", ret); return -1; }
        if ((ret = SGStartRecord(L.grabber) )!= noErr) { ERROR("SGSartRecord: ", ret); return -1; }
        if (!thread.Start()) return -1;
        return 0;
    }

    static OSErr SGDataProcL(SGChannel chan, Ptr buf, long len, long *, long, TimeValue, short, long opaque) { QuickTimeCamera *cam=(QuickTimeCamera*)opaque; return cam->SGDataProc(chan, buf, len, &cam->L); }
    static OSErr SGDataProcR(SGChannel chan, Ptr buf, long len, long *, long, TimeValue, short, long opaque) { QuickTimeCamera *cam=(QuickTimeCamera*)opaque; return cam->SGDataProc(chan, buf, len, &cam->R); }
    OSErr SGDataProc(SGChannel chan, Ptr buf, long len, Camera *cam) {
        CodecFlags codecflags; int ret;
        if (!cam->seq) {
            ImageDescriptionHandle desc = (ImageDescriptionHandle)NewHandle(0);
            if ((ret = SGGetChannelSampleDescription(chan, (Handle)desc)) != noErr) { ERROR("SGsFtChannelSampleDescription: ", ret); return -1; }
            Rect rect; rect.top=rect.left=0; rect.right=(*desc)->width; rect.bottom=(*desc)->height;
            MatrixRecord mat; RectMatrix(&mat, &rect, &cam->rect);
            if ((ret = DecompressSequenceBegin(&cam->seq, desc, cam->gworld, 0, &cam->rect, &mat, srcCopy, 0, 0, codecNormalQuality, bestSpeedCodec)) != noErr) { ERROR("DecompressSequenceBegin: ", ret); return -1; }
            DisposeHandle((Handle)desc);
        }
        if ((ret = DecompressSequenceFrameS(cam->seq, buf, len, 0, &codecflags, 0)) != noErr) { ERROR("DecompressSequenceFrame: ", ret); return -1; }

        /* write */
        char *in = (char*)(GetPixBaseAddr(cam->pixmap)+1);
        char *out = (char*)cam->frames->write(RingBuf::Peek | RingBuf::Stamp);
        int inlinesize = GetPixRowBytes(L.pixmap);
        conv.Resample(in, inlinesize, out, image_linesize);

        /* commit */
        {
            ScopedMutex ML(lock);
            cam->frames->write();
            frames_read++;
        }
        return noErr;
    }

    static int threadthunk(void *p) { return ((QuickTimeCamera*)p)->threadproc(); }
    int threadproc() {
        while (app->run) {
            if (L.grabber) SGIdle(L.grabber);
            if (R.grabber) SGIdle(R.grabber);
            Msleep(1);
        }
    }

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.frames) L.next = L.frames->ring.back;
                if (R.frames) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        image = (unsigned char*)L.frames->read(-1, L.next);
        image_timestamp = L.frames->readtimestamp(-1, L.next);
        return 1;
    }

    int free() {
        if (thread.started) thread.Wait();
        if (L.grabber) {
            SGStop(L.grabber);
            CloseComponent(L.grabber);
        }
        if (L.seq) CDSequenceEnd(L.seq);
        if (L.gworld) DisposeGWorld(L.gworld);
        return 0;
    }
};
#endif /* LFL_QUICKTIME_CAMERA */

#ifdef LFL_AVCAPTURE_CAMERA
struct AVCaptureCamera : public CameraInterface {
    Mutex lock; 

    struct Camera {
        RingBuf *frames;
        int next;
        Camera() : frames(0), next(0) {}
    } L, R;

    virtual ~AVCaptureCamera() {}
	AVCaptureCamera() {} 

    int init() {
        AVCaptureInit(FLAGS_camera_fps);
        /* hardcoded for now per AVCaptureSessionPresetLow */
        FLAGS_camera_image_height = 144;
        FLAGS_camera_image_width = 192;

        image_format = Pixel::BGR32;
        image_linesize = FLAGS_camera_image_width*4;
        return 0;
    }

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.frames) L.next = L.frames->ring.back;
                if (R.frames) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        image = (unsigned char*)L.frames->read(-1, L.next);
        image_timestamp = L.frames->readtimestamp(-1, L.next);

        return 1;
    }

    int free() {
        AVCaptureShutdown();
        return 0;
    }

    static void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
        AVCaptureCamera *camera = (AVCaptureCamera*)::camera;

        if (!camera->L.frames) 
            camera->L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, imageSize);

        memcpy(camera->L.frames->write(RingBuf::Peek | RingBuf::Stamp), imageData, imageSize);

        /* commit */  
        {
            ScopedMutex ML(camera->lock);
            camera->L.frames->write();
            camera->L.frames_read++;
        } 
    }
};

extern "C" {
    void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
        AVCaptureCamera::UpdateFrame(imageData, width, height, imageSize);
    }
};
#endif /* LFL_AVCAPTURE_CAMERA */

#ifdef LFL_QTKIT_CAMERA
struct QTKitCamera : public CameraInterface {
    Mutex lock; 

    struct Camera {
        RingBuf *frames;
        int next;
        Camera() : frames(0), next(0) {}
    } L, R;

    virtual ~QTKitCamera() {}
	QTKitCamera() {} 

    int init() {
        QTKitInit(FLAGS_camera_fps);
        return 0;
    }

    int frame() {
        bool new_frame = false;
        {
            ScopedMutex ML(lock);
            if (frames_read > last_frames_read) {

                /* frame align stream handles */
                if (L.frames) L.next = L.frames->ring.back;
                if (R.frames) R.next = R.frames->ring.back;

                new_frame = true;
            }
            last_frames_read = frames_read;
        }
        if (!new_frame) return 0;

        image = (unsigned char*)L.frames->read(-1, L.next);
        image_timestamp = L.frames->readtimestamp(-1, L.next);

        return 1;
    }

    int free() {
        QTKitShutdown();
        return 0;
    }

    static void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
        QTKitCamera *camera = (QTKitCamera*)app->camera.camera;

        if (!camera->L.frames) {
            camera->L.frames = new RingBuf(FLAGS_camera_fps, FLAGS_camera_fps, imageSize);
            FLAGS_camera_image_width = width;
            FLAGS_camera_image_height = height;
            camera->image_format = Pixel::RGB32;
            camera->image_linesize = FLAGS_camera_image_width*4;
        }

        memcpy(camera->L.frames->write(RingBuf::Peek | RingBuf::Stamp), imageData, imageSize);

        /* commit */  
        {
            ScopedMutex ML(camera->lock);
            camera->L.frames->write();
            camera->frames_read++;
        } 
    }
};

extern "C" {
    void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
        QTKitCamera::UpdateFrame(imageData, width, height, imageSize);
    }
};
#endif /* LFL_QTKIT_CAMERA */

int Camera::Init() {

#ifdef LFL_OPENCV_CAMERA
    camera = new OpenCvCamera();
#endif

#ifdef LFL_FFMPEG_CAMERA
    camera = new FFmpegCamera();
#endif

#ifdef LFL_DSVL_CAMERA
    camera = new DsvlCamera();
#endif

#ifdef LFL_DIRECTSHOW_CAMERA
    camera = new DirectShowCamera();
#endif

#ifdef LFL_QUICKTIME_CAMERA
    camera = new QuickTimeCamera();
#endif

#ifdef LFL_AVCAPTURE_CAMERA
	camera = new AVCaptureCamera();
#endif

#ifdef LFL_QTKIT_CAMERA
	camera = new QTKitCamera();
#endif

    int ret = 0;
    if (camera) ret = camera->init();
    else FLAGS_lfapp_camera = 0;
    
    if (!FLAGS_lfapp_camera) INFO("no camera found");

    return ret;
}

int Camera::Frame(unsigned t) { return camera->frame(); }

int Camera::Free() {
    int ret = camera->free();
    delete camera;
    camera = 0;
    return ret;
}

}; // namespace LFL
